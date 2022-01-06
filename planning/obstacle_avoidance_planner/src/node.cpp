// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "obstacle_avoidance_planner/node.hpp"

#include "obstacle_avoidance_planner/cv_utils.hpp"
#include "obstacle_avoidance_planner/debug_visualization.hpp"
#include "obstacle_avoidance_planner/utils.hpp"
#include "rclcpp/time.hpp"
#include "tf2/utils.h"
#include "tier4_autoware_utils/ros/update_param.hpp"
#include "tier4_autoware_utils/trajectory/tmp_conversion.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{
template <typename T1, typename T2>
size_t searchExtendedZeroVelocityIndex(
  const std::vector<T1> & fine_points, const std::vector<T2> & points)
{
  const auto opt_zero_vel_idx = tier4_autoware_utils::searchZeroVelocityIndex(points);
  const size_t zero_vel_idx = opt_zero_vel_idx ? opt_zero_vel_idx.get() : points.size() - 1;
  return tier4_autoware_utils::findNearestIndex(fine_points, points.at(zero_vel_idx).pose.position);
}

bool isPathShapeChanged(
  const geometry_msgs::msg::Pose & ego_pose,
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::unique_ptr<std::vector<autoware_auto_planning_msgs::msg::PathPoint>> &
    prev_path_points,
  const double delta_yaw_threshold, const double distance_for_path_shape_change_detection)
{
  if (!prev_path_points) {
    return false;
  }

  // truncate prev points from ego pose to fixed end points
  const auto opt_prev_begin_idx = tier4_autoware_utils::findNearestIndex(
    *prev_path_points, ego_pose, std::numeric_limits<double>::max(), delta_yaw_threshold);
  const size_t prev_begin_idx = opt_prev_begin_idx ? *opt_prev_begin_idx : 0;
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> truncated_prev_points{
    prev_path_points->begin() + prev_begin_idx, prev_path_points->end()};

  // truncate points from ego pose to fixed end points
  const auto opt_begin_idx = tier4_autoware_utils::findNearestIndex(
    path_points, ego_pose, std::numeric_limits<double>::max(), delta_yaw_threshold);
  const size_t begin_idx = opt_begin_idx ? *opt_begin_idx : 0;
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> truncated_points{
    path_points.begin() + begin_idx, path_points.end()};

  // guard for lateral offset
  if (truncated_prev_points.size() < 2 || truncated_points.size() < 2) {
    return false;
  }

  // calculate lateral deviations between truncated path_points and prev_path_points
  for (const auto & prev_point : truncated_prev_points) {
    const double dist =
      tier4_autoware_utils::calcLateralOffset(truncated_points, prev_point.pose.position);
    if (dist > distance_for_path_shape_change_detection) {
      return true;
    }
  }

  return false;
}

bool hasValidNearestPointFromEgo(
  const geometry_msgs::msg::Pose & ego_pose, const Trajectories & trajs,
  const TrajectoryParam & traj_param)
{
  const auto traj = trajs.model_predictive_trajectory;
  const auto interpolated_points =
    interpolation_utils::getInterpolatedPoints(traj, traj_param.delta_arc_length_for_trajectory);

  const auto interpolated_poses_with_yaw =
    points_utils::convertToPosesWithYawEstimation(interpolated_points);
  const auto opt_nearest_idx = tier4_autoware_utils::findNearestIndex(
    interpolated_poses_with_yaw, ego_pose, traj_param.delta_dist_threshold_for_closest_point,
    traj_param.delta_yaw_threshold_for_closest_point);

  if (!opt_nearest_idx) {
    return false;
  }
  return true;
}
}  // namespace

ObstacleAvoidancePlanner::ObstacleAvoidancePlanner(const rclcpp::NodeOptions & node_options)
: Node("obstacle_avoidance_planner", node_options), logger_ros_clock_(RCL_ROS_TIME)
{
  rclcpp::Clock::SharedPtr clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  // qos
  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();

  // publisher to other nodes
  traj_pub_ = create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/output/path", 1);
  avoiding_traj_pub_ = create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "/planning/scenario_planning/lane_driving/obstacle_avoidance_candidate_trajectory",
    durable_qos);
  is_avoidance_possible_pub_ = create_publisher<tier4_planning_msgs::msg::IsAvoidancePossible>(
    "/planning/scenario_planning/lane_driving/obstacle_avoidance_ready", durable_qos);

  // debug publisher
  debug_eb_traj_pub_ = create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/debug/eb_trajectory", durable_qos);
  debug_extended_fixed_traj_pub_ = create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/debug/extended_fixed_traj", 1);
  debug_extended_non_fixed_traj_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
      "~/debug/extended_non_fixed_traj", 1);
  debug_mpt_fixed_traj_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/debug/mpt_fixed_traj", 1);
  debug_mpt_ref_traj_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/debug/mpt_ref_traj", 1);
  debug_mpt_traj_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/debug/mpt_traj", 1);
  debug_markers_pub_ =
    create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/marker", durable_qos);
  debug_clearance_map_pub_ =
    create_publisher<nav_msgs::msg::OccupancyGrid>("~/debug/clearance_map", durable_qos);
  debug_object_clearance_map_pub_ =
    create_publisher<nav_msgs::msg::OccupancyGrid>("~/debug/object_clearance_map", durable_qos);
  debug_area_with_objects_pub_ =
    create_publisher<nav_msgs::msg::OccupancyGrid>("~/debug/area_with_objects", durable_qos);
  debug_msg_pub_ = create_publisher<tier4_debug_msgs::msg::StringStamped>("~/debug/debug_msg", 1);

  // subscriber
  path_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Path>(
    "~/input/path", rclcpp::QoS{1},
    std::bind(&ObstacleAvoidancePlanner::pathCallback, this, std::placeholders::_1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/localization/kinematic_state", rclcpp::QoS{1},
    std::bind(&ObstacleAvoidancePlanner::odomCallback, this, std::placeholders::_1));
  objects_sub_ = create_subscription<autoware_auto_perception_msgs::msg::PredictedObjects>(
    "~/input/objects", rclcpp::QoS{10},
    std::bind(&ObstacleAvoidancePlanner::objectsCallback, this, std::placeholders::_1));
  is_avoidance_sub_ = create_subscription<tier4_planning_msgs::msg::EnableAvoidance>(
    "/planning/scenario_planning/lane_driving/obstacle_avoidance_approval", rclcpp::QoS{10},
    std::bind(&ObstacleAvoidancePlanner::enableAvoidanceCallback, this, std::placeholders::_1));

  {  // vehicle param
    vehicle_param_ptr_ = std::make_unique<VehicleParam>();
    const auto vehicle_info = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();
    vehicle_param_ptr_->width = vehicle_info.vehicle_width_m;
    vehicle_param_ptr_->length = vehicle_info.vehicle_length_m;
    vehicle_param_ptr_->wheelbase = vehicle_info.wheel_base_m;
    vehicle_param_ptr_->rear_overhang = vehicle_info.rear_overhang_m;
    vehicle_param_ptr_->front_overhang = vehicle_info.front_overhang_m;
  }

  {  // option parameter
    is_publishing_clearance_map_ = declare_parameter<bool>("option.is_publishing_clearance_map");
    is_publishing_object_clearance_map_ =
      declare_parameter<bool>("option.is_publishing_object_clearance_map");
    is_publishing_area_with_objects_ =
      declare_parameter<bool>("option.is_publishing_area_with_objects");

    is_showing_debug_info_ = declare_parameter<bool>("option.is_showing_debug_info");
    is_showing_calculation_time_ = declare_parameter<bool>("option.is_showing_calculation_time");

    is_stopping_if_outside_drivable_area_ =
      declare_parameter<bool>("option.is_stopping_if_outside_drivable_area");
    is_using_vehicle_config_ = declare_parameter<bool>("option.is_using_vehicle_config");
    enable_avoidance_ = declare_parameter<bool>("option.enable_avoidance");
    visualize_sampling_num_ = declare_parameter<int>("option.visualize_sampling_num");
  }

  {  // trajectory parameter
    traj_param_ptr_ = std::make_unique<TrajectoryParam>();

    // common
    traj_param_ptr_->num_sampling_points = declare_parameter<int>("common.num_sampling_points");
    traj_param_ptr_->trajectory_length = declare_parameter<double>("common.trajectory_length");
    traj_param_ptr_->forward_fixing_distance =
      declare_parameter<double>("common.forward_fixing_distance");
    traj_param_ptr_->backward_fixing_distance =
      declare_parameter<double>("common.backward_fixing_distance");
    traj_param_ptr_->delta_arc_length_for_trajectory =
      declare_parameter<double>("common.delta_arc_length_for_trajectory");

    traj_param_ptr_->delta_dist_threshold_for_closest_point =
      declare_parameter<double>("common.delta_dist_threshold_for_closest_point");
    traj_param_ptr_->delta_yaw_threshold_for_closest_point =
      declare_parameter<double>("common.delta_yaw_threshold_for_closest_point");
    traj_param_ptr_->delta_yaw_threshold_for_straight =
      declare_parameter<double>("common.delta_yaw_threshold_for_straight");

    traj_param_ptr_->num_fix_points_for_extending =
      declare_parameter<int>("common.num_fix_points_for_extending");
    traj_param_ptr_->max_dist_for_extending_end_point =
      declare_parameter<double>("common.max_dist_for_extending_end_point");

    // object
    traj_param_ptr_->max_avoiding_ego_velocity_ms =
      declare_parameter<double>("object.max_avoiding_ego_velocity_ms");
    traj_param_ptr_->max_avoiding_objects_velocity_ms =
      declare_parameter<double>("object.max_avoiding_objects_velocity_ms");
    traj_param_ptr_->center_line_width = declare_parameter<double>("object.center_line_width");
    traj_param_ptr_->is_avoiding_unknown =
      declare_parameter<bool>("object.avoiding_object_type.unknown", true);
    traj_param_ptr_->is_avoiding_car =
      declare_parameter<bool>("object.avoiding_object_type.car", true);
    traj_param_ptr_->is_avoiding_truck =
      declare_parameter<bool>("object.avoiding_object_type.truck", true);
    traj_param_ptr_->is_avoiding_bus =
      declare_parameter<bool>("object.avoiding_object_type.bus", true);
    traj_param_ptr_->is_avoiding_bicycle =
      declare_parameter<bool>("object.avoiding_object_type.bicycle", true);
    traj_param_ptr_->is_avoiding_motorbike =
      declare_parameter<bool>("object.avoiding_object_type.motorbike", true);
    traj_param_ptr_->is_avoiding_pedestrian =
      declare_parameter<bool>("object.avoiding_object_type.pedestrian", true);
    traj_param_ptr_->is_avoiding_animal =
      declare_parameter<bool>("object.avoiding_object_type.animal", true);
  }

  {  // elastic band parameter
    eb_param_ptr_ = std::make_unique<EBParam>();

    // option
    eb_param_ptr_->is_getting_constraints_close2path_points =
      declare_parameter<bool>("eb.option.is_getting_constraints_close2path_points");

    // common
    eb_param_ptr_->num_joint_buffer_points =
      declare_parameter<int>("eb.common.num_joint_buffer_points");
    eb_param_ptr_->num_joint_buffer_points_for_extending =
      declare_parameter<int>("eb.common.num_joint_buffer_points_for_extending");
    eb_param_ptr_->num_offset_for_begin_idx =
      declare_parameter<int>("eb.common.num_offset_for_begin_idx");
    eb_param_ptr_->delta_arc_length_for_optimization =
      declare_parameter<double>("eb.common.delta_arc_length_for_optimization");

    // clearance
    eb_param_ptr_->clearance_for_straight_line =
      declare_parameter<double>("eb.clearance.clearance_for_straight_line");
    eb_param_ptr_->clearance_for_joint =
      declare_parameter<double>("eb.clearance.clearance_for_joint");
    eb_param_ptr_->range_for_extend_joint =
      declare_parameter<double>("eb.clearance.range_for_extend_joint");
    eb_param_ptr_->clearance_for_only_smoothing =
      declare_parameter<double>("eb.clearance.clearance_for_only_smoothing");
    eb_param_ptr_->clearance_from_object_for_straight =
      declare_parameter<double>("eb.clearance.clearance_from_object_for_straight");
    eb_param_ptr_->soft_clearance_from_road =
      declare_parameter<double>("eb.clearance.clearance_from_road");
    eb_param_ptr_->clearance_from_object =
      declare_parameter<double>("eb.clearance.clearance_from_object");
    eb_param_ptr_->min_object_clearance_for_joint =
      declare_parameter<double>("eb.clearance.min_object_clearance_for_joint");

    // constrain
    eb_param_ptr_->max_x_constrain_search_range =
      declare_parameter("eb.constrain.max_x_constrain_search_range", 0.4);
    eb_param_ptr_->coef_x_constrain_search_resolution =
      declare_parameter("eb.constrain.coef_x_constrain_search_resolution", 1.0);
    eb_param_ptr_->coef_y_constrain_search_resolution =
      declare_parameter("eb.constrain.coef_y_constrain_search_resolution", 0.5);
    eb_param_ptr_->keep_space_shape_x = declare_parameter("eb.constrain.keep_space_shape_x", 3.0);
    eb_param_ptr_->keep_space_shape_y = declare_parameter("eb.constrain.keep_space_shape_y", 2.0);
    eb_param_ptr_->max_lon_space_for_driveable_constraint =
      declare_parameter("eb.constrain.max_lon_space_for_driveable_constraint", 0.5);

    // qp
    eb_param_ptr_->qp_param.max_iteration = declare_parameter<int>("eb.qp.max_iteration");
    eb_param_ptr_->qp_param.eps_abs = declare_parameter<double>("eb.qp.eps_abs");
    eb_param_ptr_->qp_param.eps_rel = declare_parameter<double>("eb.qp.eps_rel");
    eb_param_ptr_->qp_param.eps_abs_for_extending =
      declare_parameter<double>("eb.qp.eps_abs_for_extending");
    eb_param_ptr_->qp_param.eps_rel_for_extending =
      declare_parameter<double>("eb.qp.eps_rel_for_extending");
    eb_param_ptr_->qp_param.eps_abs_for_visualizing =
      declare_parameter<double>("eb.qp.eps_abs_for_visualizing");
    eb_param_ptr_->qp_param.eps_rel_for_visualizing =
      declare_parameter<double>("eb.qp.eps_rel_for_visualizing");

    // other
    eb_param_ptr_->clearance_for_fixing = 0.0;
    eb_param_ptr_->min_object_clearance_for_deceleration =
      eb_param_ptr_->clearance_from_object + eb_param_ptr_->keep_space_shape_y * 0.5;
  }

  {  // mpt param
    mpt_param_ptr_ = std::make_unique<MPTParam>();

    // option
    mpt_param_ptr_->l_inf_norm = declare_parameter<bool>("mpt.option.l_inf_norm");
    mpt_param_ptr_->two_step_soft_constraint =
      declare_parameter<bool>("mpt.option.two_step_soft_constraint");
    mpt_param_ptr_->plan_from_ego = declare_parameter<bool>("mpt.option.plan_from_ego");
    mpt_param_ptr_->avoiding_constraint_type =
      declare_parameter<int>("mpt.option.avoiding_constraint_type");
    mpt_param_ptr_->is_hard_fixing_terminal_point =
      declare_parameter<bool>("mpt.option.is_hard_fixing_terminal_point");

    // common
    mpt_param_ptr_->num_curvature_sampling_points =
      declare_parameter<int>("mpt.common.num_curvature_sampling_points");

    mpt_param_ptr_->delta_arc_length_for_mpt_points =
      declare_parameter<double>("mpt.common.delta_arc_length_for_mpt_points");
    mpt_param_ptr_->forward_fixing_mpt_min_distance =
      declare_parameter<double>("mpt.common.forward_fixing_mpt_min_distance");
    mpt_param_ptr_->forward_fixing_mpt_time =
      declare_parameter<double>("mpt.common.forward_fixing_mpt_time");

    // kinematics
    mpt_param_ptr_->max_steer_rad =
      declare_parameter<double>("mpt.kinematics.max_steer_deg") * M_PI / 180.0;
    mpt_param_ptr_->steer_tau = declare_parameter<double>("mpt.kinematics.steer_tau");

    mpt_param_ptr_->optimization_center_offset = declare_parameter<double>(
      "mpt.kinematics.optimization_center_offset", vehicle_param_ptr_->wheelbase / 2.0);
    mpt_param_ptr_->avoiding_circle_offsets = declare_parameter<std::vector<double>>(
      "mpt.kinematics.avoiding_circle_offsets", std::vector<double>());
    mpt_param_ptr_->avoiding_circle_radius =
      declare_parameter<double>("mpt.kinematics.avoiding_circle_radius", -1.0);

    // clearance
    mpt_param_ptr_->hard_clearance_from_road =
      declare_parameter<double>("mpt.clearance.hard_clearance_from_road");
    mpt_param_ptr_->soft_clearance_from_road =
      declare_parameter<double>("mpt.clearance.soft_clearance_from_road");
    mpt_param_ptr_->soft_second_clearance_from_road =
      declare_parameter<double>("mpt.clearance.soft_second_clearance_from_road");
    mpt_param_ptr_->extra_desired_clearance_from_road =
      declare_parameter<double>("mpt.clearance.extra_desired_clearance_from_road");
    mpt_param_ptr_->clearance_from_object =
      declare_parameter<double>("mpt.clearance.clearance_from_object");

    // weight
    mpt_param_ptr_->soft_avoidance_weight =
      declare_parameter<double>("mpt.weight.soft_avoidance_weight");
    mpt_param_ptr_->soft_second_avoidance_weight =
      declare_parameter<double>("mpt.weight.soft_second_avoidance_weight");

    mpt_param_ptr_->lat_error_weight = declare_parameter<double>("mpt.weight.lat_error_weight");
    mpt_param_ptr_->yaw_error_weight = declare_parameter<double>("mpt.weight.yaw_error_weight");
    mpt_param_ptr_->yaw_error_rate_weight =
      declare_parameter<double>("mpt.weight.yaw_error_rate_weight");
    mpt_param_ptr_->steer_input_weight = declare_parameter<double>("mpt.weight.steer_input_weight");
    mpt_param_ptr_->steer_rate_weight = declare_parameter<double>("mpt.weight.steer_rate_weight");
    mpt_param_ptr_->steer_acc_weight = declare_parameter<double>("mpt.weight.steer_acc_weight");

    mpt_param_ptr_->obstacle_avoid_lat_error_weight =
      declare_parameter<double>("mpt.weight.obstacle_avoid_lat_error_weight");
    mpt_param_ptr_->obstacle_avoid_yaw_error_weight =
      declare_parameter<double>("mpt.weight.obstacle_avoid_yaw_error_weight");
    mpt_param_ptr_->obstacle_avoid_steer_input_weight =
      declare_parameter<double>("mpt.weight.obstacle_avoid_steer_input_weight");
    mpt_param_ptr_->near_objects_length =
      declare_parameter<double>("mpt.weight.near_objects_length");

    mpt_param_ptr_->terminal_lat_error_weight =
      declare_parameter<double>("mpt.weight.terminal_lat_error_weight");
    mpt_param_ptr_->terminal_yaw_error_weight =
      declare_parameter<double>("mpt.weight.terminal_yaw_error_weight");
    mpt_param_ptr_->terminal_path_lat_error_weight =
      declare_parameter<double>("mpt.weight.terminal_path_lat_error_weight");
    mpt_param_ptr_->terminal_path_yaw_error_weight =
      declare_parameter<double>("mpt.weight.terminal_path_yaw_error_weight");

    // update avoiding circle if required
    if (
      mpt_param_ptr_->avoiding_circle_offsets.empty() ||
      mpt_param_ptr_->avoiding_circle_radius < 0) {
      constexpr size_t avoiding_circle_num = 4;
      mpt_param_ptr_->avoiding_circle_radius = std::hypot(
        vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0,
        vehicle_param_ptr_->width / 2.0);

      mpt_param_ptr_->avoiding_circle_offsets.clear();
      for (size_t i = 0; i < avoiding_circle_num; ++i) {
        mpt_param_ptr_->avoiding_circle_offsets.push_back(
          vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0 *
            (1 + 2 * i) -
          vehicle_param_ptr_->rear_overhang);
      }
    }
  }

  {  // replan
    reset_prev_info_ = declare_parameter<bool>("replan.reset_prev_info");
    use_footprint_for_drivability_ =
      declare_parameter<bool>("replan.use_footprint_for_drivability");
    min_ego_moving_dist_for_replan_ =
      declare_parameter<double>("replan.min_ego_moving_dist_for_replan");
    min_delta_time_sec_for_replan_ =
      declare_parameter<double>("replan.min_delta_time_sec_for_replan");
    distance_for_path_shape_change_detection_ =
      declare_parameter<double>("replan.distance_for_path_shape_change_detection");
  }

  if (is_using_vehicle_config_) {
    traj_param_ptr_->center_line_width = vehicle_param_ptr_->width;
    eb_param_ptr_->keep_space_shape_y = vehicle_param_ptr_->width;
  }

  in_objects_ptr_ = std::make_unique<autoware_auto_perception_msgs::msg::PredictedObjects>();

  // set parameter callback
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&ObstacleAvoidancePlanner::paramCallback, this, std::placeholders::_1));

  initializeParam();

  self_pose_listener_.waitForFirstPose();
}

rcl_interfaces::msg::SetParametersResult ObstacleAvoidancePlanner::paramCallback(
  const std::vector<rclcpp::Parameter> & parameters)
{
  using tier4_autoware_utils::updateParam;

  {  // option parameter
    updateParam<bool>(
      parameters, "option.is_publishing_clearance_map", is_publishing_clearance_map_);
    updateParam<bool>(
      parameters, "option.is_publishing_object_clearance_map", is_publishing_object_clearance_map_);
    updateParam<bool>(
      parameters, "option.is_publishing_area_with_objects", is_publishing_area_with_objects_);

    updateParam<bool>(parameters, "option.is_showing_debug_info", is_showing_debug_info_);
    updateParam<bool>(
      parameters, "option.is_showing_calculation_time", is_showing_calculation_time_);

    updateParam<bool>(
      parameters, "option.is_stopping_if_outside_drivable_area",
      is_stopping_if_outside_drivable_area_);
    updateParam<bool>(parameters, "option.is_using_vehicle_config", is_using_vehicle_config_);
    updateParam<bool>(parameters, "option.enable_avoidance", enable_avoidance_);
    updateParam<int>(parameters, "option.visualize_sampling_num", visualize_sampling_num_);
  }

  {  // trajectory parameter
    // common
    updateParam<int>(
      parameters, "common.num_sampling_points", traj_param_ptr_->num_sampling_points);
    updateParam<double>(parameters, "common.trajectory_length", traj_param_ptr_->trajectory_length);
    updateParam<double>(
      parameters, "common.forward_fixing_distance", traj_param_ptr_->forward_fixing_distance);
    updateParam<double>(
      parameters, "common.backward_fixing_distance", traj_param_ptr_->backward_fixing_distance);
    updateParam<double>(
      parameters, "common.delta_arc_length_for_trajectory",
      traj_param_ptr_->delta_arc_length_for_trajectory);

    updateParam<double>(
      parameters, "common.delta_dist_threshold_for_closest_point",
      traj_param_ptr_->delta_dist_threshold_for_closest_point);
    updateParam<double>(
      parameters, "common.delta_yaw_threshold_for_closest_point",
      traj_param_ptr_->delta_yaw_threshold_for_closest_point);
    updateParam<double>(
      parameters, "common.delta_yaw_threshold_for_straight",
      traj_param_ptr_->delta_yaw_threshold_for_straight);

    updateParam<int>(
      parameters, "common.num_fix_points_for_extending",
      traj_param_ptr_->num_fix_points_for_extending);
    updateParam<double>(
      parameters, "common.max_dist_for_extending_end_point",
      traj_param_ptr_->max_dist_for_extending_end_point);

    // object
    updateParam<double>(
      parameters, "object.max_avoiding_ego_velocity_ms",
      traj_param_ptr_->max_avoiding_ego_velocity_ms);
    updateParam<double>(
      parameters, "object.max_avoiding_objects_velocity_ms",
      traj_param_ptr_->max_avoiding_objects_velocity_ms);
    updateParam<double>(parameters, "object.center_line_width", traj_param_ptr_->center_line_width);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.unknown", traj_param_ptr_->is_avoiding_unknown);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.car", traj_param_ptr_->is_avoiding_car);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.truck", traj_param_ptr_->is_avoiding_truck);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.bus", traj_param_ptr_->is_avoiding_bus);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.bicycle", traj_param_ptr_->is_avoiding_bicycle);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.motorbike", traj_param_ptr_->is_avoiding_motorbike);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.pedestrian",
      traj_param_ptr_->is_avoiding_pedestrian);
    updateParam<bool>(
      parameters, "object.avoiding_object_type.animal", traj_param_ptr_->is_avoiding_animal);
  }

  {  // elastic band parameter
    // optionn
    updateParam<bool>(
      parameters, "eb.option.is_getting_constraints_close2path_points",
      eb_param_ptr_->is_getting_constraints_close2path_points);

    // common
    updateParam<int>(
      parameters, "eb.common.num_joint_buffer_points", eb_param_ptr_->num_joint_buffer_points);
    updateParam<int>(
      parameters, "eb.common.num_joint_buffer_points_for_extending",
      eb_param_ptr_->num_joint_buffer_points_for_extending);
    updateParam<int>(
      parameters, "eb.common.num_offset_for_begin_idx", eb_param_ptr_->num_offset_for_begin_idx);
    updateParam<double>(
      parameters, "eb.common.delta_arc_length_for_optimization",
      eb_param_ptr_->delta_arc_length_for_optimization);

    // clearance
    updateParam<double>(
      parameters, "eb.clearance.clearance_for_straight_line",
      eb_param_ptr_->clearance_for_straight_line);
    updateParam<double>(
      parameters, "eb.clearance.clearance_for_joint", eb_param_ptr_->clearance_for_joint);
    updateParam<double>(
      parameters, "eb.clearance.range_for_extend_joint", eb_param_ptr_->range_for_extend_joint);
    updateParam<double>(
      parameters, "eb.clearance.clearance_for_only_smoothing",
      eb_param_ptr_->clearance_for_only_smoothing);
    updateParam<double>(
      parameters, "eb.clearance.clearance_from_object_for_straight",
      eb_param_ptr_->clearance_from_object_for_straight);
    updateParam<double>(
      parameters, "eb.clearance.clearance_from_road", eb_param_ptr_->soft_clearance_from_road);
    updateParam<double>(
      parameters, "eb.clearance.clearance_from_object", eb_param_ptr_->clearance_from_object);
    updateParam<double>(
      parameters, "eb.clearance.min_object_clearance_for_joint",
      eb_param_ptr_->min_object_clearance_for_joint);

    // constrain
    updateParam(
      parameters, "eb.constrain.max_x_constrain_search_range",
      eb_param_ptr_->max_x_constrain_search_range);
    updateParam(
      parameters, "eb.constrain.coef_x_constrain_search_resolution",
      eb_param_ptr_->coef_x_constrain_search_resolution);
    updateParam(
      parameters, "eb.constrain.coef_y_constrain_search_resolution",
      eb_param_ptr_->coef_y_constrain_search_resolution);
    updateParam(parameters, "eb.constrain.keep_space_shape_x", eb_param_ptr_->keep_space_shape_x);
    updateParam(parameters, "eb.constrain.keep_space_shape_y", eb_param_ptr_->keep_space_shape_y);
    updateParam(
      parameters, "eb.constrain.max_lon_space_for_driveable_constraint",
      eb_param_ptr_->max_lon_space_for_driveable_constraint);

    // qp
    updateParam<int>(parameters, "eb.qp.max_iteration", eb_param_ptr_->qp_param.max_iteration);
    updateParam<double>(parameters, "eb.qp.eps_abs", eb_param_ptr_->qp_param.eps_abs);
    updateParam<double>(parameters, "eb.qp.eps_rel", eb_param_ptr_->qp_param.eps_rel);
    updateParam<double>(
      parameters, "eb.qp.eps_abs_for_extending", eb_param_ptr_->qp_param.eps_abs_for_extending);
    updateParam<double>(
      parameters, "eb.qp.eps_rel_for_extending", eb_param_ptr_->qp_param.eps_rel_for_extending);
    updateParam<double>(
      parameters, "eb.qp.eps_abs_for_visualizing", eb_param_ptr_->qp_param.eps_abs_for_visualizing);
    updateParam<double>(
      parameters, "eb.qp.eps_rel_for_visualizing", eb_param_ptr_->qp_param.eps_rel_for_visualizing);

    // other
    /*
    eb_param_ptr_->clearance_for_fixing = 0.0;
    eb_param_ptr_->min_object_clearance_for_deceleration
      = eb_param_ptr_->clearance_from_object + eb_param_ptr_->keep_space_shape_y * 0.5;
    */
  }

  {  // mpt param
    // option
    updateParam<bool>(parameters, "mpt.option.l_inf_norm", mpt_param_ptr_->l_inf_norm);
    updateParam<bool>(
      parameters, "mpt.option.two_step_soft_constraint", mpt_param_ptr_->two_step_soft_constraint);
    updateParam<bool>(parameters, "mpt.option.plan_from_ego", mpt_param_ptr_->plan_from_ego);
    updateParam<int>(
      parameters, "mpt.option.avoiding_constraint_type", mpt_param_ptr_->avoiding_constraint_type);
    updateParam<bool>(
      parameters, "mpt.option.is_hard_fixing_terminal_point",
      mpt_param_ptr_->is_hard_fixing_terminal_point);

    // common
    updateParam<int>(
      parameters, "mpt.common.num_curvature_sampling_points",
      mpt_param_ptr_->num_curvature_sampling_points);

    updateParam<double>(
      parameters, "mpt.common.delta_arc_length_for_mpt_points",
      mpt_param_ptr_->delta_arc_length_for_mpt_points);
    updateParam<double>(
      parameters, "mpt.common.forward_fixing_mpt_min_distance",
      mpt_param_ptr_->forward_fixing_mpt_min_distance);
    updateParam<double>(
      parameters, "mpt.common.forward_fixing_mpt_time", mpt_param_ptr_->forward_fixing_mpt_time);

    // kinematics
    double max_steer_deg;
    updateParam<double>(parameters, "mpt.kinematics.max_steer_deg", max_steer_deg);
    mpt_param_ptr_->max_steer_rad = max_steer_deg * M_PI / 180.0;
    updateParam<double>(parameters, "mpt.kinematics.steer_tau", mpt_param_ptr_->steer_tau);

    updateParam<double>(
      parameters, "mpt.kinematics.optimization_center_offset",
      mpt_param_ptr_->optimization_center_offset);
    updateParam<std::vector<double>>(
      parameters, "mpt.kinematics.avoiding_circle_offsets",
      mpt_param_ptr_->avoiding_circle_offsets);
    updateParam<double>(
      parameters, "mpt.kinematics.avoiding_circle_radius", mpt_param_ptr_->avoiding_circle_radius);

    // clearance
    updateParam<double>(
      parameters, "mpt.clearance.hard_clearance_from_road",
      mpt_param_ptr_->hard_clearance_from_road);
    updateParam<double>(
      parameters, "mpt.clearance.soft_clearance_from_road",
      mpt_param_ptr_->soft_clearance_from_road);
    updateParam<double>(
      parameters, "mpt.clearance.soft_second_clearance_from_road",
      mpt_param_ptr_->soft_second_clearance_from_road);
    updateParam<double>(
      parameters, "mpt.clearance.extra_desired_clearance_from_road",
      mpt_param_ptr_->extra_desired_clearance_from_road);
    updateParam<double>(
      parameters, "mpt.clearance.clearance_from_object", mpt_param_ptr_->clearance_from_object);

    // weight
    updateParam<double>(
      parameters, "mpt.weight.soft_avoidance_weight", mpt_param_ptr_->soft_avoidance_weight);
    updateParam<double>(
      parameters, "mpt.weight.soft_second_avoidance_weight",
      mpt_param_ptr_->soft_second_avoidance_weight);

    updateParam<double>(
      parameters, "mpt.weight.lat_error_weight", mpt_param_ptr_->lat_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.yaw_error_weight", mpt_param_ptr_->yaw_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.yaw_error_rate_weight", mpt_param_ptr_->yaw_error_rate_weight);
    updateParam<double>(
      parameters, "mpt.weight.steer_input_weight", mpt_param_ptr_->steer_input_weight);
    updateParam<double>(
      parameters, "mpt.weight.steer_rate_weight", mpt_param_ptr_->steer_rate_weight);
    updateParam<double>(
      parameters, "mpt.weight.steer_acc_weight", mpt_param_ptr_->steer_acc_weight);

    updateParam<double>(
      parameters, "mpt.weight.obstacle_avoid_lat_error_weight",
      mpt_param_ptr_->obstacle_avoid_lat_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.obstacle_avoid_yaw_error_weight",
      mpt_param_ptr_->obstacle_avoid_yaw_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.obstacle_avoid_steer_input_weight",
      mpt_param_ptr_->obstacle_avoid_steer_input_weight);
    updateParam<double>(
      parameters, "mpt.weight.near_objects_length", mpt_param_ptr_->near_objects_length);

    updateParam<double>(
      parameters, "mpt.weight.terminal_lat_error_weight",
      mpt_param_ptr_->terminal_lat_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.terminal_yaw_error_weight",
      mpt_param_ptr_->terminal_yaw_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.terminal_path_lat_error_weight",
      mpt_param_ptr_->terminal_path_lat_error_weight);
    updateParam<double>(
      parameters, "mpt.weight.terminal_path_yaw_error_weight",
      mpt_param_ptr_->terminal_path_yaw_error_weight);

    // update avoiding circle if required
    /*
      if(mpt_param_ptr_->avoiding_circle_offsets.empty() || mpt_param_ptr_->avoiding_circle_radius <
      0) { constexpr size_t avoiding_circle_num  4; mpt_param_ptr_->avoiding_circle_radius
      std::hypot( vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0,
      vehicle_param_ptr_->width / 2.0);

      mpt_param_ptr_->avoiding_circle_offsets.clear();
      for (size_t i  0; i < avoiding_circle_num; ++i) {
      mpt_param_ptr_->avoiding_circle_offsets.push_back(
      vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0 * (1 + 2 * i) -
      vehicle_param_ptr_->rear_overhang);
      }
      }
    */
  }

  {  // replan
    updateParam<bool>(parameters, "replan.reset_prev_info", reset_prev_info_);
    updateParam<bool>(
      parameters, "replan.use_footprint_for_drivability", use_footprint_for_drivability_);
    updateParam<double>(
      parameters, "replan.min_ego_moving_dist_for_replan", min_ego_moving_dist_for_replan_);
    updateParam<double>(
      parameters, "replan.min_delta_time_sec_for_replan", min_delta_time_sec_for_replan_);
    updateParam<double>(
      parameters, "replan.distance_for_path_shape_change_detection",
      distance_for_path_shape_change_detection_);
  }

  initializeParam();

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void ObstacleAvoidancePlanner::initializeParam()
{
  RCLCPP_WARN(get_logger(), "[ObstacleAvoidancePlanner] Resetting");

  costmap_generator_ptr_ = std::make_unique<CostmapGenerator>();

  eb_path_optimizer_ptr_ = std::make_unique<EBPathOptimizer>(
    is_showing_debug_info_, *traj_param_ptr_, *eb_param_ptr_, *vehicle_param_ptr_);

  mpt_optimizer_ptr_ = std::make_unique<MPTOptimizer>(
    is_showing_debug_info_, *traj_param_ptr_, *vehicle_param_ptr_, *mpt_param_ptr_);

  prev_path_points_ptr_ = nullptr;
  prev_optimal_trajs_ptr_ = nullptr;
}

void ObstacleAvoidancePlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_twist_ptr_ = std::make_unique<geometry_msgs::msg::TwistStamped>();
  current_twist_ptr_->header = msg->header;
  current_twist_ptr_->twist = msg->twist.twist;
}

void ObstacleAvoidancePlanner::objectsCallback(
  const autoware_auto_perception_msgs::msg::PredictedObjects::SharedPtr msg)
{
  in_objects_ptr_ = std::make_unique<autoware_auto_perception_msgs::msg::PredictedObjects>(*msg);
}

void ObstacleAvoidancePlanner::enableAvoidanceCallback(
  const tier4_planning_msgs::msg::EnableAvoidance::SharedPtr msg)
{
  enable_avoidance_ = msg->enable_avoidance;
}

void ObstacleAvoidancePlanner::pathCallback(
  const autoware_auto_planning_msgs::msg::Path::SharedPtr msg)
{
  // std::lock_guard<std::mutex> lock(mutex_);
  current_ego_pose_ = self_pose_listener_.getCurrentPose()->pose;
  debug_data_ptr_ = std::make_shared<DebugData>();

  if (msg->points.empty() || msg->drivable_area.data.empty() || !current_twist_ptr_) {
    return;
  }

  const auto output_trajectory_msg = generateTrajectory(*msg);
  traj_pub_->publish(output_trajectory_msg);
}

autoware_auto_planning_msgs::msg::Trajectory ObstacleAvoidancePlanner::generateTrajectory(
  const autoware_auto_planning_msgs::msg::Path & path)
{
  // std::lock_guard<std::mutex> lock(mutex_);
  stop_watch_.tic(__func__);

  // generate optimized trajectory
  const auto optimized_traj_points = generateOptimizedTrajectory(path);

  // generate post processed trajectory
  const auto post_processed_traj_points =
    generatePostProcessedTrajectory(path.points, optimized_traj_points);

  // convert to output msg type
  auto output = tier4_autoware_utils::convertToTrajectory(post_processed_traj_points);
  output.header = path.header;

  // make prev variable
  prev_path_points_ptr_ =
    std::make_unique<std::vector<autoware_auto_planning_msgs::msg::PathPoint>>(path.points);

  // publish debug data
  publishDebugData(path, optimized_traj_points, *vehicle_param_ptr_);

  {  // print and publish debug msg
    debug_data_ptr_->msg_stream << __func__ << ":= " << stop_watch_.toc(__func__) << " [ms]\n"
                                << "========================================";
    tier4_debug_msgs::msg::StringStamped debug_msg_msg;
    debug_msg_msg.stamp = get_clock()->now();
    debug_msg_msg.data = debug_data_ptr_->msg_stream.getString();
    debug_msg_pub_->publish(debug_msg_msg);
  }

  return output;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::generateOptimizedTrajectory(
  const autoware_auto_planning_msgs::msg::Path & path)
{
  stop_watch_.tic(__func__);

  if (reset_prev_info_) {
    prev_path_points_ptr_ = nullptr;
    prev_optimal_trajs_ptr_ = nullptr;
  }

  // if ego pose moves far, reset optimization
  if (
    prev_optimal_trajs_ptr_ &&
    !hasValidNearestPointFromEgo(current_ego_pose_, *prev_optimal_trajs_ptr_, *traj_param_ptr_)) {
    RCLCPP_INFO(
      get_logger(), "[Avoidance] Could not find valid nearest point from ego, reset prev trajs");
    prev_optimal_trajs_ptr_ = nullptr;
  }

  // retun prev trajctory if replan is not needed
  if (!needReplan(path.points)) {
    return getPrevTrajectory(path.points);
  }

  // prepare variables
  prev_ego_pose_ptr_ = std::make_unique<geometry_msgs::msg::Pose>(current_ego_pose_);
  prev_replanned_time_ptr_ = std::make_unique<rclcpp::Time>(this->now());

  // set current pose and vel
  mpt_optimizer_ptr_->setEgoData(current_ego_pose_, current_twist_ptr_->twist.linear.x);

  // update debug_data
  debug_data_ptr_->msg_stream.is_showing_calculation_time = is_showing_calculation_time_;
  debug_data_ptr_->visualize_sampling_num = visualize_sampling_num_;
  debug_data_ptr_->current_ego_pose = current_ego_pose_;
  debug_data_ptr_->avoiding_circle_radius = mpt_param_ptr_->avoiding_circle_radius;
  debug_data_ptr_->avoiding_circle_offsets = mpt_param_ptr_->avoiding_circle_offsets;

  /*
  // filter avoiding obstacles
  // create clearance maps
  const auto avoiding_objects = object_filter_->getAvoidingObjects();
  */

  // create clearance maps
  const CVMaps cv_maps = costmap_generator_ptr_->getMaps(
    enable_avoidance_, path, /*avoiding_objects*/ in_objects_ptr_->objects, *traj_param_ptr_,
    debug_data_ptr_);

  // calculate trajectory with EB and MPT, then extend trajectory
  auto optimal_trajs = optimizeTrajectory(path, cv_maps);

  // insert 0 velocity when trajectory is over drivable area
  insertZeroVelocityOutsideDrivableArea(optimal_trajs.model_predictive_trajectory, cv_maps);

  // make previous trajectories
  prev_optimal_trajs_ptr_ =
    std::make_unique<Trajectories>(makePrevTrajectories(path.points, optimal_trajs));

  debug_data_ptr_->msg_stream << "  " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return optimal_trajs.model_predictive_trajectory;
}

bool ObstacleAvoidancePlanner::needReplan(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const
{
  if (
    !prev_ego_pose_ptr_ || !prev_replanned_time_ptr_ || !prev_path_points_ptr_ ||
    !prev_optimal_trajs_ptr_) {
    return true;
  }

  // when path from behavior changes, just replan and not reset the previous trajectories
  if (isPathShapeChanged(
        current_ego_pose_, path_points, prev_path_points_ptr_,
        traj_param_ptr_->delta_yaw_threshold_for_closest_point,
        distance_for_path_shape_change_detection_)) {
    RCLCPP_INFO(get_logger(), "[Avoidance] Path shape is changed, so replan");
    return true;
  }

  const double delta_dist =
    tier4_autoware_utils::calcDistance2d(current_ego_pose_.position, prev_ego_pose_ptr_->position);
  if (delta_dist > min_ego_moving_dist_for_replan_) {
    // RCLCPP_INFO(get_logger(), "[Avoidance] Current ego pose is far from previous ego pose, so
    // replan");
    return true;
  }

  const double delta_time_sec = (this->now() - *prev_replanned_time_ptr_).seconds();
  if (delta_time_sec > min_delta_time_sec_for_replan_) {
    return true;
  }
  return false;
}

Trajectories ObstacleAvoidancePlanner::optimizeTrajectory(
  const autoware_auto_planning_msgs::msg::Path & path, const CVMaps & cv_maps)
{
  stop_watch_.tic(__func__);

  // EB: smooth trajectory
  const auto eb_traj = eb_path_optimizer_ptr_->getEBTrajectory(
    enable_avoidance_, current_ego_pose_, path, prev_optimal_trajs_ptr_, cv_maps, debug_data_ptr_);
  if (!eb_traj) {
    return getPrevTrajs(path.points);
  }

  // MPT: optimize trajectory to be kinematically feasible and collision free
  const auto mpt_trajs = mpt_optimizer_ptr_->getModelPredictiveTrajectory(
    enable_avoidance_, eb_traj.get(), path.points, prev_optimal_trajs_ptr_, cv_maps,
    debug_data_ptr_);
  if (!mpt_trajs) {
    return getPrevTrajs(path.points);
  }

  // make trajectories, which has all optimized trajectories information
  Trajectories trajs;
  trajs.smoothed_trajectory = eb_traj.get();
  trajs.mpt_ref_points = mpt_trajs.get().ref_points;
  trajs.model_predictive_trajectory = mpt_trajs.get().mpt;

  // debug data
  debug_data_ptr_->mpt_traj = mpt_trajs.get().mpt;
  debug_data_ptr_->mpt_ref_traj =
    points_utils::convertToTrajectoryPoints(mpt_trajs.get().ref_points);
  debug_data_ptr_->eb_traj = eb_traj.get();

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return trajs;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::getExtendedOptimizedTrajectory(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & optimized_points)
{
  stop_watch_.tic(__func__);
  if (static_cast<int>(optimized_points.size()) <= traj_param_ptr_->num_fix_points_for_extending) {
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("EBPathOptimizer"), logger_ros_clock_,
      std::chrono::milliseconds(10000).count(), "[Avoidance] Not extend trajectory");
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  }

  const double accum_arc_length = tier4_autoware_utils::calcArcLength(optimized_points);
  if (
    accum_arc_length > traj_param_ptr_->trajectory_length ||
    points_utils::getLastExtendedPoint(
      path_points.back(), optimized_points.back().pose,
      traj_param_ptr_->delta_yaw_threshold_for_closest_point,
      traj_param_ptr_->max_dist_for_extending_end_point)) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(10000).count(),
      "[Avoidance] Not extend trajectory");
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  }

  // calculate end idx of optimized points on path points
  const auto opt_end_path_idx = tier4_autoware_utils::findNearestIndex(
    path_points, optimized_points.back().pose, std::numeric_limits<double>::max(),
    traj_param_ptr_->delta_yaw_threshold_for_closest_point);
  if (!opt_end_path_idx) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(10000).count(),
      "[Avoidance] Not extend traj since could not find nearest idx from last opt point");
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  }

  const size_t non_fixed_begin_path_idx = opt_end_path_idx.get();
  const size_t non_fixed_end_path_idx =
    points_utils::findForwardIndex(path_points, non_fixed_begin_path_idx, 10.0);

  if (non_fixed_begin_path_idx == non_fixed_end_path_idx) {
    // no extended trajectory
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  } else if (non_fixed_end_path_idx == path_points.size() - 1) {
    // no need to connect smoothly since extended trajectory length is short enough
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> extended_path_points{
      path_points.begin() + non_fixed_begin_path_idx + 1, path_points.end()};
    return points_utils::convertToTrajectoryPoints(extended_path_points);
  }

  // define non_fixed/fixed_traj_points
  const auto begin_point = optimized_points.back();
  const auto end_point =
    points_utils::convertToTrajectoryPoint(path_points.at(non_fixed_end_path_idx));
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> non_fixed_traj_points{
    begin_point, end_point};
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> fixed_path_points{
    path_points.begin() + non_fixed_end_path_idx + 1, path_points.end()};
  const auto fixed_traj_points = points_utils::convertToTrajectoryPoints(fixed_path_points);

  // spline interpolation to two traj points with end diff constraints
  const double begin_yaw = tf2::getYaw(begin_point.pose.orientation);
  const double end_yaw = tf2::getYaw(end_point.pose.orientation);
  const auto interpolated_non_fixed_traj_points =
    interpolation_utils::getConnectedInterpolatedPoints(
      non_fixed_traj_points, eb_param_ptr_->delta_arc_length_for_optimization, begin_yaw,
      // TODO(murooka) eb param?
      end_yaw);

  // concat interpolated_non_fixed and fixed traj points
  auto extended_points = interpolated_non_fixed_traj_points;
  extended_points.insert(extended_points.end(), fixed_traj_points.begin(), fixed_traj_points.end());

  debug_data_ptr_->extended_non_fixed_traj = interpolated_non_fixed_traj_points;
  debug_data_ptr_->extended_fixed_traj = fixed_traj_points;

  // TODO(murooka) deal with 1e10
  for (auto & point : extended_points) {
    point.longitudinal_velocity_mps = 1e10;
  }

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return extended_points;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::generatePostProcessedTrajectory(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & optimized_traj_points)
{
  stop_watch_.tic(__func__);

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> trajectory_points;
  if (path_points.empty()) {
    autoware_auto_planning_msgs::msg::TrajectoryPoint tmp_point;
    tmp_point.pose = current_ego_pose_;
    tmp_point.longitudinal_velocity_mps = 0.0;
    trajectory_points.push_back(tmp_point);
    return trajectory_points;
  }
  if (optimized_traj_points.empty()) {
    trajectory_points = points_utils::convertToTrajectoryPoints(path_points);
    return trajectory_points;
  }

  // calculate extended trajectory that connects to optimized trajectory smoothly
  const auto extended_traj_points =
    getExtendedOptimizedTrajectory(path_points, optimized_traj_points);

  // concat trajectories
  // stop_watch.tic();
  const auto full_traj_points =
    points_utils::concatTraj(optimized_traj_points, extended_traj_points);
  // RCLCPP_INFO_EXPRESSION(
  // rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
  // "    concatTraj:= %f [ms]", stop_watch.toc() * 1000.0);

  // Basically not changing the shape of trajectory,
  // re-calculate position and velocity with interpolation
  const auto full_traj_points_with_vel = reCalcTrajectoryPoints(path_points, full_traj_points);

  debug_data_ptr_->msg_stream << "  " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return full_traj_points_with_vel;
}

void ObstacleAvoidancePlanner::insertZeroVelocityOutsideDrivableArea(
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const CVMaps & cv_maps)
{
  stop_watch_.tic(__func__);

  const auto & map_info = cv_maps.map_info;
  const auto & road_clearance_map = cv_maps.clearance_map;

  const size_t nearest_idx =
    tier4_autoware_utils::findNearestIndex(traj_points, current_ego_pose_.position);
  for (size_t i = nearest_idx; i < traj_points.size(); ++i) {
    const auto & traj_point = traj_points.at(i);

    // calculate firstly outside drivable area
    const bool is_outside = [&]() {
      if (use_footprint_for_drivability_) {
        return cv_drivable_area_utils::isOutsideDrivableAreaFromRectangleFootprint(
          traj_point, road_clearance_map, map_info, *vehicle_param_ptr_);
      }
      return cv_drivable_area_utils::isOutsideDrivableAreaFromCirclesFootprint(
        traj_point, road_clearance_map, map_info, mpt_param_ptr_->avoiding_circle_offsets,
        mpt_param_ptr_->avoiding_circle_radius);
    }();

    // only insert zero velocity to the first point outside drivable area
    if (is_outside) {
      traj_points[i].longitudinal_velocity_mps = 0.0;
      debug_data_ptr_->stop_pose_by_drivable_area = traj_points[i].pose;
      break;
    }
  }

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::reCalcTrajectoryPoints(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points) const
{
  stop_watch_.tic(__func__);

  const auto fine_traj_points = [&]() {
    // interpolate pose (x, y, and yaw)
    auto interpolated_traj_points = interpolation_utils::getInterpolatedTrajectoryPoints(
      traj_points, traj_param_ptr_->delta_arc_length_for_trajectory);

    // compensate last pose
    points_utils::compensateLastPose(
      path_points.back(), *traj_param_ptr_, interpolated_traj_points);

    return interpolated_traj_points;
  }();

  // search zero velocity index of fine_traj_points
  const size_t zero_vel_path_idx = searchExtendedZeroVelocityIndex(fine_traj_points, path_points);
  const size_t zero_vel_traj_idx = searchExtendedZeroVelocityIndex(fine_traj_points, traj_points);
  const size_t zero_vel_fine_traj_idx = std::min(zero_vel_path_idx, zero_vel_traj_idx);

  // search nearest velocity to path point, and fill in result trajectory
  const auto re_calc_traj_points =
    alignVelocityWithPoints(fine_traj_points, path_points, zero_vel_fine_traj_idx);

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return re_calc_traj_points;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::alignVelocityWithPoints(
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & base_traj_points,
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & points,
  const int zero_vel_traj_idx) const
{
  stop_watch_.tic(__func__);

  auto traj_points = base_traj_points;
  size_t prev_begin_idx = 0;
  for (size_t i = 0; i < traj_points.size(); ++i) {
    const auto truncated_points = points_utils::clipForwardPoints(points, prev_begin_idx, 5.0);

    const auto & target_pos = traj_points[i].pose.position;
    const size_t closest_seg_idx =
      tier4_autoware_utils::findNearestSegmentIndex(truncated_points, target_pos);

    // lerp z
    traj_points[i].pose.position.z = lerpPoseZ(truncated_points, target_pos, closest_seg_idx);

    // lerp vx
    const double target_vel = lerpTwistX(truncated_points, target_pos, closest_seg_idx);
    if (static_cast<int>(i) >= zero_vel_traj_idx) {
      traj_points[i].longitudinal_velocity_mps = 0.0;
    } else if (target_vel < 1e-6) {
      const auto prev_idx = std::max(static_cast<int>(i) - 1, 0);
      traj_points[i].longitudinal_velocity_mps = traj_points[prev_idx].longitudinal_velocity_mps;
    } else {
      traj_points[i].longitudinal_velocity_mps = target_vel;
    }
    /* else if (static_cast<int>(i) <= max_skip_comparison_idx) {
      traj_points[i].longitudinal_velocity_mps = target_vel;
    } else {
      traj_points[i].longitudinal_velocity_mps = std::fmin(target_vel,
    traj_points[i].longitudinal_velocity_mps);
    }*/

    // NOTE: closest_seg_idx is for the clipped trajectory. This operation must be "+=".
    prev_begin_idx += closest_seg_idx;
  }

  debug_data_ptr_->msg_stream << "      " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
  return traj_points;
}

void ObstacleAvoidancePlanner::publishDebugData(
  const autoware_auto_planning_msgs::msg::Path & path,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const VehicleParam & vehicle_param)
{
  stop_watch_.tic(__func__);

  {  // publish trajectories
    auto traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->foa_data.avoiding_traj_points);
    traj.header = path.header;
    avoiding_traj_pub_->publish(traj);

    auto debug_eb_traj = tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->eb_traj);
    debug_eb_traj.header = path.header;
    debug_eb_traj_pub_->publish(debug_eb_traj);

    auto debug_extended_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->extended_fixed_traj);
    debug_extended_fixed_traj.header = path.header;
    debug_extended_fixed_traj_pub_->publish(debug_extended_fixed_traj);

    auto debug_extended_non_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->extended_non_fixed_traj);
    debug_extended_non_fixed_traj.header = path.header;
    debug_extended_non_fixed_traj_pub_->publish(debug_extended_non_fixed_traj);

    auto debug_mpt_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->mpt_fixed_traj);
    debug_mpt_fixed_traj.header = path.header;
    debug_mpt_fixed_traj_pub_->publish(debug_mpt_fixed_traj);

    auto debug_mpt_ref_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->mpt_ref_traj);
    debug_mpt_ref_traj.header = path.header;
    debug_mpt_ref_traj_pub_->publish(debug_mpt_ref_traj);

    auto debug_mpt_traj = tier4_autoware_utils::convertToTrajectory(debug_data_ptr_->mpt_traj);
    debug_mpt_traj.header = path.header;
    debug_mpt_traj_pub_->publish(debug_mpt_traj);

    tier4_planning_msgs::msg::IsAvoidancePossible is_avoidance_possible;
    is_avoidance_possible.is_avoidance_possible = debug_data_ptr_->foa_data.is_avoidance_possible;
    is_avoidance_possible_pub_->publish(is_avoidance_possible);
  }

  {  // publish markers
    /*
    std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> traj_points_debug = traj_points;
    // Add z information for virtual wall
    if (!traj_points_debug.empty()) {
      const auto opt_idx = tier4_autoware_utils::findNearestIndex(
        path.points, traj_points.back().pose, std::numeric_limits<double>::max(),
        traj_param_ptr_->delta_yaw_threshold_for_closest_point);
      const int idx = opt_idx ? *opt_idx : 0;
      traj_points_debug.back().pose.position.z = path.points.at(idx).pose.position.z + 1.0;
    }
    */

    stop_watch_.tic("getDebugVisualizationMarker");
    const auto & debug_marker = debug_visualization::getDebugVisualizationMarker(
      debug_data_ptr_, traj_points, vehicle_param, false);
    debug_data_ptr_->msg_stream << "      getDebugVisualizationMarker:= "
                                << stop_watch_.toc("getDebugVisualizationMarker") << " [ms]\n";

    stop_watch_.tic("publishDebugVisualizationMarker");
    debug_markers_pub_->publish(debug_marker);
    debug_data_ptr_->msg_stream << "      publishDebugVisualizationMarker:= "
                                << stop_watch_.toc("publishDebugVisualizationMarker") << " [ms]\n";
  }

  {  // publish clearance map
    stop_watch_.tic("publishClearanceMap");
    if (is_publishing_area_with_objects_) {  // false
      debug_area_with_objects_pub_->publish(debug_visualization::getDebugCostmap(
        debug_data_ptr_->area_with_objects_map, path.drivable_area));
    }
    if (is_publishing_object_clearance_map_) {  // false
      debug_object_clearance_map_pub_->publish(debug_visualization::getDebugCostmap(
        debug_data_ptr_->only_object_clearance_map, path.drivable_area));
    }
    if (is_publishing_clearance_map_) {  // true
      debug_clearance_map_pub_->publish(
        debug_visualization::getDebugCostmap(debug_data_ptr_->clearance_map, path.drivable_area));
    }
    debug_data_ptr_->msg_stream << "      getDebugCostMap * 3:= "
                                << stop_watch_.toc("publishClearanceMap") << " [ms]\n";
  }

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";
}

Trajectories ObstacleAvoidancePlanner::getPrevTrajs(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const
{
  if (prev_optimal_trajs_ptr_) {
    return *prev_optimal_trajs_ptr_;
  }

  const auto traj = points_utils::convertToTrajectoryPoints(path_points);
  Trajectories trajs;
  trajs.smoothed_trajectory = traj;
  trajs.model_predictive_trajectory = traj;
  return trajs;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::getPrevTrajectory(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const
{
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> traj;
  const auto & trajs = getPrevTrajs(path_points);
  return trajs.model_predictive_trajectory;
}

Trajectories ObstacleAvoidancePlanner::makePrevTrajectories(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const Trajectories & trajs)
{
  stop_watch_.tic(__func__);

  const auto post_processed_smoothed_traj =
    generatePostProcessedTrajectory(path_points, trajs.smoothed_trajectory);
  // TODO(murooka) generatePoseProcessedTrajectory may be too large
  Trajectories trajectories;
  trajectories.smoothed_trajectory = post_processed_smoothed_traj;
  trajectories.mpt_ref_points = trajs.mpt_ref_points;
  trajectories.model_predictive_trajectory = trajs.model_predictive_trajectory;

  debug_data_ptr_->msg_stream << "    " << __func__ << ":= " << stop_watch_.toc(__func__)
                              << " [ms]\n";

  return trajectories;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ObstacleAvoidancePlanner)

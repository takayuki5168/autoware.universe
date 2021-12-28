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

#include "obstacle_avoidance_planner/debug_visualization.hpp"
#include "obstacle_avoidance_planner/process_cv.hpp"
#include "obstacle_avoidance_planner/util.hpp"
#include "rclcpp/time.hpp"
#include "tf2/utils.h"
#include "tier4_autoware_utils/ros/update_param.hpp"
#include "tier4_autoware_utils/system/stop_watch.hpp"
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
size_t searchExtendedZeroVelocityIndex(const std::vector<T1> & fine_points, const std::vector<T2> & points)
{
  const auto opt_zero_vel_idx = tier4_autoware_utils::searchZeroVelocityIndex(points);
  const size_t zero_vel_idx = opt_zero_vel_idx ? opt_zero_vel_idx.get() : points.size() - 1;
  return tier4_autoware_utils::findNearestIndex(
    fine_points, points.at(zero_vel_idx).pose.position);
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
}  // namespace

ObstacleAvoidancePlanner::ObstacleAvoidancePlanner(const rclcpp::NodeOptions & node_options)
: Node("obstacle_avoidance_planner", node_options), logger_ros_clock_(RCL_ROS_TIME)
{
  rclcpp::Clock::SharedPtr clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

  rclcpp::QoS durable_qos{1};
  durable_qos.transient_local();

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

  is_publishing_area_with_objects_ = declare_parameter("is_publishing_area_with_objects", false);
  is_publishing_object_clearance_map_ =
    declare_parameter("is_publishing_object_clearance_map", false);
  is_publishing_clearance_map_ = declare_parameter("is_publishing_clearance_map", false);
  is_showing_debug_info_ = declare_parameter("is_showing_debug_info", false);
  is_using_vehicle_config_ = declare_parameter("is_using_vehicle_config", false);
  is_stopping_if_outside_drivable_area_ =
    declare_parameter("is_stopping_if_outside_drivable_area", true);
  enable_avoidance_ = declare_parameter("enable_avoidance", true);

  qp_param_ptr_ = std::make_unique<QPParam>();
  qp_param_ptr_->max_iteration = declare_parameter("qp_max_iteration", 10000);
  qp_param_ptr_->eps_abs = declare_parameter("qp_eps_abs", 1.0e-8);
  qp_param_ptr_->eps_rel = declare_parameter("qp_eps_rel", 1.0e-11);
  qp_param_ptr_->eps_abs_for_extending = declare_parameter("qp_eps_abs_for_extending", 1.0e-6);
  qp_param_ptr_->eps_rel_for_extending = declare_parameter("qp_eps_rel_for_extending", 1.0e-8);
  qp_param_ptr_->eps_abs_for_visualizing = declare_parameter("qp_eps_abs_for_visualizing", 1.0e-6);
  qp_param_ptr_->eps_rel_for_visualizing = declare_parameter("qp_eps_rel_for_visualizing", 1.0e-8);

  traj_param_ptr_ = std::make_unique<TrajectoryParam>();
  traj_param_ptr_->num_sampling_points = declare_parameter("num_sampling_points", 100);
  traj_param_ptr_->num_joint_buffer_points = declare_parameter("num_joint_buffer_points", 2);
  traj_param_ptr_->num_joint_buffer_points_for_extending =
    declare_parameter("num_joint_buffer_points_for_extending", 4);
  traj_param_ptr_->num_offset_for_begin_idx = declare_parameter("num_offset_for_begin_idx", 2);
  traj_param_ptr_->num_fix_points_for_extending =
    declare_parameter("num_fix_points_for_extending", 2);
  traj_param_ptr_->delta_arc_length_for_optimization =
    declare_parameter("delta_arc_length_for_optimization", 1.0);
  traj_param_ptr_->delta_arc_length_for_trajectory =
    declare_parameter("delta_arc_length_for_trajectory", 0.1);
  traj_param_ptr_->delta_dist_threshold_for_closest_point =
    declare_parameter("delta_dist_threshold_for_closest_point", 3.0);
  traj_param_ptr_->delta_yaw_threshold_for_closest_point =
    declare_parameter("delta_yaw_threshold_for_closest_point", 1.0);
  traj_param_ptr_->delta_yaw_threshold_for_straight =
    declare_parameter("delta_yaw_threshold_for_straight", 0.02);
  traj_param_ptr_->trajectory_length = declare_parameter("trajectory_length", 200.0);
  traj_param_ptr_->forward_fixing_distance = declare_parameter("forward_fixing_distance", 10.0);
  traj_param_ptr_->backward_fixing_distance = declare_parameter("backward_fixing_distance", 5.0);
  traj_param_ptr_->max_avoiding_ego_velocity_ms =
    declare_parameter("max_avoiding_ego_velocity_ms", 6.0);
  traj_param_ptr_->max_avoiding_objects_velocity_ms =
    declare_parameter("max_avoiding_objects_velocity_ms", 0.1);
  traj_param_ptr_->center_line_width = declare_parameter("center_line_width", 1.7);
  traj_param_ptr_->acceleration_for_non_deceleration_range =
    declare_parameter("acceleration_for_non_deceleration_range", 1.0);
  traj_param_ptr_->max_dist_for_extending_end_point =
    declare_parameter("max_dist_for_extending_end_point", 5.0);
  traj_param_ptr_->is_avoiding_unknown = declare_parameter("avoiding_object_type.unknown", true);
  traj_param_ptr_->is_avoiding_car = declare_parameter("avoiding_object_type.car", true);
  traj_param_ptr_->is_avoiding_truck = declare_parameter("avoiding_object_type.truck", true);
  traj_param_ptr_->is_avoiding_bus = declare_parameter("avoiding_object_type.bus", true);
  traj_param_ptr_->is_avoiding_bicycle = declare_parameter("avoiding_object_type.bicycle", true);
  traj_param_ptr_->is_avoiding_motorbike =
    declare_parameter("avoiding_object_type.motorbike", true);
  traj_param_ptr_->is_avoiding_pedestrian =
    declare_parameter("avoiding_object_type.pedestrian", true);

  // mpt param in traj_param_ptr_
  traj_param_ptr_->forward_fixing_mpt_min_distance =
    declare_parameter("mpt.forward_fixing_mpt_min_distance", 3.0);
  traj_param_ptr_->delta_arc_length_for_mpt_points =
    declare_parameter("mpt.delta_arc_length_for_mpt_points", 1.0);
  traj_param_ptr_->is_avoiding_animal = declare_parameter("avoiding_object_type.animal", true);
  traj_param_ptr_->forward_fixing_mpt_time = declare_parameter("mpt.forward_fixing_mpt_time", 1.0);

  const double hard_clearance_from_road = declare_parameter("hard_clearance_from_road", 0.1);
  const double soft_clearance_from_road = declare_parameter("soft_clearance_from_road", 0.1);
  const double soft_second_clearance_from_road =
    declare_parameter("soft_second_clearance_from_road", 0.1);
  const double clearance_from_object = declare_parameter("clearance_from_object", 0.6);
  const double extra_desired_clearance_from_road =
    declare_parameter("extra_desired_clearance_from_road", 0.2);

  constrain_param_ptr_ = std::make_unique<ConstrainParam>();
  constrain_param_ptr_->is_getting_constraints_close2path_points =
    declare_parameter("is_getting_constraints_close2path_points", false);
  constrain_param_ptr_->clearance_for_straight_line =
    declare_parameter("clearance_for_straight_line", 0.05);
  constrain_param_ptr_->clearance_for_joint = declare_parameter("clearance_for_joint", 0.1);
  constrain_param_ptr_->range_for_extend_joint = declare_parameter("range_for_extend_joint", 1.6);
  constrain_param_ptr_->clearance_for_only_smoothing =
    declare_parameter("clearance_for_only_smoothing", 0.1);
  constrain_param_ptr_->clearance_from_object_for_straight =
    declare_parameter("clearance_from_object_for_straight", 10.0);
  constrain_param_ptr_->soft_clearance_from_road = soft_clearance_from_road;
  constrain_param_ptr_->clearance_from_object = clearance_from_object;
  constrain_param_ptr_->min_object_clearance_for_joint =
    declare_parameter("min_object_clearance_for_joint", 3.2);
  constrain_param_ptr_->max_x_constrain_search_range =
    declare_parameter("max_x_constrain_search_range", 0.4);
  constrain_param_ptr_->coef_x_constrain_search_resolution =
    declare_parameter("coef_x_constrain_search_resolution", 1.0);
  constrain_param_ptr_->coef_y_constrain_search_resolution =
    declare_parameter("coef_y_constrain_search_resolution", 0.5);
  constrain_param_ptr_->keep_space_shape_x = declare_parameter("keep_space_shape_x", 3.0);
  constrain_param_ptr_->keep_space_shape_y = declare_parameter("keep_space_shape_y", 2.0);
  constrain_param_ptr_->max_lon_space_for_driveable_constraint =
    declare_parameter("max_lon_space_for_driveable_constraint", 0.5);
  constrain_param_ptr_->clearance_for_fixing = 0.0;
  constrain_param_ptr_->min_object_clearance_for_deceleration =
    constrain_param_ptr_->clearance_from_object + constrain_param_ptr_->keep_space_shape_y * 0.5;

  min_ego_moving_dist_for_replan_ = declare_parameter<double>("min_ego_moving_dist_for_replan");
  min_delta_time_sec_for_replan_ = declare_parameter("min_delta_time_sec_for_replan", 1.0);
  distance_for_path_shape_change_detection_ =
    declare_parameter("distance_for_path_shape_change_detection", 2.0);

  // vehicle param
  vehicle_param_ptr_ = std::make_unique<VehicleParam>();
  const auto vehicle_info = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();
  vehicle_param_ptr_->width = vehicle_info.vehicle_width_m;
  vehicle_param_ptr_->length = vehicle_info.vehicle_length_m;
  vehicle_param_ptr_->wheelbase = vehicle_info.wheel_base_m;
  vehicle_param_ptr_->rear_overhang = vehicle_info.rear_overhang_m;
  vehicle_param_ptr_->front_overhang = vehicle_info.front_overhang_m;

  if (is_using_vehicle_config_) {
    double vehicle_width = vehicle_info.vehicle_width_m;
    traj_param_ptr_->center_line_width = vehicle_width;
    constrain_param_ptr_->keep_space_shape_y = vehicle_width;
  }

  double max_steer_deg = 0;
  max_steer_deg = declare_parameter<double>("mpt.max_steer_deg");
  vehicle_param_ptr_->max_steer_rad = max_steer_deg * M_PI / 180.0;
  vehicle_param_ptr_->steer_tau = declare_parameter<double>("mpt.steer_tau");

  // mpt param
  mpt_param_ptr_ = std::make_unique<MPTParam>();
  mpt_param_ptr_->is_hard_fixing_terminal_point =
    declare_parameter("mpt.is_hard_fixing_terminal_point", true);
  mpt_param_ptr_->num_curvature_sampling_points =
    declare_parameter("mpt.num_curvature_sampling_points", 5);
  mpt_param_ptr_->soft_avoidance_weight = declare_parameter<double>("mpt.soft_avoidance_weight");
  mpt_param_ptr_->soft_second_avoidance_weight =
    declare_parameter<double>("mpt.soft_second_avoidance_weight");
  mpt_param_ptr_->avoiding_constraint_type = declare_parameter<int>("mpt.avoiding_constraint_type");
  mpt_param_ptr_->l_inf_norm = declare_parameter<bool>("mpt.l_inf_norm");
  mpt_param_ptr_->two_step_soft_constraint =
    declare_parameter<bool>("mpt.two_step_soft_constraint");
  mpt_param_ptr_->plan_from_ego = declare_parameter<bool>("mpt.plan_from_ego");

  mpt_param_ptr_->lat_error_weight = declare_parameter<double>("mpt.lat_error_weight");
  mpt_param_ptr_->yaw_error_weight = declare_parameter<double>("mpt.yaw_error_weight");
  mpt_param_ptr_->yaw_error_rate_weight = declare_parameter<double>("mpt.yaw_error_rate_weight");
  mpt_param_ptr_->obstacle_avoid_lat_error_weight =
    declare_parameter<double>("mpt.obstacle_avoid_lat_error_weight");
  mpt_param_ptr_->obstacle_avoid_yaw_error_weight =
    declare_parameter<double>("mpt.obstacle_avoid_yaw_error_weight");
  mpt_param_ptr_->near_objects_length = declare_parameter<double>("mpt.near_objects_length");
  mpt_param_ptr_->steer_input_weight = declare_parameter<double>("mpt.steer_input_weight");
  mpt_param_ptr_->obstacle_avoid_steer_input_weight =
    declare_parameter<double>("mpt.obstacle_avoid_steer_input_weight");
  mpt_param_ptr_->steer_rate_weight = declare_parameter<double>("mpt.steer_rate_weight");
  mpt_param_ptr_->steer_acc_weight = declare_parameter<double>("mpt.steer_acc_weight");
  mpt_param_ptr_->steer_limit = max_steer_deg * M_PI / 180.0;
  mpt_param_ptr_->terminal_lat_error_weight =
    declare_parameter<double>("mpt.terminal_lat_error_weight");
  mpt_param_ptr_->terminal_yaw_error_weight =
    declare_parameter<double>("mpt.terminal_yaw_error_weight");
  mpt_param_ptr_->terminal_path_lat_error_weight =
    declare_parameter<double>("mpt.terminal_path_lat_error_weight");
  mpt_param_ptr_->terminal_path_yaw_error_weight =
    declare_parameter<double>("mpt.terminal_path_yaw_error_weight");
  mpt_param_ptr_->zero_ff_steer_angle = declare_parameter<double>("mpt.zero_ff_steer_angle");
  mpt_param_ptr_->optimization_center_offset = declare_parameter<double>(
    "mpt.optimization_center_offset", vehicle_param_ptr_->wheelbase / 2.0);

  mpt_param_ptr_->avoiding_circle_offsets =
    declare_parameter<std::vector<double>>("mpt.avoiding_circle_offsets", std::vector<double>());
  mpt_param_ptr_->avoiding_circle_radius =
    declare_parameter<double>("mpt.avoiding_circle_radius", -1);
  if (
    mpt_param_ptr_->avoiding_circle_offsets.empty() || mpt_param_ptr_->avoiding_circle_radius < 0) {
    constexpr size_t avoiding_circle_num = 4;
    mpt_param_ptr_->avoiding_circle_radius = std::hypot(
      vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0,
      vehicle_param_ptr_->width / 2.0);

    mpt_param_ptr_->avoiding_circle_offsets.clear();
    for (size_t i = 0; i < avoiding_circle_num; ++i) {
      mpt_param_ptr_->avoiding_circle_offsets.push_back(
        vehicle_param_ptr_->length / static_cast<double>(avoiding_circle_num) / 2.0 * (1 + 2 * i) -
        vehicle_param_ptr_->rear_overhang);
    }
  }

  {  // calculate avoid circle from base link distance alongside wheel base
    mpt_param_ptr_->hard_clearance_from_road = hard_clearance_from_road;
    mpt_param_ptr_->soft_clearance_from_road = soft_clearance_from_road;
    mpt_param_ptr_->soft_second_clearance_from_road = soft_second_clearance_from_road;
    mpt_param_ptr_->extra_desired_clearance_from_road = extra_desired_clearance_from_road;
    mpt_param_ptr_->clearance_from_object = clearance_from_object;
  }

  reset_prev_info_ = declare_parameter<bool>("reset_prev_info");
  use_footprint_for_drivability_ = declare_parameter<bool>("use_footprint_for_drivability");

  visualize_sampling_num_ = declare_parameter<int>("mpt.visualize_sampling_num", 1);

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

  double max_steer_deg = 0.0;
  updateParam<double>(parameters, "mpt.max_steer_deg", max_steer_deg);
  vehicle_param_ptr_->max_steer_rad = max_steer_deg * M_PI / 180.0;

  updateParam<double>(
    parameters, "mpt.delta_arc_length_for_mpt_points",
    traj_param_ptr_->delta_arc_length_for_mpt_points);
  updateParam<int>(parameters, "num_sampling_points", traj_param_ptr_->num_sampling_points);

  updateParam<double>(parameters, "mpt.steer_tau", vehicle_param_ptr_->steer_tau);

  // trajectory total/fixing length
  updateParam<double>(parameters, "trajectory_length", traj_param_ptr_->trajectory_length);
  updateParam<double>(
    parameters, "forward_fixing_distance", traj_param_ptr_->forward_fixing_distance);
  updateParam<double>(
    parameters, "backward_fixing_distance", traj_param_ptr_->backward_fixing_distance);
  updateParam<double>(
    parameters, "mpt.forward_fixing_mpt_time", traj_param_ptr_->forward_fixing_mpt_time);
  updateParam<double>(
    parameters, "mpt.forward_fixing_mpt_min_distance",
    traj_param_ptr_->forward_fixing_mpt_min_distance);

  // clearance for unique points
  updateParam<double>(
    parameters, "clearance_for_straight_line", constrain_param_ptr_->clearance_for_straight_line);
  updateParam<double>(parameters, "clearance_for_joint", constrain_param_ptr_->clearance_for_joint);
  updateParam<double>(
    parameters, "clearance_for_only_smoothing", constrain_param_ptr_->clearance_for_only_smoothing);
  updateParam<double>(
    parameters, "clearance_from_object_for_straight",
    constrain_param_ptr_->clearance_from_object_for_straight);
  updateParam<double>(
    parameters, "min_object_clearance_for_joint",
    constrain_param_ptr_->min_object_clearance_for_joint);

  updateParam<double>(
    parameters, "hard_clearance_from_road", mpt_param_ptr_->hard_clearance_from_road);
  updateParam<double>(
    parameters, "soft_clearance_from_road", mpt_param_ptr_->soft_clearance_from_road);
  updateParam<double>(
    parameters, "soft_second_clearance_from_road", mpt_param_ptr_->soft_second_clearance_from_road);
  updateParam<double>(parameters, "clearance_from_object", mpt_param_ptr_->clearance_from_object);
  updateParam<double>(
    parameters, "extra_desired_clearance_from_road",
    mpt_param_ptr_->extra_desired_clearance_from_road);

  // clearance(distance) when generating trajectory
  constrain_param_ptr_->soft_clearance_from_road = mpt_param_ptr_->soft_clearance_from_road;
  constrain_param_ptr_->clearance_from_object = mpt_param_ptr_->clearance_from_object;

  // avoiding param
  updateParam<double>(
    parameters, "max_avoiding_objects_velocity_ms",
    traj_param_ptr_->max_avoiding_objects_velocity_ms);
  updateParam<double>(
    parameters, "max_avoiding_ego_velocity_ms", traj_param_ptr_->max_avoiding_ego_velocity_ms);
  updateParam<double>(parameters, "center_line_width", traj_param_ptr_->center_line_width);
  updateParam<double>(
    parameters, "acceleration_for_non_deceleration_range",
    traj_param_ptr_->acceleration_for_non_deceleration_range);

  // mpt param
  updateParam<int>(
    parameters, "mpt.avoiding_constraint_type", mpt_param_ptr_->avoiding_constraint_type);
  updateParam<bool>(parameters, "mpt.l_inf_norm", mpt_param_ptr_->l_inf_norm);
  updateParam<bool>(
    parameters, "mpt.two_step_soft_constraint", mpt_param_ptr_->two_step_soft_constraint);
  updateParam<bool>(parameters, "mpt.plan_from_ego", mpt_param_ptr_->plan_from_ego);

  updateParam<double>(
    parameters, "mpt.soft_avoidance_weight", mpt_param_ptr_->soft_avoidance_weight);
  updateParam<double>(
    parameters, "mpt.soft_second_avoidance_weight", mpt_param_ptr_->soft_second_avoidance_weight);
  updateParam<double>(parameters, "mpt.lat_error_weight", mpt_param_ptr_->lat_error_weight);
  updateParam<double>(parameters, "mpt.yaw_error_weight", mpt_param_ptr_->yaw_error_weight);
  updateParam<double>(
    parameters, "mpt.obstacle_avoid_lat_error_weight",
    mpt_param_ptr_->obstacle_avoid_lat_error_weight);
  updateParam<double>(
    parameters, "mpt.obstacle_avoid_yaw_error_weight",
    mpt_param_ptr_->obstacle_avoid_yaw_error_weight);
  updateParam<double>(parameters, "mpt.near_objects_length", mpt_param_ptr_->near_objects_length);

  updateParam<double>(
    parameters, "mpt.yaw_error_rate_weight", mpt_param_ptr_->yaw_error_rate_weight);
  updateParam<double>(parameters, "mpt.steer_input_weight", mpt_param_ptr_->steer_input_weight);
  updateParam<double>(
    parameters, "mpt.obstacle_avoid_steer_input_weight",
    mpt_param_ptr_->obstacle_avoid_steer_input_weight);
  updateParam<double>(parameters, "mpt.steer_rate_weight", mpt_param_ptr_->steer_rate_weight);
  updateParam<double>(parameters, "mpt.steer_acc_weight", mpt_param_ptr_->steer_acc_weight);
  updateParam<double>(
    parameters, "mpt.optimization_center_offset", mpt_param_ptr_->optimization_center_offset);

  updateParam<std::vector<double>>(
    parameters, "mpt.avoiding_circle_offsets", mpt_param_ptr_->avoiding_circle_offsets);
  updateParam<double>(
    parameters, "mpt.avoiding_circle_radius", mpt_param_ptr_->avoiding_circle_radius);

  updateParam<int>(parameters, "mpt.visualize_sampling_num", visualize_sampling_num_);

  updateParam<bool>(parameters, "reset_prev_info", reset_prev_info_);
  updateParam<bool>(parameters, "use_footprint_for_drivability", use_footprint_for_drivability_);

  initializeParam();

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";
  return result;
}

void ObstacleAvoidancePlanner::initializeParam()
{
  RCLCPP_WARN(get_logger(), "[ObstacleAvoidancePlanner] Resetting");

  eb_path_optimizer_ptr_ = std::make_unique<EBPathOptimizer>(
    is_showing_debug_info_, *qp_param_ptr_, *traj_param_ptr_, *constrain_param_ptr_,
    *vehicle_param_ptr_);

  mpt_optimizer_ptr_ = std::make_unique<MPTOptimizer>(
    is_showing_debug_info_, *traj_param_ptr_, *vehicle_param_ptr_, *mpt_param_ptr_);

  prev_path_points_ptr_ = nullptr;
  prev_optimal_trajs_ptr_ = nullptr;
}

void ObstacleAvoidancePlanner::pathCallback(
  const autoware_auto_planning_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mutex_);
  current_ego_pose_ = self_pose_listener_.getCurrentPose()->pose;

  if (msg->points.empty() || msg->drivable_area.data.empty() || !current_twist_ptr_) {
    return;
  }

  const auto output_trajectory_msg = generateTrajectory(*msg);
  traj_pub_->publish(output_trajectory_msg);
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

autoware_auto_planning_msgs::msg::Trajectory ObstacleAvoidancePlanner::generateTrajectory(
  const autoware_auto_planning_msgs::msg::Path & path)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic();

  debug_data_ = DebugData();

  const auto optimized_traj_points = generateOptimizedTrajectory(path);

  const auto post_processed_traj_points =
    generatePostProcessedTrajectory(path.points, optimized_traj_points);
  auto output = tier4_autoware_utils::convertToTrajectory(post_processed_traj_points);
  output.header = path.header;

  prev_path_points_ptr_ =
    std::make_unique<std::vector<autoware_auto_planning_msgs::msg::PathPoint>>(path.points);

  publishDebugData(debug_data_, path, optimized_traj_points, *vehicle_param_ptr_);
  const double total_ms = stop_watch.toc() * 1000.0;
  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "total:= %f [ms]\n==========================", total_ms);
  return output;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::generateOptimizedTrajectory(
  const autoware_auto_planning_msgs::msg::Path & path)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic("total");

  if (reset_prev_info_) {
    prev_path_points_ptr_ = nullptr;
    prev_optimal_trajs_ptr_ = nullptr;
  }

  // if ego pose moves far, reset optimization
  if (
    prev_optimal_trajs_ptr_ &&
    !util::hasValidNearestPointFromEgo(current_ego_pose_, *prev_optimal_trajs_ptr_, *traj_param_ptr_)) {
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
  debug_data_.visualize_sampling_num = visualize_sampling_num_;
  debug_data_.current_ego_pose = current_ego_pose_;
  debug_data_.avoiding_circle_radius = mpt_param_ptr_->avoiding_circle_radius;
  debug_data_.avoiding_circle_offsets = mpt_param_ptr_->avoiding_circle_offsets;

  /*
  // filter avoiding obstacles
  // create clearance maps
  const auto avoiding_objects = object_filter_->getAvoidingObjects();
  */

  // create clearance maps
  const CVMaps cv_maps = process_cv::getMaps(
    enable_avoidance_, path, /*avoiding_objects*/ in_objects_ptr_->objects, *traj_param_ptr_,
    debug_data_, is_showing_debug_info_);

  // calculate trajectory with EB and MPT, then extend trajectory
  auto optimal_trajs = calcTrajectories(path, cv_maps);

  // insert 0 velocity when trajectory is over drivable area
  stop_watch.tic();
  calcTrajectoryInsideDrivableArea(optimal_trajs.model_predictive_trajectory, cv_maps);

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    calcTrajectoryInsideDrivableArea:= %f [ms]", stop_watch.toc() * 1000.0);

  // make previous trajectories
  prev_optimal_trajs_ptr_ = std::make_unique<Trajectories>(
    makePrevTrajectories(path.points, optimal_trajs));

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "  node.generateOptimizedTrajectory:= %f [ms]", stop_watch.toc("total") * 1000.0);

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

Trajectories ObstacleAvoidancePlanner::calcTrajectories(
  const autoware_auto_planning_msgs::msg::Path & path, const CVMaps & cv_maps)
{
  tier4_autoware_utils::StopWatch stop_watch;

  // smooth trajectory with EB
  const auto eb_traj = eb_path_optimizer_ptr_->getEBTrajectory(
    enable_avoidance_, current_ego_pose_, path, prev_optimal_trajs_ptr_, cv_maps, debug_data_);
  if (!eb_traj) {
    return getPrevTrajs(path.points);
  }

  // optimize trajectory to be kinematically feasible and collision free with MPT
  const auto mpt_trajs = mpt_optimizer_ptr_->getModelPredictiveTrajectory(
    enable_avoidance_, eb_traj.get(), path.points, prev_optimal_trajs_ptr_, cv_maps, debug_data_);
  if (!mpt_trajs) {
    return getPrevTrajs(path.points);
  }

  // debug data
  debug_data_.mpt_traj = mpt_trajs.get().mpt;
  debug_data_.mpt_ref_traj = util::convertToTrajectoryPoints(mpt_trajs.get().ref_points);

  Trajectories trajs;
  trajs.smoothed_trajectory = eb_traj.get();
  trajs.mpt_ref_points = mpt_trajs.get().ref_points;
  trajs.model_predictive_trajectory = mpt_trajs.get().mpt;

  debug_data_.eb_traj = eb_traj.get();
  return trajs;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::getExtendedOptimizedTrajectory(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & optimized_points)
{
  if (static_cast<int>(optimized_points.size()) <= traj_param_ptr_->num_fix_points_for_extending) {
    RCLCPP_INFO_THROTTLE(
      rclcpp::get_logger("EBPathOptimizer"), logger_ros_clock_,
      std::chrono::milliseconds(10000).count(), "[Avoidance] Not extend trajectory");
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  }

  const double accum_arc_length = tier4_autoware_utils::calcArcLength(optimized_points);
  if (
    accum_arc_length > traj_param_ptr_->trajectory_length ||
    util::getLastExtendedPoint(
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
    util::findForwardIndex(path_points, non_fixed_begin_path_idx, 10.0);

  if (non_fixed_begin_path_idx == non_fixed_end_path_idx) {
    // no extended trajectory
    return std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{};
  } else if (non_fixed_end_path_idx == path_points.size() - 1) {
    // no need to connect smoothly since extended trajectory length is short enough
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> extended_path_points{
      path_points.begin() + non_fixed_begin_path_idx + 1, path_points.end()};
    return util::convertToTrajectoryPoints(extended_path_points);
  }

  // define non_fixed/fixed_traj_points
  const auto begin_point = optimized_points.back();
  const auto end_point = util::convertToTrajectoryPoint(path_points.at(non_fixed_end_path_idx));
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> non_fixed_traj_points{
    begin_point, end_point};
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> fixed_path_points{
    path_points.begin() + non_fixed_end_path_idx + 1, path_points.end()};
  const auto fixed_traj_points = util::convertToTrajectoryPoints(fixed_path_points);

  // spline interpolation to two traj points with end diff constraints
  const double begin_yaw = tf2::getYaw(begin_point.pose.orientation);
  const double end_yaw = tf2::getYaw(end_point.pose.orientation);
  const auto interpolated_non_fixed_traj_points = util::getConnectedInterpolatedPoints(
    non_fixed_traj_points, traj_param_ptr_->delta_arc_length_for_optimization, begin_yaw, end_yaw);

  // concat interpolated_non_fixed and fixed traj points
  auto extended_points = interpolated_non_fixed_traj_points;
  extended_points.insert(extended_points.end(), fixed_traj_points.begin(), fixed_traj_points.end());

  debug_data_.extended_non_fixed_traj = interpolated_non_fixed_traj_points;
  debug_data_.extended_fixed_traj = fixed_traj_points;

  // TODO(murooka) deal with 1e10
  for (auto & point : extended_points) {
    point.longitudinal_velocity_mps = 1e10;
  }

  return extended_points;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::generatePostProcessedTrajectory(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & optimized_traj_points)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic();

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> trajectory_points;
  if (path_points.empty()) {
    autoware_auto_planning_msgs::msg::TrajectoryPoint tmp_point;
    tmp_point.pose = current_ego_pose_;
    tmp_point.longitudinal_velocity_mps = 0.0;
    trajectory_points.push_back(tmp_point);
    return trajectory_points;
  }
  if (optimized_traj_points.empty()) {
    trajectory_points = util::convertToTrajectoryPoints(path_points);
    return trajectory_points;
  }

  // calculate extended trajectory that connects to optimized trajectory smoothly
  stop_watch.tic();
  const auto extended_traj_points = getExtendedOptimizedTrajectory(path_points, optimized_traj_points);
  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "      getExtendedOptimizedTrajectory:= %f [ms]", stop_watch.toc() * 1000.0);

  // concat trajectories
  stop_watch.tic();
  const auto full_traj_points = util::concatTraj(optimized_traj_points, extended_traj_points);
  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    concatTraj:= %f [ms]", stop_watch.toc() * 1000.0);

  // Basically not changing the shape of trajectory,
  // re-calculate position and velocity with interpolation
  const auto full_traj_points_with_vel = reCalcTrajectoryPoints(path_points, full_traj_points);

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    generatePostProcessedTrajectory:= %f [ms]", stop_watch.toc() * 1000.0);

  return full_traj_points_with_vel;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>
ObstacleAvoidancePlanner::reCalcTrajectoryPoints(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points) const
{
  tier4_autoware_utils::StopWatch stop_watch;

  const auto fine_traj_points =
    [&]() {
      // interpolate pose (x, y, and yaw)
      auto interpolated_traj_points = util::getInterpolatedTrajectoryPoints(
        traj_points, traj_param_ptr_->delta_arc_length_for_trajectory);

      // compensate last pose
      util::compensateLastPose(path_points.back(), *traj_param_ptr_, interpolated_traj_points);

      return interpolated_traj_points;
    }();

  // search zero velocity index of fine_traj_points
  const size_t zero_vel_path_idx = searchExtendedZeroVelocityIndex(fine_traj_points, path_points);
  const size_t zero_vel_traj_idx = searchExtendedZeroVelocityIndex(fine_traj_points, traj_points);
  const size_t zero_vel_fine_traj_idx = std::min(zero_vel_path_idx, zero_vel_traj_idx);

  // search nearest velocity to path point, and fill in result trajectory
  stop_watch.tic();
  const auto re_calc_traj_points = util::alignVelocityWithPoints(
    fine_traj_points, path_points, zero_vel_fine_traj_idx);
  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    alignVelocity:= %f [ms]", stop_watch.toc() * 1000.0);

  return re_calc_traj_points;
}

void ObstacleAvoidancePlanner::publishDebugData(
  const DebugData & debug_data_, const autoware_auto_planning_msgs::msg::Path & path,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const VehicleParam & vehicle_param)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic("total");

  {  // publish trajectories
    auto traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_.foa_data.avoiding_traj_points);
    traj.header = path.header;
    avoiding_traj_pub_->publish(traj);

    auto debug_eb_traj = tier4_autoware_utils::convertToTrajectory(debug_data_.eb_traj);
    debug_eb_traj.header = path.header;
    debug_eb_traj_pub_->publish(debug_eb_traj);

    auto debug_extended_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_.extended_fixed_traj);
    debug_extended_fixed_traj.header = path.header;
    debug_extended_fixed_traj_pub_->publish(debug_extended_fixed_traj);

    auto debug_extended_non_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_.extended_non_fixed_traj);
    debug_extended_non_fixed_traj.header = path.header;
    debug_extended_non_fixed_traj_pub_->publish(debug_extended_non_fixed_traj);

    auto debug_mpt_fixed_traj =
      tier4_autoware_utils::convertToTrajectory(debug_data_.mpt_fixed_traj);
    debug_mpt_fixed_traj.header = path.header;
    debug_mpt_fixed_traj_pub_->publish(debug_mpt_fixed_traj);

    auto debug_mpt_ref_traj = tier4_autoware_utils::convertToTrajectory(debug_data_.mpt_ref_traj);
    debug_mpt_ref_traj.header = path.header;
    debug_mpt_ref_traj_pub_->publish(debug_mpt_ref_traj);

    auto debug_mpt_traj = tier4_autoware_utils::convertToTrajectory(debug_data_.mpt_traj);
    debug_mpt_traj.header = path.header;
    debug_mpt_traj_pub_->publish(debug_mpt_traj);

    tier4_planning_msgs::msg::IsAvoidancePossible is_avoidance_possible;
    is_avoidance_possible.is_avoidance_possible = debug_data_.foa_data.is_avoidance_possible;
    is_avoidance_possible_pub_->publish(is_avoidance_possible);
  }

  {  // publish markers
    std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> traj_points_debug = traj_points;
    // Add z information for virtual wall
    if (!traj_points_debug.empty()) {
      const auto opt_idx = tier4_autoware_utils::findNearestIndex(
        path.points, traj_points.back().pose, std::numeric_limits<double>::max(),
        traj_param_ptr_->delta_yaw_threshold_for_closest_point);
      const int idx = opt_idx ? *opt_idx : 0;
      traj_points_debug.back().pose.position.z = path.points.at(idx).pose.position.z + 1.0;
    }

    stop_watch.tic();
    const auto & debug_marker = debug_visualization::getDebugVisualizationMarker(
      debug_data_, traj_points_debug, vehicle_param, is_showing_debug_info_, false);
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "      getDebugVisualizationMarker:= %f [ms]", stop_watch.toc() * 1000.0);

    stop_watch.tic();
    debug_markers_pub_->publish(debug_marker);
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "      debug_markers publish:= %f [ms]", stop_watch.toc() * 1000.0);
  }

  {  // publish clearance map
    stop_watch.tic();
    if (is_publishing_area_with_objects_) {  // false
      debug_area_with_objects_pub_->publish(debug_visualization::getDebugCostmap(
        debug_data_.area_with_objects_map, path.drivable_area));
    }
    if (is_publishing_object_clearance_map_) {  // false
      debug_object_clearance_map_pub_->publish(debug_visualization::getDebugCostmap(
        debug_data_.only_object_clearance_map, path.drivable_area));
    }
    if (is_publishing_clearance_map_) {  // true
      debug_clearance_map_pub_->publish(
        debug_visualization::getDebugCostmap(debug_data_.clearance_map, path.drivable_area));
    }
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "      getDebugCostMap * 3:= %f [ms]", stop_watch.toc() * 1000.0);
  }

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    publishDebugData:= %f [ms]", stop_watch.toc("total") * 1000.0);
}

void ObstacleAvoidancePlanner::calcTrajectoryInsideDrivableArea(
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
  const CVMaps & cv_maps)
{
  const auto & map_info = cv_maps.map_info;
  const auto & road_clearance_map = cv_maps.clearance_map;

  const size_t nearest_idx = tier4_autoware_utils::findNearestIndex(traj_points, current_ego_pose_.position);
  for (size_t i = nearest_idx; i < traj_points.size(); ++i) {
    const auto & traj_point = traj_points.at(i);

    const bool is_outside =
      [&]() {
        if (use_footprint_for_drivability_) {
          return process_cv::isOutsideDrivableAreaFromRectangleFootprint(traj_point, road_clearance_map, map_info, *vehicle_param_ptr_);
        }
        return process_cv::isOutsideDrivableAreaFromCirclesFootprint(traj_point,
                                                  road_clearance_map, map_info,
                                                  mpt_param_ptr_->avoiding_circle_offsets, mpt_param_ptr_->avoiding_circle_radius);
      }();

    if (is_outside) {
      traj_points[i].longitudinal_velocity_mps = 0.0;
      debug_data_.stop_pose_by_drivable_area = traj_points[i].pose;
      return;
    }
  }
}

Trajectories ObstacleAvoidancePlanner::getPrevTrajs(
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const
{
  if (prev_optimal_trajs_ptr_) {
    return *prev_optimal_trajs_ptr_;
  }

  const auto traj = util::convertToTrajectoryPoints(path_points);
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
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic();

  const auto post_processed_smoothed_traj =
    generatePostProcessedTrajectory(path_points, trajs.smoothed_trajectory);
  // TODO(murooka) generatePoseProcessedTrajectory may be too large
  Trajectories trajectories;
  trajectories.smoothed_trajectory = post_processed_smoothed_traj;
  trajectories.mpt_ref_points = trajs.mpt_ref_points;
  trajectories.model_predictive_trajectory = trajs.model_predictive_trajectory;

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
    "    makePrevTrajectories:= %f [ms]", stop_watch.toc() * 1000.0);

  return trajectories;
}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ObstacleAvoidancePlanner)

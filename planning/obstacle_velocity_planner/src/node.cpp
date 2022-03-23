// Copyright 2022 Tier IV, Inc.
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

#include "obstacle_velocity_planner/node.hpp"

#include "obstacle_velocity_planner/polygon_utils.hpp"
#include "obstacle_velocity_planner/utils.hpp"

#include <tier4_autoware_utils/ros/update_param.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <tier4_autoware_utils/trajectory/tmp_conversion.hpp>

#include <algorithm>
#include <chrono>

namespace
{
template <class T>
size_t getIndexWithLongitudinalOffset(
  const T & points, const double longitudinal_offset, boost::optional<size_t> start_idx)
{
  if (points.empty()) {
    throw std::logic_error("points is empty.");
  }

  if (start_idx) {
    if (start_idx.get() < 0 || points.size() <= start_idx.get()) {
      throw std::out_of_range("start_idx is out of range.");
    }
  } else {
    if (longitudinal_offset > 0) {
      start_idx = 0;
    } else {
      start_idx = points.size() - 1;
    }
  }

  double sum_length = 0.0;
  if (longitudinal_offset > 0) {
    for (size_t i = start_idx.get(); i < points.size() - 1; ++i) {
      const double segment_length =
        tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i + 1));
      sum_length += segment_length;
      if (sum_length >= longitudinal_offset) {
        const double front_length = segment_length;
        const double back_length = sum_length - longitudinal_offset;
        if (front_length < back_length) {
          return i;
        } else {
          return i + 1;
        }
      }
    }
    return points.size() - 1;
  }

  for (size_t i = start_idx.get(); i > 0; --i) {
    const double segment_length =
      tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i + 1));
    sum_length += segment_length;
    if (sum_length >= -longitudinal_offset) {
      const double front_length = segment_length;
      const double back_length = sum_length + longitudinal_offset;
      if (front_length < back_length) {
        return i;
      } else {
        return i + 1;
      }
    }
  }
  return 0;
}
}  // namespace

/*
geometry_msgs::msg::Point operator+(
  const geometry_msgs::msg::Point & front_pos, const geometry_msgs::msg::Point & back_pos)
{
  geometry_msgs::msg::Point added_pos;
  added_pos.x = front_pos.x + back_pos.x;
  added_pos.y = front_pos.y + back_pos.y;
  added_pos.z = front_pos.z + back_pos.z;
  return added_pos;
}

geometry_msgs::msg::Point operator*(
  const geometry_msgs::msg::Point & pos, const double val)
{
  geometry_msgs::msg::Point added_pos;
  added_pos.x = pos.x * val;
  added_pos.y = pos.y * val;
  added_pos.z = pos.z * val;
  return added_pos;
}

geometry_msgs::msg::Point lerpPoint(const geometry_msgs::msg::Point & front_pos, const
geometry_msgs::msg::Point & back_pos, const double ratio)
{
  return front_pos * ratio + back_pos * (1 - ratio);
}

template <class T>
geometry_msgs::msg::Point calcInterpolatedPointWithLongitudinalOffset(const std::vector<T> & points,
const double longitudinal_offset, const boost::optional<size_t> start_idx)
{
  if (points.empty()) {
    throw std::logic_error("points is empty.");
  }

  if (start_idx) {
    if (start_idx.get() < 0 || points.size() <= start_idx.get()) {
      throw std::out_of_range("start_idx is out of range.");
    }
  } else {
    if (longitudinal_offset > 0) {
      start_idx = 0;
    } else {
      start_idx = points.size() - 1;
    }
  }

  double sum_length = 0.0;
  if (longitudinal_offset > 0) {
    for (size_t i = start_idx; i < points.size() - 1; ++i) {
      const double segment_length = tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i +
1)); sum_length += segment_length; if (sum_length >= longitudinal_offset) { const double
front_length = segment_length - (sum_length - longitudinal_offset); return
lerpPoint(tier4_autoware_utils::getPoint(points.at(i)),tier4_autoware_utils::getPoint(points.at(i +
1)), front_length / segment_length);
      }
    }
    return tier4_autoware_utils::getPoint(points.back());
  }

  for (size_t i = start_idx; i > 0; --i) {
    const double segment_length = tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i +
1)); sum_length += segment_length; if (sum_length >= -longitudinal_offset) { const double
front_length = segment_length - (sum_length + longitudinal_offset); return
lerpPoint(tier4_autoware_utils::getPoint(points.at(i)),tier4_autoware_utils::getPoint(points.at(i +
1)), front_length / segment_length);
    }
  }
  return tier4_autoware_utils::getPoint(points.front());
}
*/

ObstacleVelocityPlanner::ObstacleVelocityPlanner(const rclcpp::NodeOptions & node_options)
: Node("obstacle_velocity_planner", node_options),
  self_pose_listener_(this),
  pid_controller_(PIDController(0.003, 0.0, 0.01))  // TODO(murooka) use rosparam
{
  using std::placeholders::_1;

  // Subscriber
  trajectory_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/input/trajectory", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::trajectoryCallback, this, _1));
  smoothed_trajectory_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "/planning/scenario_planning/trajectory", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::smoothedTrajectoryCallback, this, _1));
  objects_sub_ = create_subscription<autoware_auto_perception_msgs::msg::PredictedObjects>(
    "/perception/object_recognition/objects", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::objectsCallback, this, _1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/localization/kinematic_state", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::odomCallback, this, std::placeholders::_1));
  sub_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "/map/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ObstacleVelocityPlanner::mapCallback, this, std::placeholders::_1));
  /*
  sub_external_velocity_limit_ = create_subscription<tier4_planning_msgs::msg::VelocityLimit>(
    "/planning/scenario_planning/max_velocity", 1,
    std::bind(&ObstacleVelocityPlanner::onExternalVelocityLimit, this, _1));
  */

  // Publisher
  trajectory_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/output/trajectory", 1);
  external_vel_limit_pub_ = create_publisher<tier4_planning_msgs::msg::VelocityLimit>(
    "/planning/scenario_planning/max_velocity", 1);
  debug_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/marker", 1);
  debug_wall_marker_pub_ =
    create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/wall_marker", 1);
  debug_rss_wall_marker_pub_ =
    create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/rss_wall_marker", 1);

  // Obstacle
  in_objects_ptr_ = std::make_unique<autoware_auto_perception_msgs::msg::PredictedObjects>();

  // Vehicle Parameters
  vehicle_info_ = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();

  // Parameters
  max_accel_ = declare_parameter("max_accel", 1.0);
  min_accel_ = declare_parameter("min_accel", -1.0);
  max_jerk_ = declare_parameter("max_jerk", 1.0);
  min_jerk_ = declare_parameter("min_jerk", -1.0);
  min_object_accel_ = declare_parameter("min_object_accel", -3.0);
  t_idling_ = declare_parameter("t_idling", 2.0);

  // Parameters for OptimizationBasedPlanner
  const double resampling_s_interval = declare_parameter("resampling_s_interval", 1.0);
  const double max_trajectory_length = declare_parameter("max_trajectory_length", 200.0);
  const double dense_resampling_time_interval =
    declare_parameter("dense_resampling_time_interval", 0.1);
  const double sparse_resampling_time_interval =
    declare_parameter("sparse_resampling_time_interval", 0.5);
  const double dense_time_horizon = declare_parameter("dense_time_horizon", 5.0);
  const double max_time_horizon = declare_parameter("max_time_horizon", 15.0);

  const double delta_yaw_threshold_of_nearest_index =
    tier4_autoware_utils::deg2rad(declare_parameter("delta_yaw_threshold_of_nearest_index", 60.0));
  const double delta_yaw_threshold_of_object_and_ego = tier4_autoware_utils::deg2rad(
    declare_parameter("delta_yaw_threshold_of_object_and_ego", 180.0));
  const double object_zero_velocity_threshold =
    declare_parameter("object_zero_velocity_threshold", 1.5);
  const double object_low_velocity_threshold =
    declare_parameter("object_low_velocity_threshold", 3.0);
  const double external_velocity_limit = declare_parameter("external_velocity_limit", 20.0);
  const double collision_time_threshold = declare_parameter("collision_time_threshold", 10.0);
  const double safe_distance_margin = declare_parameter("safe_distance_margin", 2.0);
  const double t_dangerous = declare_parameter("t_dangerous", 0.5);
  const double initial_velocity_margin = declare_parameter("initial_velocity_margin", 0.2);
  const bool enable_adaptive_cruise = declare_parameter("enable_adaptive_cruise", true);
  const bool use_object_acceleration = declare_parameter("use_object_acceleration", true);
  const bool use_hd_map = declare_parameter("use_hd_map", true);

  const double replan_vel_deviation = declare_parameter("replan_vel_deviation", 5.53);
  const double engage_velocity = declare_parameter("engage_velocity", 0.25);
  const double engage_acceleration = declare_parameter("engage_acceleration", 0.1);
  const double engage_exit_ratio = declare_parameter("engage_exit_ratio", 0.5);
  const double stop_dist_to_prohibit_engage =
    declare_parameter("stop_dist_to_prohibit_engage", 0.5);

  // Velocity Optimizer
  const double max_s_weight = declare_parameter("max_s_weight", 10.0);
  const double max_v_weight = declare_parameter("max_v_weight", 100.0);
  const double over_s_safety_weight = declare_parameter("over_s_safety_weight", 1000000.0);
  const double over_s_ideal_weight = declare_parameter("over_s_ideal_weight", 800.0);
  const double over_v_weight = declare_parameter("over_v_weight", 500000.0);
  const double over_a_weight = declare_parameter("over_a_weight", 1000.0);
  const double over_j_weight = declare_parameter("over_j_weight", 50000.0);

  // Wait for first self pose
  self_pose_listener_.waitForFirstPose();

  optimization_based_planner_ptr_ = std::make_unique<OptimizationBasedPlanner>(
    *this, max_accel_, min_accel_, max_jerk_, min_jerk_, min_object_accel_, t_idling_,
    resampling_s_interval, max_trajectory_length, dense_resampling_time_interval,
    sparse_resampling_time_interval, dense_time_horizon, max_time_horizon,
    delta_yaw_threshold_of_nearest_index, delta_yaw_threshold_of_object_and_ego,
    object_zero_velocity_threshold, object_low_velocity_threshold, external_velocity_limit,
    collision_time_threshold, safe_distance_margin, t_dangerous, initial_velocity_margin,
    enable_adaptive_cruise, use_object_acceleration, use_hd_map, replan_vel_deviation,
    engage_velocity, engage_acceleration, engage_exit_ratio, stop_dist_to_prohibit_engage,
    max_s_weight, max_v_weight, over_s_safety_weight, over_s_ideal_weight, over_v_weight,
    over_a_weight, over_j_weight);
}

void ObstacleVelocityPlanner::mapCallback(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  auto lanelet_map_ptr = std::make_shared<lanelet::LaneletMap>();
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr;

  RCLCPP_INFO(get_logger(), "[Obstacle Velocity Planner]: Start loading lanelet");
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr, &traffic_rules_ptr, &routing_graph_ptr);

  if (optimization_based_planner_ptr_) {
    optimization_based_planner_ptr_->setMaps(lanelet_map_ptr, traffic_rules_ptr, routing_graph_ptr);
    RCLCPP_INFO(get_logger(), "[Obstacle Velocity Planner]: Map is loaded");
  }
}

void ObstacleVelocityPlanner::objectsCallback(
  const autoware_auto_perception_msgs::msg::PredictedObjects::SharedPtr msg)
{
  in_objects_ptr_ = msg;
}

void ObstacleVelocityPlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_twist_ptr_ = std::make_unique<geometry_msgs::msg::TwistStamped>();
  current_twist_ptr_->header = msg->header;
  current_twist_ptr_->twist = msg->twist.twist;
}

void ObstacleVelocityPlanner::smoothedTrajectoryCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  optimization_based_planner_ptr_->setSmoothedTrajectory(msg);
}

/*
void ObstacleVelocityPlanner::onExternalVelocityLimit(
  const tier4_planning_msgs::msg::VelocityLimit::ConstSharedPtr msg)
{
  external_velocity_limit_ = msg->max_velocity;
}
*/

void ObstacleVelocityPlanner::trajectoryCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic();

  std::lock_guard<std::mutex> lock(mutex_);

  if (msg->points.empty() || !current_twist_ptr_ || !in_objects_ptr_) {
    return;
  }

  // convert obstacle type
  const auto obstacles = TargetObstacle::convertToTargetObstacles(*in_objects_ptr_);

  // Prepare algorithmic data
  ObstacleVelocityPlannerData planner_data;
  planner_data.current_time = get_clock()->now();
  planner_data.traj = *msg;
  planner_data.current_pose = self_pose_listener_.getCurrentPose()->pose;
  planner_data.current_vel = current_twist_ptr_->twist.linear.x;
  // planner_data.external_velocity_limit = external_velocity_limit_;
  planner_data.target_obstacles = filterObstacles(
    obstacles, planner_data.traj, planner_data.current_pose, planner_data.current_vel);

  // generate Trajectory
  const Method method = Method::OPTIMIZATION_BASE;
  const auto output = [&]() {
    if (method == Method::OPTIMIZATION_BASE) {
      return optimization_based_planner_ptr_->generateOptimizationTrajectory(planner_data);
    } else if (method == Method::RULE_BASE) {
      return generateRuleBaseTrajectory(planner_data);
    }

    std::logic_error("Designated method is not supported.");
  }();

  // Publish trajectory
  trajectory_pub_->publish(output);

  const double elapsed_time = stop_watch.toc();
  // RCLCPP_WARN_STREAM(get_logger(), elapsed_time);
}

std::vector<TargetObstacle> ObstacleVelocityPlanner::filterObstacles(
  const std::vector<TargetObstacle> & obstacles,
  const autoware_auto_planning_msgs::msg::Trajectory & traj,
  const geometry_msgs::msg::Pose & current_pose, const double current_vel)
{
  std::vector<TargetObstacle> target_obstacles;

  // TODO(murooka) parametrise these parameters
  constexpr double margin_between_traj_and_obstacle = 0.3;
  constexpr double min_obstacle_velocity = 3.0;  // 10.8 [km/h]
  constexpr double margin_for_collision_time = 3.0;
  constexpr double max_ego_obj_overlap_time = 1.0;
  constexpr double max_prediction_time_for_collision_check = 20.0;

  const auto traj_polygons = polygon_utils::createOneStepPolygons(traj, vehicle_info_);

  for (const auto & obstacle : obstacles) {
    const auto first_within_idx = polygon_utils::getFirstCollisionIndex(
      traj_polygons, polygon_utils::convertObstacleToPolygon(obstacle.pose, obstacle.shape),
      margin_between_traj_and_obstacle);

    if (first_within_idx) {  // obsacles inside the trajectory
      if (
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::CAR ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::BUS ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE) {  // vehicle
                                                                                   // obstacle

        if (std::abs(obstacle.velocity) > min_obstacle_velocity) {  // running obstacle
          const double time_to_collision = [&]() {
            // TODO(murooka) consider obstacle width/length the same as
            // vehicle_info_.max_longitudinal_offset_m
            const double dist_to_ego =
              tier4_autoware_utils::calcSignedArcLength(
                traj.points, current_pose.position, first_within_idx.get()) -
              vehicle_info_.max_longitudinal_offset_m;
            return dist_to_ego / std::max(1e-6, current_vel);
          }();

          const double time_to_obstacle_getting_out = [&]() {
            const auto obstacle_getting_out_idx = polygon_utils::getFirstNonCollisionIndex(
              traj_polygons, obstacle.predicted_paths.at(0), obstacle.shape, first_within_idx.get(),
              margin_between_traj_and_obstacle);
            if (!obstacle_getting_out_idx) {
              return std::numeric_limits<double>::max();
            }

            const double dist_to_obstacle_getting_out = tier4_autoware_utils::calcSignedArcLength(
              traj.points, obstacle.pose.position, obstacle_getting_out_idx.get());
            return dist_to_obstacle_getting_out / obstacle.velocity;
          }();

          /*
          RCLCPP_ERROR_STREAM(
            get_logger(), first_within_idx.get()
                            << "/" << traj.points.size() << " " << time_to_collision << " < "
                            << time_to_obstacle_getting_out);
          */

          if (time_to_collision > time_to_obstacle_getting_out + margin_for_collision_time) {
            RCLCPP_INFO_EXPRESSION(
              get_logger(), true, "Ignore obstacles since it will not collide with the ego.");
            // False Condition 1. Ignore vehicle obstacles inside the trajectory, which is running
            // and does not collide with ego in a certain time.
            // continue;
          }
        }
      }
    } else {  // obstacles outside the trajectory
      const double max_dist =
        3.0;  // std::max(vehicle_info_.max_longitudinal_offset_m, vehicle_info_.rear_overhang) +
              // std::max(shape.dimensions.x, shape.dimensions.y) / 2.0;
      const bool will_collide = polygon_utils::willCollideWithSurroundObstacle(
        traj, traj_polygons, obstacle.predicted_paths.at(0), obstacle.shape,
        margin_between_traj_and_obstacle, max_dist, max_ego_obj_overlap_time,
        max_prediction_time_for_collision_check);
      if (!will_collide) {
        // False Condition 2. Ignore vehicle obstacles outside the trajectory, whose predicted path
        // overlaps the ego trajectory in a certain time.
        continue;
      }
    }

    target_obstacles.push_back(obstacle);
  }

  // TODO(murooka) change shape of markers based on its shape
  // publish filtered obstacles
  visualization_msgs::msg::MarkerArray object_msg;
  for (size_t i = 0; i < target_obstacles.size(); ++i) {
    const auto marker = obstacle_velocity_utils::getObjectMarkerArray(
      target_obstacles.at(i).pose, i, "target_objects", 0.7, 0.7, 0.0);
    tier4_autoware_utils::appendMarkerArray(marker, &object_msg);
  }
  debug_marker_pub_->publish(object_msg);

  return target_obstacles;
}

autoware_auto_planning_msgs::msg::Trajectory ObstacleVelocityPlanner::generateRuleBaseTrajectory(
  const ObstacleVelocityPlannerData & planner_data)
{
  auto output_traj = planner_data.traj;
  tier4_planning_msgs::msg::VelocityLimit vel_limit_msg;
  vel_limit_msg.max_velocity = 12.0;  // TODO(murooka)

  const size_t ego_idx =
    tier4_autoware_utils::findNearestIndex(output_traj.points, planner_data.current_pose.position);

  // TODO(murooka) use ros param
  constexpr double max_obj_velocity_for_stop = 1.0;
  constexpr double safe_distance_margin = 4.0;

  // boost::optional<double> min_dist_to_rss_wall;
  boost::optional<double> min_dist_to_stop;
  boost::optional<double> min_dist_to_slow_down;
  for (const auto & obstacle : planner_data.target_obstacles) {
    if (std::abs(obstacle.velocity) < max_obj_velocity_for_stop) {  // stop
      // calculate distance to stop
      const double distance_to_stop_for_obstacle = tier4_autoware_utils::calcSignedArcLength(
        output_traj.points, planner_data.current_pose.position, obstacle.pose.position);
      const double distance_to_stop_with_acc_limit = [&]() {
        constexpr double epsilon = 1e-6;
        if (planner_data.current_vel < epsilon) {
          return 0.0;
        }

        constexpr double strong_min_accel = -2.0;  // TODO(murooka)
        const double time_to_stop_with_acc_limit = -planner_data.current_vel / strong_min_accel;
        return planner_data.current_vel * time_to_stop_with_acc_limit + strong_min_accel +
               std::pow(time_to_stop_with_acc_limit, 2);
      }();

      const double distance_to_stop =
        std::max(distance_to_stop_for_obstacle, distance_to_stop_with_acc_limit) -
        safe_distance_margin;
      if (!min_dist_to_stop || distance_to_stop < min_dist_to_stop.get()) {
        min_dist_to_stop = distance_to_stop;
      }
    } else {  // adaptive cruise
      // calculate distance between ego and obstacle based on RSS
      const double rss_dist =
        planner_data.current_vel * t_idling_ + 0.5 * max_accel_ * std::pow(t_idling_, 2) +
        std::pow(planner_data.current_vel + max_accel_ * t_idling_, 2) * 0.5 /
          std::abs(min_accel_) -
        std::pow(obstacle.velocity, 2) * 0.5 / std::abs(min_object_accel_) + safe_distance_margin;
      std::cerr << planner_data.current_vel << " " << obstacle.velocity << " " << rss_dist
                << std::endl;
      const double rss_dist_with_vehicle_offset =
        rss_dist + vehicle_info_.max_longitudinal_offset_m + obstacle.shape.dimensions.x / 2.0;

      // calculate current obstacle pose
      const auto current_obstacle_pose =
        obstacle_velocity_utils::getCurrentObjectPoseFromPredictedPath(
          obstacle.predicted_paths.at(0), obstacle.time_stamp, get_clock()->now());
      if (!current_obstacle_pose) {
        continue;
      }

      // calculate error distance
      const double actual_dist = tier4_autoware_utils::calcSignedArcLength(
        output_traj.points, planner_data.current_pose.position, current_obstacle_pose->position);
      const double error_dist = actual_dist - rss_dist_with_vehicle_offset;

      if (!min_dist_to_slow_down || error_dist < min_dist_to_slow_down.get()) {
        min_dist_to_slow_down = error_dist;
      }
    }
  }

  if (min_dist_to_stop) {
    RCLCPP_INFO_EXPRESSION(get_logger(), true, "stop planning");

    // insert zero velocity
    // TODO(murooka) Should I use interpolation?
    const size_t stop_idx = getIndexWithLongitudinalOffset(
      output_traj.points, min_dist_to_stop.get() - vehicle_info_.max_longitudinal_offset_m,
      ego_idx);
    for (size_t i = stop_idx; i < output_traj.points.size(); ++i) {
      output_traj.points.at(i).longitudinal_velocity_mps = 0.0;
    }

    // virtual wall marker for stop
    const auto marker_pose = obstacle_velocity_utils::calcForwardPose(
      output_traj, planner_data.current_pose.position, min_dist_to_stop.get());
    if (marker_pose) {
      visualization_msgs::msg::MarkerArray wall_msg;
      const auto markers = tier4_autoware_utils::createStopVirtualWallMarker(
        marker_pose.get(), "obstacle to stop", get_clock()->now(), 0);
      tier4_autoware_utils::appendMarkerArray(markers, &wall_msg);

      // publish wall marker
      debug_wall_marker_pub_->publish(wall_msg);
    }
  }

  if (min_dist_to_slow_down) {
    RCLCPP_INFO_EXPRESSION(get_logger(), true, "slow down planning");

    // adaptive cruise TODO
    // calculate target velocity with acceleration limit by PID controller
    const double diff_vel = pid_controller_.calc(min_dist_to_slow_down.get());
    const double prev_vel = prev_target_vel_ ? prev_target_vel_.get() : planner_data.current_vel;
    const double target_vel_with_acc_limit =
      prev_vel + std::max(
                   std::min(diff_vel, max_accel_ * 0.1),
                   min_accel_ * 0.1);  // TODO(murooka) accel * 0.1 (time step)
    RCLCPP_INFO_EXPRESSION(get_logger(), true, "target_velocity %f", target_vel_with_acc_limit);

    prev_target_vel_ = target_vel_with_acc_limit;

    // set velocity limit
    if (target_vel_with_acc_limit < vel_limit_msg.max_velocity) {
      vel_limit_msg.max_velocity = target_vel_with_acc_limit;
    }

    // virtual wall marker for slow down
    const double dist_to_rss_wall =
      min_dist_to_slow_down.get() + vehicle_info_.max_longitudinal_offset_m;
    const size_t wall_idx =
      getIndexWithLongitudinalOffset(output_traj.points, dist_to_rss_wall, ego_idx);

    visualization_msgs::msg::MarkerArray rss_wall_msg;

    const auto markers = tier4_autoware_utils::createSlowDownVirtualWallMarker(
      output_traj.points.at(wall_idx).pose, "rss distance", get_clock()->now(), 0);
    tier4_autoware_utils::appendMarkerArray(markers, &rss_wall_msg);

    debug_rss_wall_marker_pub_->publish(rss_wall_msg);
  } else {
    // reset previous target velocity if adaptive cruise is not enabled
    prev_target_vel_ = {};
  }

  // publish velocity limit
  external_vel_limit_pub_->publish(vel_limit_msg);

  return output_traj;
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ObstacleVelocityPlanner)

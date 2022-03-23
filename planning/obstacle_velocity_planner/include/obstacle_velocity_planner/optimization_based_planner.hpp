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

#ifndef OBSTACLE_VELOCITY_PLANNER__OPTIMIZATION_BASED_PLANNER_HPP_
#define OBSTACLE_VELOCITY_PLANNER__OPTIMIZATION_BASED_PLANNER_HPP_

#include "obstacle_velocity_planner/box2d.hpp"
#include "obstacle_velocity_planner/common/s_boundary.hpp"
#include "obstacle_velocity_planner/common/st_point.hpp"
#include "obstacle_velocity_planner/common_structs.hpp"
#include "obstacle_velocity_planner/velocity_optimizer.hpp"

#include <lanelet2_extension/utility/message_conversion.hpp>
#include <lanelet2_extension/utility/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/system/stop_watch.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <tier4_debug_msgs/msg/float32_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <boost/optional.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/BoundingBox.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <memory>
#include <tuple>
#include <vector>

class OptimizationBasedPlanner
{
public:
  OptimizationBasedPlanner(
    rclcpp::Node & node, const double max_accel, const double min_accel, const double max_jerk,
    const double min_jerk, const double min_object_accel, const double t_idling,
    const double resampling_s_interval, const double max_trajectory_length,
    const double dense_resampling_time_interval, const double sparse_resampling_time_interval,
    const double dense_time_horizon, const double max_time_horizon,
    const double delta_yaw_threshold_of_nearest_index,
    const double delta_yaw_threshold_of_object_and_ego, const double object_zero_velocity_threshold,
    const double object_low_velocity_threshold, const double external_velocity_limit,
    const double collision_time_threshold, const double safe_distance_margin,
    const double t_dangerous, const double initial_velocity_margin,
    const bool enable_adaptive_cruise, const bool use_object_acceleration, const bool use_hd_map,
    const double replan_vel_deviation, const double engage_velocity,
    const double engage_acceleration, const double engage_exit_ratio,
    const double stop_dist_to_prohibit_engage, const double max_s_weight, const double max_v_weight,
    const double over_s_safety_weight, const double over_s_ideal_weight, const double over_v_weight,
    const double over_a_weight, const double over_j_weight)
  : max_accel_(max_accel),
    min_accel_(min_accel),
    max_jerk_(max_jerk),
    min_jerk_(min_jerk),
    min_object_accel_(min_object_accel),
    t_idling_(t_idling),
    resampling_s_interval_(resampling_s_interval),
    max_trajectory_length_(max_trajectory_length),
    dense_resampling_time_interval_(dense_resampling_time_interval),
    sparse_resampling_time_interval_(sparse_resampling_time_interval),
    dense_time_horizon_(dense_time_horizon),
    max_time_horizon_(max_time_horizon),
    delta_yaw_threshold_of_nearest_index_(delta_yaw_threshold_of_nearest_index),
    delta_yaw_threshold_of_object_and_ego_(delta_yaw_threshold_of_object_and_ego),
    object_zero_velocity_threshold_(object_zero_velocity_threshold),
    object_low_velocity_threshold_(object_low_velocity_threshold),
    external_velocity_limit_(external_velocity_limit),
    collision_time_threshold_(collision_time_threshold),
    safe_distance_margin_(safe_distance_margin),
    t_dangerous_(t_dangerous),
    initial_velocity_margin_(initial_velocity_margin),
    enable_adaptive_cruise_(enable_adaptive_cruise),
    use_object_acceleration_(use_object_acceleration),
    use_hd_map_(use_hd_map),
    replan_vel_deviation_(replan_vel_deviation),
    engage_velocity_(engage_velocity),
    engage_acceleration_(engage_acceleration),
    engage_exit_ratio_(engage_exit_ratio),
    stop_dist_to_prohibit_engage_(stop_dist_to_prohibit_engage)
  {
    vehicle_info_ = vehicle_info_util::VehicleInfoUtil(node).getVehicleInfo();

    velocity_optimizer_ptr_ = std::make_shared<VelocityOptimizer>(
      max_s_weight, max_v_weight, over_s_safety_weight, over_s_ideal_weight, over_v_weight,
      over_a_weight, over_j_weight);
    optimized_sv_pub_ = node.create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
      "~/optimized_sv_trajectory", 1);
    optimized_st_graph_pub_ = node.create_publisher<autoware_auto_planning_msgs::msg::Trajectory>(
      "~/optimized_st_graph", 1);
    boundary_pub_ =
      node.create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/boundary", 1);
    distance_to_closest_obj_pub_ =
      node.create_publisher<tier4_debug_msgs::msg::Float32Stamped>("~/distance_to_closest_obj", 1);
    debug_calculation_time_ =
      node.create_publisher<tier4_debug_msgs::msg::Float32Stamped>("~/calculation_time", 1);
    debug_wall_marker_pub_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/wall_marker", 1);
  }

  autoware_auto_planning_msgs::msg::Trajectory generateOptimizationTrajectory(
    const ObstacleVelocityPlannerData & planner_data);

  void setMaps(
    const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr,
    const std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr,
    const std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr)
  {
    lanelet_map_ptr_ = lanelet_map_ptr;
    traffic_rules_ptr_ = traffic_rules_ptr;
    routing_graph_ptr_ = routing_graph_ptr;
  }

  void setSmoothedTrajectory(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj)
  {
    smoothed_trajectory_ptr_ = traj;
  }

private:
  struct TrajectoryData
  {
    TrajectoryData() {}

    autoware_auto_planning_msgs::msg::Trajectory traj;
    std::vector<double> s;
  };

  struct ObjectData
  {
    geometry_msgs::msg::Pose pose;
    double length;
    double width;
    double time;
  };

  // Member Functions
  std::vector<double> createTimeVector();

  double getClosestStopDistance(
    const ObstacleVelocityPlannerData & planner_data, const TrajectoryData & ego_traj_data,
    const std::vector<double> & resolutions);

  std::tuple<double, double> calcInitialMotion(
    const double current_vel, const autoware_auto_planning_msgs::msg::Trajectory & input_traj,
    const size_t input_closest, const autoware_auto_planning_msgs::msg::Trajectory & prev_traj,
    const double closest_stop_dist);

  autoware_auto_planning_msgs::msg::TrajectoryPoint calcInterpolatedTrajectoryPoint(
    const autoware_auto_planning_msgs::msg::Trajectory & trajectory,
    const geometry_msgs::msg::Pose & target_pose);

  bool checkHasReachedGoal(
    const autoware_auto_planning_msgs::msg::Trajectory & traj, const size_t closest_idx,
    const double v0);

  TrajectoryData getTrajectoryData(
    const autoware_auto_planning_msgs::msg::Trajectory & traj, const size_t closest_idx);

  TrajectoryData resampleTrajectoryData(
    const TrajectoryData & base_traj_data, const double resampling_s_interval,
    const double max_traj_length, const double stop_dist);

  autoware_auto_planning_msgs::msg::Trajectory resampleTrajectory(
    const std::vector<double> & base_index,
    const autoware_auto_planning_msgs::msg::Trajectory & base_trajectory,
    const std::vector<double> & query_index, const bool use_spline_for_pose = false);

  boost::optional<SBoundaries> getSBoundaries(
    const ObstacleVelocityPlannerData & planner_data, const TrajectoryData & ego_traj_data,
    const std::vector<double> & time_vec);

  boost::optional<SBoundaries> getSBoundaries(
    const rclcpp::Time & current_time, const TrajectoryData & ego_traj_data,
    const TargetObstacle & object, const rclcpp::Time & obj_base_time,
    const std::vector<double> & time_vec);

  boost::optional<SBoundaries> getSBoundaries(
    const TrajectoryData & ego_traj_data, const std::vector<double> & time_vec,
    const double safety_distance, const TargetObstacle & object,
    const double dist_to_collision_point);

  boost::optional<SBoundaries> getSBoundaries(
    const rclcpp::Time & current_time, const TrajectoryData & ego_traj_data,
    const std::vector<double> & time_vec, const double safety_distance,
    const TargetObstacle & object, const rclcpp::Time & obj_base_time,
    const autoware_auto_perception_msgs::msg::PredictedPath & predicted_path);

  bool checkOnMapObject(
    const TargetObstacle & object, const lanelet::ConstLanelets & valid_lanelets);

  lanelet::ConstLanelets getSurroundingLanelets(const geometry_msgs::msg::Pose & current_pose);

  void addValidLanelet(
    const lanelet::routing::LaneletPaths & candidate_paths,
    lanelet::ConstLanelets & valid_lanelets);

  bool checkIsFrontObject(
    const TargetObstacle & object, const autoware_auto_planning_msgs::msg::Trajectory & traj);

  boost::optional<autoware_auto_perception_msgs::msg::PredictedPath> resampledPredictedPath(
    const TargetObstacle & object, const rclcpp::Time & obj_base_time,
    const rclcpp::Time & current_time, const std::vector<double> & resolutions,
    const double horizon);

  boost::optional<geometry_msgs::msg::Pose> calcForwardPose(
    const autoware_auto_planning_msgs::msg::Trajectory & traj,
    const geometry_msgs::msg::Point & point, const double target_length);

  boost::optional<geometry_msgs::msg::Pose> calcForwardPose(
    const TrajectoryData & ego_traj_data, const geometry_msgs::msg::Point & point,
    const double target_length);

  boost::optional<double> getDistanceToCollisionPoint(
    const TrajectoryData & ego_traj_data, const ObjectData & obj_data,
    const double delta_yaw_threshold);

  boost::optional<size_t> getCollisionIdx(
    const TrajectoryData & ego_traj, const Box2d & obj_box, const size_t start_idx,
    const size_t end_idx);

  double getObjectLongitudinalPosition(
    const TrajectoryData & traj_data, const geometry_msgs::msg::Pose & obj_pose);

  geometry_msgs::msg::Pose transformBaseLink2Center(
    const geometry_msgs::msg::Pose & pose_base_link);

  boost::optional<VelocityOptimizer::OptimizationResult> processOptimizedResult(
    const double v0, const VelocityOptimizer::OptimizationResult & opt_result);

  void publishDebugTrajectory(
    const rclcpp::Time & current_time, const autoware_auto_planning_msgs::msg::Trajectory & traj,
    const size_t closest_idx, const std::vector<double> & time_vec,
    const SBoundaries & s_boundaries, const VelocityOptimizer::OptimizationResult & opt_result);

  // Calculation time watcher
  tier4_autoware_utils::StopWatch<std::chrono::milliseconds> stop_watch_;

  // Lanelet Map Pointers
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr_;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr_;
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr_;

  autoware_auto_planning_msgs::msg::Trajectory prev_output_;
  autoware_auto_planning_msgs::msg::Trajectory::SharedPtr smoothed_trajectory_ptr_;

  // Velocity Optimizer
  std::shared_ptr<VelocityOptimizer> velocity_optimizer_ptr_;

  // Vehicle Parameters
  vehicle_info_util::VehicleInfo vehicle_info_;

  // Publisher
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr boundary_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr optimized_sv_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr
    optimized_st_graph_pub_;
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr distance_to_closest_obj_pub_;
  rclcpp::Publisher<tier4_debug_msgs::msg::Float32Stamped>::SharedPtr debug_calculation_time_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_wall_marker_pub_;

  // Parameters
  double max_accel_;
  double min_accel_;
  double max_jerk_;
  double min_jerk_;
  double min_object_accel_;

  // Resampling Parameter
  double resampling_s_interval_;
  double max_trajectory_length_;
  double dense_resampling_time_interval_;
  double sparse_resampling_time_interval_;
  double dense_time_horizon_;
  double max_time_horizon_;

  double delta_yaw_threshold_of_nearest_index_;
  double delta_yaw_threshold_of_object_and_ego_;
  double object_zero_velocity_threshold_;
  double object_low_velocity_threshold_;
  double external_velocity_limit_;
  double collision_time_threshold_;
  double safe_distance_margin_;
  double t_dangerous_;
  double t_idling_;
  double initial_velocity_margin_;
  bool enable_adaptive_cruise_;
  bool use_object_acceleration_;
  bool use_hd_map_;

  double replan_vel_deviation_;
  double engage_velocity_;
  double engage_acceleration_;
  double engage_exit_ratio_;
  double stop_dist_to_prohibit_engage_;
};

#endif  // OBSTACLE_VELOCITY_PLANNER__OPTIMIZATION_BASED_PLANNER_HPP_

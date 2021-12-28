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
#ifndef OBSTACLE_AVOIDANCE_PLANNER__NODE_HPP_
#define OBSTACLE_AVOIDANCE_PLANNER__NODE_HPP_

#include "obstacle_avoidance_planner/common_structs.hpp"
#include "obstacle_avoidance_planner/eb_path_optimizer.hpp"
#include "obstacle_avoidance_planner/mpt_optimizer.hpp"
#include "opencv2/core.hpp"
#include "rclcpp/clock.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tier4_autoware_utils/ros/self_pose_listener.hpp"

#include "autoware_auto_perception_msgs/msg/predicted_objects.hpp"
#include "autoware_auto_planning_msgs/msg/path.hpp"
#include "autoware_auto_planning_msgs/msg/path_point.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory.hpp"
#include "autoware_auto_planning_msgs/msg/trajectory_point.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/map_meta_data.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tier4_planning_msgs/msg/enable_avoidance.hpp"
#include "tier4_planning_msgs/msg/is_avoidance_possible.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "boost/optional.hpp"

#include <memory>
#include <mutex>
#include <vector>

class ObstacleAvoidancePlanner : public rclcpp::Node
{
private:
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  bool is_publishing_area_with_objects_;
  bool is_publishing_object_clearance_map_;
  bool is_publishing_clearance_map_;
  bool is_showing_debug_info_;
  bool is_using_vehicle_config_;
  bool is_stopping_if_outside_drivable_area_;
  bool enable_avoidance_;
  bool reset_prev_info_;
  bool use_footprint_for_drivability_;
  std::mutex mutex_;

  // params outside logic
  double min_ego_moving_dist_for_replan_;
  double min_delta_time_sec_for_replan_;
  double max_dist_for_extending_end_point_;
  double distance_for_path_shape_change_detection_;

  // logic
  std::unique_ptr<EBPathOptimizer> eb_path_optimizer_ptr_;
  std::unique_ptr<MPTOptimizer> mpt_optimizer_ptr_;

  // params
  std::unique_ptr<QPParam> qp_param_ptr_;
  std::unique_ptr<TrajectoryParam> traj_param_ptr_;
  std::unique_ptr<ConstrainParam> constrain_param_ptr_;
  std::unique_ptr<VehicleParam> vehicle_param_ptr_;
  std::unique_ptr<MPTParam> mpt_param_ptr_;
  int visualize_sampling_num_;

  // debug
  DebugData debug_data_;

  geometry_msgs::msg::Pose current_ego_pose_;
  std::unique_ptr<geometry_msgs::msg::TwistStamped> current_twist_ptr_;
  std::unique_ptr<geometry_msgs::msg::Pose> prev_ego_pose_ptr_;
  std::unique_ptr<Trajectories> prev_optimal_trajs_ptr_;
  std::unique_ptr<std::vector<autoware_auto_planning_msgs::msg::PathPoint>> prev_path_points_ptr_;
  std::unique_ptr<autoware_auto_perception_msgs::msg::PredictedObjects> in_objects_ptr_;

  std::unique_ptr<rclcpp::Time> prev_replanned_time_ptr_;
  tier4_autoware_utils::SelfPoseListener self_pose_listener_{this};

  // ROS
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr avoiding_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr
    debug_extended_fixed_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr
    debug_extended_non_fixed_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr debug_eb_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr
    debug_mpt_fixed_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr
    debug_mpt_ref_traj_pub_;
  rclcpp::Publisher<autoware_auto_planning_msgs::msg::Trajectory>::SharedPtr debug_mpt_traj_pub_;

  rclcpp::Publisher<tier4_planning_msgs::msg::IsAvoidancePossible>::SharedPtr
    is_avoidance_possible_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_markers_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_clearance_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_object_clearance_map_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr debug_area_with_objects_pub_;
  rclcpp::Subscription<autoware_auto_planning_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<autoware_auto_perception_msgs::msg::PredictedObjects>::SharedPtr
    objects_sub_;
  rclcpp::Subscription<tier4_planning_msgs::msg::EnableAvoidance>::SharedPtr is_avoidance_sub_;
  rclcpp::Clock logger_ros_clock_;

  // callback functions
  void pathCallback(const autoware_auto_planning_msgs::msg::Path::SharedPtr);
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr);
  void objectsCallback(const autoware_auto_perception_msgs::msg::PredictedObjects::SharedPtr);
  void enableAvoidanceCallback(const tier4_planning_msgs::msg::EnableAvoidance::SharedPtr);

  void initializeParam();

  // generate fine trajectory
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> generatePostProcessedTrajectory(
    const geometry_msgs::msg::Pose & ego_pose,
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
    const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & merged_optimized_points);

  bool needReplan(
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const;

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> getExtendedOptimizedTrajectory(
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
    const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & optimized_points);

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> generateOptimizedTrajectory(
    const autoware_auto_planning_msgs::msg::Path & input_path);

  Trajectories calcTrajectories(
    const autoware_auto_planning_msgs::msg::Path & path, const CVMaps & cv_maps);

  autoware_auto_planning_msgs::msg::Trajectory generateTrajectory(
    const autoware_auto_planning_msgs::msg::Path & in_path);

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> reCalcTrajectoryPoints(
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
    const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points) const;

  void publishDebugData(
    const DebugData & debug_data, const autoware_auto_planning_msgs::msg::Path & path,
    const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
    const VehicleParam & vehicle_param);

  void calcTrajectoryInsideArea(
    std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & traj_points,
    const CVMaps & cv_maps);

  Trajectories getPrevTrajs(
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const;

  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> getPrevTrajectory(
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points) const;

  Trajectories makePrevTrajectories(
    const geometry_msgs::msg::Pose & ego_pose,
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
    const Trajectories & trajs);

  void declareObstacleAvoidancePlannerParameters();

  rcl_interfaces::msg::SetParametersResult paramCallback(
    const std::vector<rclcpp::Parameter> & parameters);

public:
  explicit ObstacleAvoidancePlanner(const rclcpp::NodeOptions & node_options);
};

#endif  // OBSTACLE_AVOIDANCE_PLANNER__NODE_HPP_

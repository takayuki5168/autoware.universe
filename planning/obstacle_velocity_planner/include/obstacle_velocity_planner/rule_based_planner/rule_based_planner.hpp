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

#ifndef OBSTACLE_VELOCITY_PLANNER__RULE_BASED_PLANNER__RULE_BASED_PLANNER_HPP_
#define OBSTACLE_VELOCITY_PLANNER__RULE_BASED_PLANNER__RULE_BASED_PLANNER_HPP_

#include "obstacle_velocity_planner/planner_interface.hpp"
#include "obstacle_velocity_planner/rule_based_planner/pid_controller.hpp"

#include "visualization_msgs/msg/marker_array.hpp"

class RuleBasedPlanner : public PlannerInterface
{
public:
  RuleBasedPlanner(
    rclcpp::Node & node, const double max_accel, const double min_accel, const double max_jerk,
    const double min_jerk, const double min_object_accel, const double t_idling,
    const vehicle_info_util::VehicleInfo & vehicle_info)
  : PlannerInterface(
      max_accel, min_accel, max_jerk, min_jerk, min_object_accel, t_idling, vehicle_info),
    pid_controller_(PIDController(0.003, 0.0, 0.01))  // TODO(murooka) use rosparam
  {
    debug_wall_marker_pub_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/wall_marker", 1);
    debug_rss_wall_marker_pub_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/rss_wall_marker", 1);
  }

  autoware_auto_planning_msgs::msg::Trajectory generateTrajectory(
    const ObstacleVelocityPlannerData & planner_data) override;

  boost::optional<double> calcVelocityLimit(
    const ObstacleVelocityPlannerData & planner_data) override;

private:
  PIDController pid_controller_;

  // Publisher
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_wall_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_rss_wall_marker_pub_;

  boost::optional<double> vel_limit_;

  boost::optional<double> prev_target_vel_;
};

#endif  // OBSTACLE_VELOCITY_PLANNER__RULE_BASED_PLANNER__RULE_BASED_PLANNER_HPP_

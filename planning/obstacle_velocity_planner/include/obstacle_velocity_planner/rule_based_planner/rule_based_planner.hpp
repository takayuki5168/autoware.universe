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
#include "obstacle_velocity_planner/rule_based_planner/debug_values.hpp"

#include "visualization_msgs/msg/marker_array.hpp"
#include "tier4_debug_msgs/msg/float32_multi_array_stamped.hpp"

#include <boost/optional.hpp>
#include <memory>
#include <vector>

using tier4_debug_msgs::msg::Float32MultiArrayStamped;

class RuleBasedPlanner : public PlannerInterface
{
public:
  RuleBasedPlanner(
    rclcpp::Node & node, const double max_accel, const double min_accel, const double max_jerk,
    const double min_jerk, const double min_object_accel, const double idling_time,
    const vehicle_info_util::VehicleInfo & vehicle_info);

  boost::optional<size_t> getZeroVelocityIndexWithVelocityLimit(
    const ObstacleVelocityPlannerData & planner_data, boost::optional<VelocityLimit> & vel_limit) override;

  void updateParam(const std::vector<rclcpp::Parameter> & parameters) override;

private:
  // ROS param
  std::unique_ptr<PIDController> pid_controller_;
  double vel_to_acc_weight_;
  double min_slow_down_target_vel_;
  double max_obj_velocity_for_stop_;
  double safe_distance_margin_;
  double strong_min_accel_;

  // Publisher
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_wall_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_rss_wall_marker_pub_;
  rclcpp::Publisher<Float32MultiArrayStamped>::SharedPtr debug_values_pub_;

  boost::optional<double> prev_target_vel_;

  size_t doStop(const ObstacleVelocityPlannerData & planner_data, const double dist_to_stop) const;
  VelocityLimit doSlowDown(const ObstacleVelocityPlannerData & planner_data, const double dist_to_slow_down);

  DebugValues debug_values_;
};

#endif  // OBSTACLE_VELOCITY_PLANNER__RULE_BASED_PLANNER__RULE_BASED_PLANNER_HPP_

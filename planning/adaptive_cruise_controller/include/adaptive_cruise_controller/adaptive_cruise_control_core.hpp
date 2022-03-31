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

#ifndef ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_CORE_HPP_
#define ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_CORE_HPP_

#include <adaptive_cruise_controller/acc_pid.hpp>
#include <adaptive_cruise_controller/utilities.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <nav_msgs/msg/odometry.hpp>

namespace motion_planning
{

using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;
using nav_msgs::msg::Odometry;

class AdaptiveCruiseControlCore
{
public:
  explicit AdaptiveCruiseControlCore(const double baselink2front, const AccParam & acc_param);
  std::shared_ptr<AdaptiveCruiseInformation> getAccInfoPtr() { return acc_info_ptr_; }
  std::shared_ptr<AdaptiveCruiseInformation> getPrevAccInfoPtr() { return prev_acc_info_ptr_; }

  void calcInformationForAdaptiveCruise(
    const TrajectoryPoints & trajectory_points, const geometry_msgs::msg::PoseStamped & pose,
    const Odometry & odometry, const PredictedObject & object, const rclcpp::Time & object_time);

  void calculate();
  bool getTargetMotion(double & target_velocity, double & target_acc, double & target_jerk);
  TrajectoryPoints getAccTrajectory(bool & emergency_flag);
  State getState();
  AccMotion getAccMotion() { return acc_motion_; };

private:
  // parameter
  double baselink2front_;
  AccParam acc_param_;

  // variables
  std::shared_ptr<AccPidNode> acc_pid_node_ptr_;
  double prev_target_velocity_;
  std::shared_ptr<AdaptiveCruiseInformation> acc_info_ptr_;
  std::shared_ptr<AdaptiveCruiseInformation> prev_acc_info_ptr_;
  AccMotion acc_motion_;

  // functions
  double getIdealDistanceToObject(const double current_velocity, const double object_velocity);
};

}  // namespace motion_planning

#endif  // ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_CORE_HPP_

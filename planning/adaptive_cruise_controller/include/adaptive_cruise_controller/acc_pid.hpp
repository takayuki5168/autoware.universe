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

#ifndef ADAPTIVE_CRUISE_CONTROLLER__ACC_PID_HPP_
#define ADAPTIVE_CRUISE_CONTROLLER__ACC_PID_HPP_

#include <adaptive_cruise_controller/utilities.hpp>
#include <rclcpp/rclcpp.hpp>

namespace motion_planning
{

class AccPidNode
{
public:
  explicit AccPidNode(const double baselink2front, const AccParam & acc_param);

  void updateState(
    const rclcpp::Time & current_time, const double ego_velocity, const double obstacle_velocity,
    const double dist_to_obstacle);
  State getState() { return current_state_; }
  void calculate(const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion);

private:
  double calcStoppingDistFromCurrentVel(const double current_velocity);
  double calcEmergencyDistFromVel(const double current_velocity, const double obj_velocity);
  void calculateTargetMotion(const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion);
  void calcTrajectoryWithStopPoints(
    const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion);
  TrajectoryPoints insertStopPoint(
    const double stop_distance_from_ego, const TrajectoryPoints & trajectory_points,
    const geometry_msgs::msg::Pose & ego_pose, geometry_msgs::msg::Pose & stop_pose);

  double baselink2front_;
  AccParam acc_param_;
  State current_state_ = State::NONE;
  std::shared_ptr<AdaptiveCruiseInformation> prev_acc_info_ptr_;
};

}  // namespace motion_planning

#endif  // ADAPTIVE_CRUISE_CONTROLLER__ACC_PID_HPP_

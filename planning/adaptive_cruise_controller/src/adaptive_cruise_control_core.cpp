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

#include <adaptive_cruise_controller/adaptive_cruise_control_core.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace motion_planning
{

AdaptiveCruiseControlCore::AdaptiveCruiseControlCore(
  const double baselink2front, const AccParam & acc_param)
: baselink2front_(baselink2front), acc_param_(acc_param)
{
  acc_pid_node_ptr_ = std::make_shared<AccPidNode>(baselink2front, acc_param);
}

void AdaptiveCruiseControlCore::calcInformationForAdaptiveCruise(
  const TrajectoryPoints & trajectory_points, const geometry_msgs::msg::PoseStamped & pose,
  const Odometry & odometry, const PredictedObject & object, const rclcpp::Time & object_time)
{
  const auto object_pose = object.kinematics.initial_pose_with_covariance.pose;
  const double object_velocity =
    std::fabs(object.kinematics.initial_twist_with_covariance.twist.linear.x);
  double object_diff_angle = 0.0;
  getDiffAngleWithTrajectory(object_pose, trajectory_points, object_diff_angle);

  /* calculate the object velocity along trajectory */
  const double object_velocity_along_traj = object_velocity * std::cos(object_diff_angle);

  /* calculate distance to object */
  // calculate raw distance to the center-position of object from the base_link of ego-car
  double dist_to_object = tier4_autoware_utils::calcSignedArcLength(
    trajectory_points, pose.pose.position, object_pose.position);

  // considering the size of ego-car and object
  dist_to_object -= (baselink2front_ + object.shape.dimensions.x / 2.0);

  // delay compensation ( add object's travling distance in the delay time )
  if (object_velocity_along_traj > acc_param_.object_stop_velocity_thresh) {
    const double delay_time = (rclcpp::Time(pose.header.stamp) - object_time).seconds();
    const double running_distance_in_delay = delay_time * object_velocity_along_traj;
    dist_to_object += running_distance_in_delay;
  }

  /* input acc information */
  if (acc_info_ptr_) {
    prev_acc_info_ptr_ = std::make_shared<AdaptiveCruiseInformation>(*acc_info_ptr_);
  }

  acc_info_ptr_ = std::make_shared<AdaptiveCruiseInformation>();
  acc_info_ptr_->current_distance_to_object = dist_to_object;
  acc_info_ptr_->current_ego_velocity = odometry.twist.twist.linear.x;
  acc_info_ptr_->current_object_velocity = object_velocity_along_traj;
  acc_info_ptr_->ideal_distance_to_object =
    getIdealDistanceToObject(odometry.twist.twist.linear.x, object_velocity_along_traj);
  acc_info_ptr_->info_time = rclcpp::Time(pose.header.stamp);
  acc_info_ptr_->current_ego_pose = pose.pose;
  acc_info_ptr_->target_object = object;
  acc_info_ptr_->original_trajectory = trajectory_points;
}

double AdaptiveCruiseControlCore::getIdealDistanceToObject(
  const double current_velocity, const double object_velocity)
{
  // ego current velocity ... v_ego
  // ego minimum deceleration ... a_ego
  // object velocity ... v_obj
  // object minimum deceleration ... a_obj
  // idling_time ... t_idling
  // ideal_distance ... D
  // minimum_margin_distance ... d_margin
  // D = d_margin + v_e * t_idling + v_ego^2/(2*|a_ego|) - v_obj^2/(2*|a_obj|)

  const double margin_distance = acc_param_.minimum_margin_distance;
  const double idle_traving_distance = current_velocity * acc_param_.idling_time;
  const double ego_braking_distance =
    (current_velocity * current_velocity) / (2.0 * std::fabs(acc_param_.acc_min_acceleration));
  const double object_braking_distance =
    (object_velocity * object_velocity) / (2.0 * std::fabs(acc_param_.object_min_acceleration));

  return margin_distance + idle_traving_distance + ego_braking_distance - object_braking_distance;
}

void AdaptiveCruiseControlCore::calculate()
{
  if (!acc_info_ptr_) {
    return;
  }

  acc_pid_node_ptr_->calculate(*acc_info_ptr_, acc_motion_);
}

State AdaptiveCruiseControlCore::getState()
{
  if (!acc_pid_node_ptr_) {
    return State::NONE;
  }
  return acc_pid_node_ptr_->getState();
}

bool AdaptiveCruiseControlCore::getTargetMotion(
  double & target_velocity, double & target_acc, double & target_jerk)
{
  if (!acc_pid_node_ptr_ || !acc_info_ptr_) {
    return false;
  }

  if (!acc_motion_.use_target_motion) {
    // no target motion
    return false;
  }

  target_velocity = acc_motion_.target_velocity;
  target_acc = acc_motion_.target_acceleration;
  target_jerk = acc_motion_.target_jerk;

  return true;
}

TrajectoryPoints AdaptiveCruiseControlCore::getAccTrajectory(bool & emergency_flag)
{
  if (!acc_pid_node_ptr_ || !acc_info_ptr_) {
    emergency_flag = false;
    return acc_info_ptr_->original_trajectory;
  }

  if (!acc_motion_.use_trajectory) {
    // no trajectory update
    emergency_flag = false;
    return acc_info_ptr_->original_trajectory;
  }

  emergency_flag = acc_motion_.emergency;
  return acc_motion_.planned_trajectory;
}

}  // namespace motion_planning
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

#include <adaptive_cruise_controller/acc_pid.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace motion_planning
{

AccPidNode::AccPidNode(const double baselink2front, const AccParam & acc_param)
: baselink2front_(baselink2front), acc_param_(acc_param)
{
}

void AccPidNode::updateState(
  const rclcpp::Time & current_time, const double ego_velocity, const double obstacle_velocity,
  const double dist_to_obstacle)
{
  if (!prev_acc_info_ptr_) {
    current_state_ = State::NONE;
    return;
  }

  if (
    (current_time - prev_acc_info_ptr_->info_time).seconds() > acc_param_.reset_time_to_acc_state) {
    // reset current_state if previous acc_time is too old
    current_state_ = State::NONE;
    return;
  }

  double thresh_velocity = acc_param_.object_low_velocity_thresh;

  // provide a hysteresis in the state to prevent chattering
  if (current_state_ == State::STOP) {
    thresh_velocity += acc_param_.object_velocity_hysteresis_margin;
  }

  // change state depending on obstacle velocity
  if (obstacle_velocity >= thresh_velocity) {
    current_state_ = State::ACC;
  } else {
    current_state_ = State::STOP;
  }

  // if current distance to object is too short, change state to STOP
  if (calcEmergencyDistFromVel(obstacle_velocity, ego_velocity) > dist_to_obstacle) {
    current_state_ = State::STOP;
  }
}

void AccPidNode::calculate(const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion)
{
  updateState(
    acc_info.info_time, acc_info.current_ego_velocity, acc_info.current_object_velocity,
    acc_info.current_distance_to_object);

  if (current_state_ == State::STOP) {
    acc_motion.use_target_motion = false;
    calcTrajectoryWithStopPoints(acc_info, acc_motion);
  } else if (current_state_ == State::ACC) {
    calculateTargetMotion(acc_info, acc_motion);
    acc_motion.use_trajectory = false;
  }

  prev_acc_info_ptr_ = std::make_shared<AdaptiveCruiseInformation>(acc_info);
}

double AccPidNode::calcStoppingDistFromCurrentVel(const double current_velocity)
{
  const double idling_travel_distance = current_velocity * acc_param_.breaking_delay_time;
  const double braking_distance =
    (current_velocity * current_velocity) / (2.0 * -acc_param_.stop_min_acceleration);
  return idling_travel_distance + braking_distance;
}

double AccPidNode::calcEmergencyDistFromVel(
  const double current_velocity, const double obj_velocity)
{
  // when the target object is faster than ego-vehicle
  if (current_velocity < obj_velocity) {
    return acc_param_.minimum_margin_distance;
  }

  const double idling_travel_distance = current_velocity * acc_param_.breaking_delay_time;
  const double braking_distance =
    (current_velocity * current_velocity) / (2.0 * acc_param_.stop_min_acceleration);
  const double object_braking_distance =
    (obj_velocity * obj_velocity) / (2.0 * acc_param_.stop_min_acceleration);
  const double emergency_dist = idling_travel_distance + braking_distance - object_braking_distance;
  return std::max(acc_param_.minimum_margin_distance, emergency_dist);
}

void AccPidNode::calculateTargetMotion(
  const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion)
{
  const double diff_distance_to_object =
    acc_info.current_distance_to_object - acc_info.ideal_distance_to_object;
  const double min_target_velocity = acc_param_.object_low_velocity_thresh;
  const double additional_velocity = acc_param_.p_term_in_velocity_pid * diff_distance_to_object;

  double target_velocity = acc_info.current_ego_velocity + additional_velocity;
  target_velocity = std::max(min_target_velocity, target_velocity);
  acc_motion.target_velocity = target_velocity;

  const double target_accleration =
    -acc_param_.velocity_to_accleration_weight * additional_velocity;
  acc_motion.target_acceleration = std::max(target_accleration, acc_param_.acc_min_acceleration);
  acc_motion.target_jerk = acc_param_.acc_min_jerk;
  acc_motion.use_target_motion = true;
}

void AccPidNode::calcTrajectoryWithStopPoints(
  const AdaptiveCruiseInformation & acc_info, AccMotion & acc_motion)
{
  const double original_stop_dist =
    acc_info.current_distance_to_object - acc_param_.minimum_margin_distance;
  const double min_stop_dist = calcStoppingDistFromCurrentVel(acc_info.current_ego_velocity);

  double target_stop_dist;
  if (original_stop_dist >= min_stop_dist) {
    target_stop_dist = original_stop_dist;
    acc_motion.emergency = false;
  } else {
    target_stop_dist = min_stop_dist;
    acc_motion.emergency = true;
  }

  acc_motion.planned_trajectory = insertStopPoint(
    target_stop_dist, acc_info.original_trajectory, acc_info.current_ego_pose,
    acc_motion.stop_pose);
  acc_motion.use_trajectory = true;
}

TrajectoryPoints AccPidNode::insertStopPoint(
  const double stop_distance_from_ego, const TrajectoryPoints & trajectory_points,
  const geometry_msgs::msg::Pose & ego_pose, geometry_msgs::msg::Pose & stop_pose)
{
  TrajectoryPoints trajectory_with_stop_point = trajectory_points;

  if (trajectory_points.empty()) {
    return trajectory_points;
  }

  // calculate arc length from trajectory.front() to ego pose
  const auto arc_length_to_ego = tier4_autoware_utils::calcSignedArcLength(
    trajectory_points, trajectory_points.front().pose.position, ego_pose.position);

  // calulcate stop index and stop point
  const double stop_distance = arc_length_to_ego + stop_distance_from_ego;
  double accumulated_length = 0;
  double insert_idx = 0;
  bool find_idx = false;
  for (size_t i = 1; i < trajectory_points.size(); i++) {
    const auto prev_pose = trajectory_points.at(i - 1).pose;
    const auto curr_pose = trajectory_points.at(i).pose;
    const double segment_length = tier4_autoware_utils::calcDistance3d(prev_pose, curr_pose);
    accumulated_length += segment_length;
    if (accumulated_length > stop_distance) {
      insert_idx = i;
      const double ratio = 1 - (accumulated_length - stop_distance) / segment_length;
      stop_pose = lerpByPose(prev_pose, curr_pose, ratio);
      find_idx = true;
      break;
    }
  }

  if (!find_idx) {
    // insert stop velocity to the last point of trajectory
    stop_pose = trajectory_points.back().pose;
    trajectory_with_stop_point.back().longitudinal_velocity_mps = 0.0;
    trajectory_with_stop_point.back().lateral_velocity_mps = 0.0;
    return trajectory_with_stop_point;
  }

  // insert stop point
  TrajectoryPoint stop_point;
  stop_point.longitudinal_velocity_mps = 0.0;
  stop_point.lateral_velocity_mps = 0.0;
  stop_point.pose = stop_pose;
  trajectory_with_stop_point.insert(trajectory_with_stop_point.begin() + insert_idx, stop_point);
  // set 0 velocity to points after the stop point
  for (size_t i = insert_idx; i < trajectory_with_stop_point.size(); i++) {
    trajectory_with_stop_point.at(i).longitudinal_velocity_mps = 0.0;
    trajectory_with_stop_point.at(i).lateral_velocity_mps = 0.0;
  }
  return trajectory_with_stop_point;
}

}  // namespace motion_planning
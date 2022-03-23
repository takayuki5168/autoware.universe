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

#include "obstacle_velocity_planner/rule_based_planner/rule_based_planner.hpp"

#include "obstacle_velocity_planner/utils.hpp"
#include "tier4_autoware_utils/tier4_autoware_utils.hpp"

#include "tier4_planning_msgs/msg/velocity_limit.hpp"

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

autoware_auto_planning_msgs::msg::Trajectory RuleBasedPlanner::generateTrajectory(
  const ObstacleVelocityPlannerData & planner_data)
{
  auto output_traj = planner_data.traj;
  vel_limit_ = {};

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
        calcRSSDistance(planner_data.current_vel, obstacle.velocity, safe_distance_margin);
      std::cerr << planner_data.current_vel << " " << obstacle.velocity << " " << rss_dist
                << std::endl;
      const double rss_dist_with_vehicle_offset =
        rss_dist + vehicle_info_.max_longitudinal_offset_m + obstacle.shape.dimensions.x / 2.0;

      // calculate current obstacle pose
      const auto current_obstacle_pose =
        obstacle_velocity_utils::getCurrentObjectPoseFromPredictedPath(
          obstacle.predicted_paths.at(0), obstacle.time_stamp, planner_data.current_time);
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
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "stop planning");

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
        marker_pose.get(), "obstacle to stop", planner_data.current_time, 0);
      tier4_autoware_utils::appendMarkerArray(markers, &wall_msg);

      // publish wall marker
      debug_wall_marker_pub_->publish(wall_msg);
    }
  }

  if (min_dist_to_slow_down) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "slow down planning");

    // adaptive cruise TODO
    // calculate target velocity with acceleration limit by PID controller
    const double diff_vel = pid_controller_.calc(min_dist_to_slow_down.get());
    const double prev_vel = prev_target_vel_ ? prev_target_vel_.get() : planner_data.current_vel;
    const double target_vel_with_acc_limit =
      prev_vel + std::max(
                   std::min(diff_vel, longitudinal_info_.max_accel * 0.1),
                   longitudinal_info_.min_accel * 0.1);  // TODO(murooka) accel * 0.1 (time step)
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "target_velocity %f",
      target_vel_with_acc_limit);

    prev_target_vel_ = target_vel_with_acc_limit;

    // set velocity limit
    if (!vel_limit_ || target_vel_with_acc_limit < vel_limit_.get()) {
      vel_limit_ = target_vel_with_acc_limit;
    }

    // virtual wall marker for slow down
    const double dist_to_rss_wall =
      min_dist_to_slow_down.get() + vehicle_info_.max_longitudinal_offset_m;
    const size_t wall_idx =
      getIndexWithLongitudinalOffset(output_traj.points, dist_to_rss_wall, ego_idx);

    visualization_msgs::msg::MarkerArray rss_wall_msg;

    const auto markers = tier4_autoware_utils::createSlowDownVirtualWallMarker(
      output_traj.points.at(wall_idx).pose, "rss distance", planner_data.current_time, 0);
    tier4_autoware_utils::appendMarkerArray(markers, &rss_wall_msg);

    debug_rss_wall_marker_pub_->publish(rss_wall_msg);
  } else {
    // reset previous target velocity if adaptive cruise is not enabled
    prev_target_vel_ = {};
  }

  return output_traj;
}

boost::optional<double> RuleBasedPlanner::calcVelocityLimit(
  const ObstacleVelocityPlannerData & planner_data)
{
  return vel_limit_;
}

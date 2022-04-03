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
VelocityLimit createVelocityLimitMsg(
  const rclcpp::Time & current_time, const double vel, const double acc, const double max_jerk,
  const double min_jerk)
{
  VelocityLimit msg;
  msg.stamp = current_time;
  msg.sender = "obstacle_velocity_planner";
  msg.use_constraints = true;

  msg.max_velocity = vel;
  if (acc < 0) {
    msg.constraints.min_acceleration = acc;
  }
  msg.constraints.max_jerk = max_jerk;
  msg.constraints.min_jerk = min_jerk;

  return msg;
}

Float32MultiArrayStamped convertDebugValuesToMsg(
  const rclcpp::Time & current_time, const DebugValues & debug_values)
{
  Float32MultiArrayStamped debug_msg{};
  debug_msg.stamp = current_time;
  for (const auto & v : debug_values.getValues()) {
    debug_msg.data.push_back(v);
  }
  return debug_msg;
}

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

double calcMinimumDistanceToStop(const double initial_vel, const double min_acc)
{
  return -std::pow(initial_vel, 2) / 2.0 / min_acc;
}

tier4_planning_msgs::msg::StopReasonArray makeStopReasonArray(
  const rclcpp::Time & current_time, const geometry_msgs::msg::Pose & stop_pose)
{
  // create header
  std_msgs::msg::Header header;
  header.frame_id = "map";
  header.stamp = current_time;

  // create stop factor
  tier4_planning_msgs::msg::StopFactor stop_factor;
  stop_factor.stop_pose = stop_pose;
  // TODO(murooka)
  // stop_factor.stop_factor_points.emplace_back();

  // create stop reason stamped
  tier4_planning_msgs::msg::StopReason stop_reason_msg;
  stop_reason_msg.reason = tier4_planning_msgs::msg::StopReason::OBSTACLE_STOP;
  stop_reason_msg.stop_factors.emplace_back(stop_factor);

  // create stop reason array
  tier4_planning_msgs::msg::StopReasonArray stop_reason_array;
  stop_reason_array.header = header;
  stop_reason_array.stop_reasons.emplace_back(stop_reason_msg);
  return stop_reason_array;
}
}  // namespace

RuleBasedPlanner::RuleBasedPlanner(
  rclcpp::Node & node, const LongitudinalInfo & longitudinal_info,
  const vehicle_info_util::VehicleInfo & vehicle_info)
: PlannerInterface(longitudinal_info, vehicle_info)
{
  // pid controller
  const double kp = node.declare_parameter<double>("rule_based_planner.kp");
  const double ki = node.declare_parameter<double>("rule_based_planner.ki");
  const double kd = node.declare_parameter<double>("rule_based_planner.kd");
  pid_controller_ = std::make_unique<PIDController>(kp, ki, kd);
  output_ratio_during_accel_ =
    node.declare_parameter<double>("rule_based_planner.output_ratio_during_accel");

  // vel_to_acc_weight
  vel_to_acc_weight_ = node.declare_parameter<double>("rule_based_planner.vel_to_acc_weight");

  // min_slow_down_target_vel
  min_slow_down_target_vel_ =
    node.declare_parameter<double>("rule_based_planner.min_slow_down_target_vel");

  max_vehicle_obj_velocity_to_stop_ =
    node.declare_parameter<double>("rule_based_planner.max_vehicle_obj_velocity_to_stop");
  max_non_vehicle_obj_velocity_to_stop_ =
    node.declare_parameter<double>("rule_based_planner.max_non_vehicle_obj_velocity_to_stop");
  safe_distance_margin_ = node.declare_parameter<double>("rule_based_planner.safe_distance_margin");
  min_obstacle_stop_accel_ =
    node.declare_parameter<double>("rule_based_planner.min_obstacle_stop_accel");

  // Publisher
  stop_reasons_pub_ =
    node.create_publisher<tier4_planning_msgs::msg::StopReasonArray>("~/output/stop_reasons", 1);
  stop_speed_exceeded_pub_ =
    node.create_publisher<StopSpeedExceeded>("~/output/stop_speed_exceeded", 1);
  debug_wall_marker_pub_ =
    node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/wall_marker", 1);
  debug_rss_wall_marker_pub_ =
    node.create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/rss_wall_marker", 1);
  debug_values_pub_ = node.create_publisher<Float32MultiArrayStamped>("~/debug/values", 1);
}

boost::optional<size_t> RuleBasedPlanner::getZeroVelocityIndexWithVelocityLimit(
  const ObstacleVelocityPlannerData & planner_data, boost::optional<VelocityLimit> & vel_limit)
{
  debug_values_.resetValues();
  debug_values_.setValues(DebugValues::TYPE::CURRENT_VELOCITY, planner_data.current_vel);
  debug_values_.setValues(DebugValues::TYPE::CURRENT_ACCELERATION, planner_data.current_acc);

  // search highest probability obstacle for stop and slow down
  boost::optional<double> min_dist_to_stop;
  boost::optional<std::pair<size_t, double>> min_dist_to_slow_down;
  for (size_t o_idx = 0; o_idx < planner_data.target_obstacles.size(); ++o_idx) {
    const auto & obstacle = planner_data.target_obstacles.at(o_idx);

    /*
    // interpolate current obstacle pose
    const auto current_interpolated_obstacle_pose =
    obstacle_velocity_utils::getCurrentObjectPoseFromPredictedPath( obstacle.predicted_paths.at(0),
    obstacle.time_stamp, planner_data.current_time); if (!current_interpolated_obstacle_pose) {
      continue;
    }
    const double dist_to_obstacle = tier4_autoware_utils::calcSignedArcLength(
      planner_data.traj.points, planner_data.current_pose.position,
    current_interpolated_obstacle_pose->position);
    */
    const double dist_to_obstacle = tier4_autoware_utils::calcSignedArcLength(
      planner_data.traj.points, planner_data.current_pose.position, obstacle.pose.position);

    const bool is_vehicle = obstacle_velocity_utils::isVehicle(obstacle.classification.label);
    const bool is_stop_required = [&]() {
      if (is_vehicle) {
        return std::abs(obstacle.velocity) < max_vehicle_obj_velocity_to_stop_;
      }
      return std::abs(obstacle.velocity) < max_non_vehicle_obj_velocity_to_stop_;
    }();
    if (is_stop_required) {  // stop
      // calculate distance to stop
      // TODO vehicle offset
      const double dist_to_stop_with_acc_limit = [&]() {
        constexpr double epsilon = 1e-6;
        if (planner_data.current_vel < epsilon) {
          return 0.0;
        }

        const double time_to_stop_with_acc_limit =
          -planner_data.current_vel / min_obstacle_stop_accel_;
        return planner_data.current_vel * time_to_stop_with_acc_limit + min_obstacle_stop_accel_ +
               std::pow(time_to_stop_with_acc_limit, 2);
      }();

      const double dist_to_stop =
        std::max(dist_to_obstacle, dist_to_stop_with_acc_limit) - safe_distance_margin_;
      [&]() {
        if (min_dist_to_stop) {
          if (dist_to_stop > min_dist_to_stop.get()) {
            return;
          }
        }
        min_dist_to_stop = dist_to_stop;

        // calculate error distance
        const double error_dist = dist_to_obstacle - dist_to_stop;

        // update debug values
        debug_values_.setValues(DebugValues::TYPE::STOP_CURRENT_OBJECT_DISTANCE, dist_to_obstacle);
        debug_values_.setValues(DebugValues::TYPE::STOP_CURRENT_OBJECT_VELOCITY, obstacle.velocity);
        debug_values_.setValues(DebugValues::TYPE::STOP_TARGET_OBJECT_DISTANCE, dist_to_stop);
        debug_values_.setValues(
          DebugValues::TYPE::STOP_TARGET_ACCELERATION, min_obstacle_stop_accel_);
        debug_values_.setValues(DebugValues::TYPE::STOP_ERROR_OBJECT_DISTANCE, error_dist);
      }();
    } else {  // adaptive cruise
      // calculate distance between ego and obstacle based on RSS
      const double rss_dist =
        calcRSSDistance(planner_data.current_vel, obstacle.velocity, safe_distance_margin_);
      const double rss_dist_with_vehicle_offset =
        rss_dist + vehicle_info_.max_longitudinal_offset_m + obstacle.shape.dimensions.x / 2.0;

      // calculate error distance
      const double error_dist = dist_to_obstacle - rss_dist_with_vehicle_offset;

      [&]() {
        if (min_dist_to_slow_down) {
          if (error_dist > min_dist_to_slow_down->second) {
            return;
          }
        }
        min_dist_to_slow_down = std::make_pair(o_idx, error_dist);

        // update debug values
        debug_values_.setValues(
          DebugValues::TYPE::SLOW_DOWN_CURRENT_OBJECT_VELOCITY, obstacle.velocity);
        debug_values_.setValues(
          DebugValues::TYPE::SLOW_DOWN_CURRENT_OBJECT_DISTANCE, dist_to_obstacle);
        debug_values_.setValues(
          DebugValues::TYPE::SLOW_DOWN_TARGET_OBJECT_DISTANCE, rss_dist_with_vehicle_offset);
        debug_values_.setValues(DebugValues::TYPE::SLOW_DOWN_ERROR_OBJECT_DISTANCE, error_dist);
      }();
    }
  }

  bool will_collide_with_obstacle = false;

  // do stop
  boost::optional<size_t> zero_vel_idx = {};
  if (min_dist_to_stop) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "stop planning");

    const double dist_to_stop = min_dist_to_stop.get();

    // set zero velocity index
    zero_vel_idx = doStop(planner_data, dist_to_stop);

    // check if the ego will collide with the obstacle
    const double feasible_dist_to_stop =
      calcMinimumDistanceToStop(planner_data.current_vel, longitudinal_info_.min_accel);
    if (dist_to_stop < feasible_dist_to_stop) {
      will_collide_with_obstacle = true;
    }
  }

  // do slow down
  if (min_dist_to_slow_down) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "slow down planning");

    const size_t target_obstacle_idx = min_dist_to_slow_down->first;
    const double dist_to_slow_down = min_dist_to_slow_down->second;

    vel_limit = doSlowDown(planner_data, target_obstacle_idx, dist_to_slow_down);

    // update debug values
    debug_values_.setValues(DebugValues::TYPE::SLOW_DOWN_TARGET_VELOCITY, vel_limit->max_velocity);
    debug_values_.setValues(
      DebugValues::TYPE::SLOW_DOWN_TARGET_ACCELERATION, longitudinal_info_.min_accel);
  } else {
    // reset previous target velocity if adaptive cruise is not enabled
    prev_target_vel_ = {};
  }

  // pulish stop_speed_exceeded if the ego will collide with the obstacle
  StopSpeedExceeded msg{};
  msg.stamp = planner_data.current_time;
  msg.stop_speed_exceeded = will_collide_with_obstacle;
  stop_speed_exceeded_pub_->publish(msg);

  // publish debug values
  const auto debug_values_msg = convertDebugValuesToMsg(planner_data.current_time, debug_values_);
  debug_values_pub_->publish(debug_values_msg);

  // publish stop reason
  if (zero_vel_idx) {
    const auto stop_pose = planner_data.traj.points.at(zero_vel_idx.get()).pose;
    const auto stop_reasons_msg = makeStopReasonArray(planner_data.current_time, stop_pose);
    stop_reasons_pub_->publish(stop_reasons_msg);
  }

  return zero_vel_idx;
}

size_t RuleBasedPlanner::doStop(
  const ObstacleVelocityPlannerData & planner_data, const double dist_to_stop) const
{
  const size_t ego_idx = tier4_autoware_utils::findNearestIndex(
    planner_data.traj.points, planner_data.current_pose.position);

  // TODO(murooka) Should I use interpolation?
  const size_t zero_vel_idx = getIndexWithLongitudinalOffset(
    planner_data.traj.points, dist_to_stop - vehicle_info_.max_longitudinal_offset_m, ego_idx);

  // virtual wall marker for stop
  const auto marker_pose = obstacle_velocity_utils::calcForwardPose(
    planner_data.traj, planner_data.current_pose.position, dist_to_stop);
  if (marker_pose) {
    visualization_msgs::msg::MarkerArray wall_msg;
    const auto markers = tier4_autoware_utils::createStopVirtualWallMarker(
      marker_pose.get(), "obstacle to stop", planner_data.current_time, 0);
    tier4_autoware_utils::appendMarkerArray(markers, &wall_msg);

    // publish wall marker
    debug_wall_marker_pub_->publish(wall_msg);
  }

  return zero_vel_idx;
}

VelocityLimit RuleBasedPlanner::doSlowDown(
  const ObstacleVelocityPlannerData & planner_data, const size_t target_obstacle_idx,
  const double dist_to_slow_down)
{
  const auto & obstacle = planner_data.target_obstacles.at(target_obstacle_idx);

  const size_t ego_idx = tier4_autoware_utils::findNearestIndex(
    planner_data.traj.points, planner_data.current_pose.position);

  // calculate target velocity with acceleration limit by PID controller
  const double pid_output_vel = pid_controller_->calc(dist_to_slow_down);
  [[maybe_unused]] const double prev_vel =
    prev_target_vel_ ? prev_target_vel_.get() : planner_data.current_vel;

  const double additional_vel = [&]() {
    if (dist_to_slow_down > 0) {
      return pid_output_vel * output_ratio_during_accel_;
    }
    return pid_output_vel;
  }();

  // std::clamp(pid_output_vel, longitudinal_info_.min_accel * 0.1, longitudinal_info_.max_accel *
  // 0.1);
  // // TODO(murooka) accel * 0.1 (time step)
  const double target_vel_with_acc_limit =
    // std::max(min_slow_down_target_vel_, prev_vel + additional_vel); // TODO
    // std::max(min_slow_down_target_vel_, planner_data.current_vel + additional_vel);  // TODO
    std::max(0.0, planner_data.current_vel + additional_vel);  // TODO
  // std::max(min_slow_down_target_vel_, obstacle.velocity + additional_vel);  // TODO

  // calculate target acceleration
  const double target_acc = vel_to_acc_weight_ * additional_vel;
  const double target_acc_with_acc_limit =
    std::clamp(target_acc, longitudinal_info_.min_accel, longitudinal_info_.max_accel);

  RCLCPP_INFO_EXPRESSION(
    rclcpp::get_logger("ObstacleVelocityPlanner::RuleBasedPlanner"), true, "target_velocity %f",
    target_vel_with_acc_limit);

  prev_target_vel_ = target_vel_with_acc_limit;

  // set target longitudinal motion
  const auto vel_limit = createVelocityLimitMsg(
    planner_data.current_time, target_vel_with_acc_limit, target_acc_with_acc_limit,
    longitudinal_info_.max_jerk, longitudinal_info_.min_jerk);

  // virtual wall marker for slow down
  const double dist_to_rss_wall = dist_to_slow_down + vehicle_info_.max_longitudinal_offset_m;
  const size_t wall_idx =
    getIndexWithLongitudinalOffset(planner_data.traj.points, dist_to_rss_wall, ego_idx);

  visualization_msgs::msg::MarkerArray rss_wall_msg;

  const auto markers = tier4_autoware_utils::createSlowDownVirtualWallMarker(
    planner_data.traj.points.at(wall_idx).pose, "rss distance", planner_data.current_time, 0);
  tier4_autoware_utils::appendMarkerArray(markers, &rss_wall_msg);

  debug_rss_wall_marker_pub_->publish(rss_wall_msg);

  return vel_limit;
}

void RuleBasedPlanner::updateParam(const std::vector<rclcpp::Parameter> & parameters)
{
  // pid controller
  double kp = pid_controller_->getKp();
  double ki = pid_controller_->getKi();
  double kd = pid_controller_->getKd();

  tier4_autoware_utils::updateParam<double>(parameters, "rule_based_planner.kp", kp);
  tier4_autoware_utils::updateParam<double>(parameters, "rule_based_planner.ki", ki);
  tier4_autoware_utils::updateParam<double>(parameters, "rule_based_planner.kd", kd);
  tier4_autoware_utils::updateParam<double>(
    parameters, "rule_based_planner.output_ratio_during_accel", output_ratio_during_accel_);

  // vel_to_acc_weight
  tier4_autoware_utils::updateParam<double>(
    parameters, "rule_based_planner.vel_to_acc_weight", vel_to_acc_weight_);

  // min_slow_down_target_vel
  tier4_autoware_utils::updateParam<double>(
    parameters, "rule_based_planner.min_slow_down_target_vel", min_slow_down_target_vel_);

  pid_controller_->updateParam(kp, ki, kd);
}

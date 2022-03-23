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

#include "obstacle_velocity_planner/node.hpp"

#include "obstacle_velocity_planner/polygon_utils.hpp"
#include "obstacle_velocity_planner/utils.hpp"

#include <tier4_autoware_utils/ros/update_param.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>
#include <tier4_autoware_utils/trajectory/tmp_conversion.hpp>

#include <algorithm>
#include <chrono>

/*
geometry_msgs::msg::Point operator+(
  const geometry_msgs::msg::Point & front_pos, const geometry_msgs::msg::Point & back_pos)
{
  geometry_msgs::msg::Point added_pos;
  added_pos.x = front_pos.x + back_pos.x;
  added_pos.y = front_pos.y + back_pos.y;
  added_pos.z = front_pos.z + back_pos.z;
  return added_pos;
}

geometry_msgs::msg::Point operator*(
  const geometry_msgs::msg::Point & pos, const double val)
{
  geometry_msgs::msg::Point added_pos;
  added_pos.x = pos.x * val;
  added_pos.y = pos.y * val;
  added_pos.z = pos.z * val;
  return added_pos;
}

geometry_msgs::msg::Point lerpPoint(const geometry_msgs::msg::Point & front_pos, const
geometry_msgs::msg::Point & back_pos, const double ratio)
{
  return front_pos * ratio + back_pos * (1 - ratio);
}

template <class T>
geometry_msgs::msg::Point calcInterpolatedPointWithLongitudinalOffset(const std::vector<T> & points,
const double longitudinal_offset, const boost::optional<size_t> start_idx)
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
    for (size_t i = start_idx; i < points.size() - 1; ++i) {
      const double segment_length = tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i +
1)); sum_length += segment_length; if (sum_length >= longitudinal_offset) { const double
front_length = segment_length - (sum_length - longitudinal_offset); return
lerpPoint(tier4_autoware_utils::getPoint(points.at(i)),tier4_autoware_utils::getPoint(points.at(i +
1)), front_length / segment_length);
      }
    }
    return tier4_autoware_utils::getPoint(points.back());
  }

  for (size_t i = start_idx; i > 0; --i) {
    const double segment_length = tier4_autoware_utils::calcDistance2d(points.at(i), points.at(i +
1)); sum_length += segment_length; if (sum_length >= -longitudinal_offset) { const double
front_length = segment_length - (sum_length + longitudinal_offset); return
lerpPoint(tier4_autoware_utils::getPoint(points.at(i)),tier4_autoware_utils::getPoint(points.at(i +
1)), front_length / segment_length);
    }
  }
  return tier4_autoware_utils::getPoint(points.front());
}
*/

ObstacleVelocityPlanner::ObstacleVelocityPlanner(const rclcpp::NodeOptions & node_options)
: Node("obstacle_velocity_planner", node_options),
  self_pose_listener_(this),
  planning_method_(PlanningMethod::RULE_BASE)
// planning_method_(PlanningMethod::OPTIMIZATION_BASE)
{
  using std::placeholders::_1;

  // Subscriber
  trajectory_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "~/input/trajectory", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::trajectoryCallback, this, _1));
  smoothed_trajectory_sub_ = create_subscription<autoware_auto_planning_msgs::msg::Trajectory>(
    "/planning/scenario_planning/trajectory", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::smoothedTrajectoryCallback, this, _1));
  objects_sub_ = create_subscription<autoware_auto_perception_msgs::msg::PredictedObjects>(
    "/perception/object_recognition/objects", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::objectsCallback, this, _1));
  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    "/localization/kinematic_state", rclcpp::QoS{1},
    std::bind(&ObstacleVelocityPlanner::odomCallback, this, std::placeholders::_1));
  sub_map_ = this->create_subscription<autoware_auto_mapping_msgs::msg::HADMapBin>(
    "/map/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ObstacleVelocityPlanner::mapCallback, this, std::placeholders::_1));
  /*
  sub_external_velocity_limit_ = create_subscription<tier4_planning_msgs::msg::VelocityLimit>(
    "/planning/scenario_planning/max_velocity", 1,
    std::bind(&ObstacleVelocityPlanner::onExternalVelocityLimit, this, _1));
  */

  // Publisher
  trajectory_pub_ =
    create_publisher<autoware_auto_planning_msgs::msg::Trajectory>("~/output/trajectory", 1);
  external_vel_limit_pub_ = create_publisher<tier4_planning_msgs::msg::VelocityLimit>(
    "/planning/scenario_planning/max_velocity", 1);
  debug_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/marker", 1);

  // Obstacle
  in_objects_ptr_ = std::make_unique<autoware_auto_perception_msgs::msg::PredictedObjects>();

  // Vehicle Parameters
  vehicle_info_ = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();

  // Parameters
  const double max_accel = declare_parameter("max_accel", 1.0);
  const double min_accel = declare_parameter("min_accel", -1.0);
  const double max_jerk = declare_parameter("max_jerk", 1.0);
  const double min_jerk = declare_parameter("min_jerk", -1.0);
  const double min_object_accel = declare_parameter("min_object_accel", -3.0);
  const double t_idling = declare_parameter("t_idling", 2.0);

  // Wait for first self pose
  self_pose_listener_.waitForFirstPose();

  if (planning_method_ == PlanningMethod::OPTIMIZATION_BASE) {
    planner_ptr_ = std::make_unique<OptimizationBasedPlanner>(
      *this, max_accel, min_accel, max_jerk, min_jerk, min_object_accel, t_idling, vehicle_info_);
  } else if (planning_method_ == PlanningMethod::RULE_BASE) {
    planner_ptr_ = std::make_unique<RuleBasedPlanner>(
      *this, max_accel, min_accel, max_jerk, min_jerk, min_object_accel, t_idling, vehicle_info_);
  } else {
    std::logic_error("Designated method is not supported.");
  }
}

void ObstacleVelocityPlanner::mapCallback(
  const autoware_auto_mapping_msgs::msg::HADMapBin::ConstSharedPtr msg)
{
  auto lanelet_map_ptr = std::make_shared<lanelet::LaneletMap>();
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr;

  RCLCPP_INFO(get_logger(), "[Obstacle Velocity Planner]: Start loading lanelet");
  lanelet::utils::conversion::fromBinMsg(
    *msg, lanelet_map_ptr, &traffic_rules_ptr, &routing_graph_ptr);

  if (planner_ptr_ && planning_method_ == PlanningMethod::OPTIMIZATION_BASE) {
    planner_ptr_->setMaps(lanelet_map_ptr, traffic_rules_ptr, routing_graph_ptr);
    RCLCPP_INFO(get_logger(), "[Obstacle Velocity Planner]: Map is loaded");
  }
}

void ObstacleVelocityPlanner::objectsCallback(
  const autoware_auto_perception_msgs::msg::PredictedObjects::SharedPtr msg)
{
  in_objects_ptr_ = msg;
}

void ObstacleVelocityPlanner::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  current_twist_ptr_ = std::make_unique<geometry_msgs::msg::TwistStamped>();
  current_twist_ptr_->header = msg->header;
  current_twist_ptr_->twist = msg->twist.twist;
}

void ObstacleVelocityPlanner::smoothedTrajectoryCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  planner_ptr_->setSmoothedTrajectory(msg);
}

/*
void ObstacleVelocityPlanner::onExternalVelocityLimit(
  const tier4_planning_msgs::msg::VelocityLimit::ConstSharedPtr msg)
{
  external_velocity_limit_ = msg->max_velocity;
}
*/

void ObstacleVelocityPlanner::trajectoryCallback(
  const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr msg)
{
  tier4_autoware_utils::StopWatch stop_watch;
  stop_watch.tic();

  std::lock_guard<std::mutex> lock(mutex_);

  if (msg->points.empty() || !current_twist_ptr_ || !in_objects_ptr_) {
    return;
  }

  // convert obstacle type
  const auto obstacles = TargetObstacle::convertToTargetObstacles(*in_objects_ptr_);

  // Prepare algorithmic data
  ObstacleVelocityPlannerData planner_data;
  planner_data.current_time = get_clock()->now();
  planner_data.traj = *msg;
  planner_data.current_pose = self_pose_listener_.getCurrentPose()->pose;
  planner_data.current_vel = current_twist_ptr_->twist.linear.x;
  // planner_data.external_velocity_limit = external_velocity_limit_;
  planner_data.target_obstacles = filterObstacles(
    obstacles, planner_data.traj, planner_data.current_pose, planner_data.current_vel);

  // generate Trajectory
  const auto output = planner_ptr_->generateTrajectory(planner_data);

  // publish velocity limit if required
  const auto vel_limit = planner_ptr_->calcVelocityLimit(planner_data);
  if (vel_limit) {
    tier4_planning_msgs::msg::VelocityLimit vel_limit_msg;
    vel_limit_msg.max_velocity = vel_limit.get();
    std::cerr << vel_limit.get() << std::endl;
    external_vel_limit_pub_->publish(vel_limit_msg);
  }

  // Publish trajectory
  trajectory_pub_->publish(output);

  const double elapsed_time = stop_watch.toc();
  // RCLCPP_WARN_STREAM(get_logger(), elapsed_time);
}

std::vector<TargetObstacle> ObstacleVelocityPlanner::filterObstacles(
  const std::vector<TargetObstacle> & obstacles,
  const autoware_auto_planning_msgs::msg::Trajectory & traj,
  const geometry_msgs::msg::Pose & current_pose, const double current_vel)
{
  std::vector<TargetObstacle> target_obstacles;

  // TODO(murooka) parametrise these parameters
  constexpr double margin_between_traj_and_obstacle = 0.3;
  constexpr double min_obstacle_velocity = 3.0;  // 10.8 [km/h]
  constexpr double margin_for_collision_time = 3.0;
  constexpr double max_ego_obj_overlap_time = 1.0;
  constexpr double max_prediction_time_for_collision_check = 20.0;

  const auto traj_polygons = polygon_utils::createOneStepPolygons(traj, vehicle_info_);

  for (const auto & obstacle : obstacles) {
    const auto first_within_idx = polygon_utils::getFirstCollisionIndex(
      traj_polygons, polygon_utils::convertObstacleToPolygon(obstacle.pose, obstacle.shape),
      margin_between_traj_and_obstacle);

    if (first_within_idx) {  // obsacles inside the trajectory
      if (
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::CAR ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::BUS ||
        obstacle.classification.label ==
          autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE) {  // vehicle
                                                                                   // obstacle

        if (std::abs(obstacle.velocity) > min_obstacle_velocity) {  // running obstacle
          const double time_to_collision = [&]() {
            // TODO(murooka) consider obstacle width/length the same as
            // vehicle_info_.max_longitudinal_offset_m
            const double dist_to_ego =
              tier4_autoware_utils::calcSignedArcLength(
                traj.points, current_pose.position, first_within_idx.get()) -
              vehicle_info_.max_longitudinal_offset_m;
            return dist_to_ego / std::max(1e-6, current_vel);
          }();

          const double time_to_obstacle_getting_out = [&]() {
            const auto obstacle_getting_out_idx = polygon_utils::getFirstNonCollisionIndex(
              traj_polygons, obstacle.predicted_paths.at(0), obstacle.shape, first_within_idx.get(),
              margin_between_traj_and_obstacle);
            if (!obstacle_getting_out_idx) {
              return std::numeric_limits<double>::max();
            }

            const double dist_to_obstacle_getting_out = tier4_autoware_utils::calcSignedArcLength(
              traj.points, obstacle.pose.position, obstacle_getting_out_idx.get());
            return dist_to_obstacle_getting_out / obstacle.velocity;
          }();

          /*
          RCLCPP_ERROR_STREAM(
            get_logger(), first_within_idx.get()
                            << "/" << traj.points.size() << " " << time_to_collision << " < "
                            << time_to_obstacle_getting_out);
          */

          if (time_to_collision > time_to_obstacle_getting_out + margin_for_collision_time) {
            RCLCPP_INFO_EXPRESSION(
              get_logger(), true, "Ignore obstacles since it will not collide with the ego.");
            // False Condition 1. Ignore vehicle obstacles inside the trajectory, which is running
            // and does not collide with ego in a certain time.
            // continue;
          }
        }
      }
    } else {  // obstacles outside the trajectory
      const double max_dist =
        3.0;  // std::max(vehicle_info_.max_longitudinal_offset_m, vehicle_info_.rear_overhang) +
              // std::max(shape.dimensions.x, shape.dimensions.y) / 2.0;
      const bool will_collide = polygon_utils::willCollideWithSurroundObstacle(
        traj, traj_polygons, obstacle.predicted_paths.at(0), obstacle.shape,
        margin_between_traj_and_obstacle, max_dist, max_ego_obj_overlap_time,
        max_prediction_time_for_collision_check);
      if (!will_collide) {
        // False Condition 2. Ignore vehicle obstacles outside the trajectory, whose predicted path
        // overlaps the ego trajectory in a certain time.
        continue;
      }
    }

    target_obstacles.push_back(obstacle);
  }

  // TODO(murooka) change shape of markers based on its shape
  // publish filtered obstacles
  visualization_msgs::msg::MarkerArray object_msg;
  for (size_t i = 0; i < target_obstacles.size(); ++i) {
    const auto marker = obstacle_velocity_utils::getObjectMarkerArray(
      target_obstacles.at(i).pose, i, "target_objects", 0.7, 0.7, 0.0);
    tier4_autoware_utils::appendMarkerArray(marker, &object_msg);
  }
  debug_marker_pub_->publish(object_msg);

  return target_obstacles;
}

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(ObstacleVelocityPlanner)

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

#ifndef ADAPTIVE_CRUISE_CONTROLLER__UTILITIES_HPP_
#define ADAPTIVE_CRUISE_CONTROLLER__UTILITIES_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>

#include <boost/assert.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/format.hpp>
#include <boost/geometry.hpp>

#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace motion_planning
{

namespace bg = boost::geometry;

using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_perception_msgs::msg::PredictedPath;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using TrajectoryPoints = std::vector<TrajectoryPoint>;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Polygon2d;

struct AccParam
{
  double object_low_velocity_thresh;
  double object_stop_velocity_thresh;
  double object_velocity_hysteresis_margin;
  double reset_time_to_acc_state;
  double velocity_to_accleration_weight;
  double acc_min_acceleration;
  double acc_min_jerk;
  double stop_min_acceleration;
  double object_min_acceleration;
  double minimum_margin_distance;
  double idling_time;
  double breaking_delay_time;
  double p_term_in_velocity_pid;
};

enum State { NONE = 0, ACC = 1, STOP = 2 };

struct AccMotion
{
  bool use_target_motion;
  double target_velocity;
  double target_acceleration;
  double target_jerk;
  bool use_trajectory;
  TrajectoryPoints planned_trajectory;
  geometry_msgs::msg::Pose stop_pose;
  bool emergency;  // true when ACC makes a plan to collide with a car in front
};

struct AdaptiveCruiseInformation
{
  rclcpp::Time info_time;  // current time of pose stamped
  double current_ego_velocity;
  double current_object_velocity;
  double current_distance_to_object;
  double ideal_distance_to_object;
  geometry_msgs::msg::Pose current_ego_pose;
  PredictedObject target_object;
  TrajectoryPoints original_trajectory;
};

void convexHull(const std::vector<cv::Point2d> & points, std::vector<cv::Point2d> & polygon_points);
bool isObjectWithinPolygon(
  const std::vector<Polygon2d> & target_polygons, const Polygon2d & object_polygon,
  const double overlap_threshold = 0.1);
bool getDiffAngleWithTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  double & diff_angle);
bool isAngleAlignedWithTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  const double threshold_angle);
bool isPoseNearTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  const double threshold_dist, const double threshold_angle);
bool isPathNearTrajectory(
  const PredictedPath & path, const TrajectoryPoints & trajectory_points,
  const double threshold_dist, const double threshold_angle, const double minmum_overlap_time);

void convertObjectToBoostPolygon(const PredictedObject & object, Polygon2d & object_polygon);
void convertcvPointsToBoostPolygon(
  const std::vector<cv::Point2d> & points, Polygon2d & object_polygon);
bool isClockWise(const Polygon2d & polygon);
Polygon2d inverseClockWise(const Polygon2d & polygon);
geometry_msgs::msg::Pose lerpByPose(
  const geometry_msgs::msg::Pose & p1, const geometry_msgs::msg::Pose & p2, const double t);

}  // namespace motion_planning

#endif  // ADAPTIVE_CRUISE_CONTROLLER__UTILITIES_HPP_

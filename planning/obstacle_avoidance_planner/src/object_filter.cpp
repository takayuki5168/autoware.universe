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
/*
 * This Code is inspired by code from https://github.com/LiJiangnanBit/path_optimizer
 *
 * MIT License
 *
 * Copyright (c) 2020 Li Jiangnan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "obstacle_avoidance_planner/object_filter.hpp"

bool ObjectFilter::getAvoidingObjects(
  const PolygonPoints & polygon_points,
  const autoware_auto_perception_msgs::msg::PredictedObject & object, const cv::Mat & clearance_map,
  const nav_msgs::msg::MapMetaData & map_info,
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const TrajectoryParam & traj_param)
{
}

bool ObjectFilter::isAvoidingObject(
  const PolygonPoints & polygon_points,
  const autoware_auto_perception_msgs::msg::PredictedObject & object, const cv::Mat & clearance_map,
  const nav_msgs::msg::MapMetaData & map_info,
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const TrajectoryParam & traj_param)
{
  if (path_points.empty()) {
    return false;
  }
  if (!isAvoidingObjectType(object, traj_param)) {
    return false;
  }
  const auto image_point = util::transformMapToOptionalImage(
    object.kinematics.initial_pose_with_covariance.pose.position, map_info);
  if (!image_point) {
    return false;
  }

  const int nearest_idx = tier4_autoware_utils::findNearestIndex(
    path_points, object.kinematics.initial_pose_with_covariance.pose.position);
  const auto nearest_path_point = path_points[nearest_idx];
  const auto rel_p = util::transformToRelativeCoordinate2D(
    object.kinematics.initial_pose_with_covariance.pose.position, nearest_path_point.pose);
  // skip object located back the beginning of path points
  if (nearest_idx == 0 && rel_p.x < 0) {
    return false;
  }

  /*
  const float object_clearance_from_road =
    clearance_map.ptr<float>(
      static_cast<int>(image_point.get().y))[static_cast<int>(image_point.get().x)] *
    map_info.resolution;
    */
  const geometry_msgs::msg::Vector3 twist =
    object.kinematics.initial_twist_with_covariance.twist.linear;
  const double vel = std::sqrt(twist.x * twist.x + twist.y * twist.y + twist.z * twist.z);
  /*
  const auto nearest_path_point_image =
    util::transformMapToOptionalImage(nearest_path_point.pose.position, map_info);
  if (!nearest_path_point_image) {
    return false;
  }
  const float nearest_path_point_clearance =
    clearance_map.ptr<float>(static_cast<int>(
      nearest_path_point_image.get().y))[static_cast<int>(nearest_path_point_image.get().x)] *
    map_info.resolution;
  */
  const double lateral_offset_to_path = tier4_autoware_utils::calcLateralOffset(
    path_points, object.kinematics.initial_pose_with_covariance.pose.position);
  if (
    // nearest_path_point_clearance - traj_param.center_line_width * 0.5 <
    // object_clearance_from_road ||
    std::abs(lateral_offset_to_path) < traj_param.center_line_width * 0.5 ||
    vel > traj_param.max_avoiding_objects_velocity_ms ||
    !arePointsInsideDriveableArea(polygon_points.points_in_image, clearance_map)) {
    return false;
  }
  return true;
}

bool ObjectFilter::isAvoidingObjectType(
  const autoware_auto_perception_msgs::msg::PredictedObject & object,
  const TrajectoryParam & traj_param)
{
  if (
    (object.classification.at(0).label == object.classification.at(0).UNKNOWN &&
     traj_param.is_avoiding_unknown) ||
    (object.classification.at(0).label == object.classification.at(0).CAR &&
     traj_param.is_avoiding_car) ||
    (object.classification.at(0).label == object.classification.at(0).TRUCK &&
     traj_param.is_avoiding_truck) ||
    (object.classification.at(0).label == object.classification.at(0).BUS &&
     traj_param.is_avoiding_bus) ||
    (object.classification.at(0).label == object.classification.at(0).BICYCLE &&
     traj_param.is_avoiding_bicycle) ||
    (object.classification.at(0).label == object.classification.at(0).MOTORCYCLE &&
     traj_param.is_avoiding_motorbike) ||
    (object.classification.at(0).label == object.classification.at(0).PEDESTRIAN &&
     traj_param.is_avoiding_pedestrian) ||
    (object.classification.at(0).label == object.classification.at(0).ANIMAL &&
     traj_param.is_avoiding_animal)) {
    return true;
  }
  return false;
}

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

#include <adaptive_cruise_controller/utilities.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace motion_planning
{

void convexHull(const std::vector<cv::Point2d> & points, std::vector<cv::Point2d> & polygon_points)
{
  cv::Point2d centroid;
  centroid.x = 0;
  centroid.y = 0;
  for (const auto & point : points) {
    centroid.x += point.x;
    centroid.y += point.y;
  }
  centroid.x = centroid.x / static_cast<double>(points.size());
  centroid.y = centroid.y / static_cast<double>(points.size());

  std::vector<cv::Point> normalized_points;
  std::vector<cv::Point> normalized_polygon_points;
  for (size_t i = 0; i < points.size(); ++i) {
    // 1mm-order
    normalized_points.push_back(
      cv::Point((points.at(i).x - centroid.x) * 1000.0, (points.at(i).y - centroid.y) * 1000.0));
  }
  cv::convexHull(normalized_points, normalized_polygon_points);

  for (size_t i = 0; i < normalized_polygon_points.size(); ++i) {
    cv::Point2d polygon_point;
    polygon_point.x = (normalized_polygon_points.at(i).x / 1000.0 + centroid.x);
    polygon_point.y = (normalized_polygon_points.at(i).y / 1000.0 + centroid.y);
    polygon_points.push_back(polygon_point);
  }
}

bool isObjectWithinPolygon(
  const std::vector<Polygon2d> & target_polygons, const Polygon2d & object_polygon,
  const double overlap_threshold)
{
  Polygon2d boost_target_polygon;

  for (const auto & target_polygon : target_polygons) {
    std::vector<Polygon2d> overlap_polygons;
    bg::intersection(target_polygon, object_polygon, overlap_polygons);
    double overlap_polygon_area = 0.0;
    for (const auto overlap_polygon : overlap_polygons) {
      overlap_polygon_area += bg::area(overlap_polygon);
    }

    const double overlap_area_rate = overlap_polygon_area / bg::area(object_polygon);

    if (overlap_area_rate > overlap_threshold) {
      return true;
    }
  }
  return false;
}

bool getDiffAngleWithTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  double & diff_angle)
{
  const auto nearest_index = tier4_autoware_utils::findNearestIndex(trajectory_points, pose);
  if (!nearest_index) {
    return false;
  }

  const auto pose_angle = tf2::getYaw(pose.orientation);
  const auto trajectory_angle = tf2::getYaw(trajectory_points.at(*nearest_index).pose.orientation);

  diff_angle = tier4_autoware_utils::normalizeRadian(pose_angle - trajectory_angle);
  return true;
}

bool isAngleAlignedWithTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  const double threshold_angle)
{
  double diff_angle;

  if (!getDiffAngleWithTrajectory(pose, trajectory_points, diff_angle)) {
    return false;
  }

  return std::fabs(diff_angle) <= threshold_angle;
}

bool isPoseNearTrajectory(
  const geometry_msgs::msg::Pose & pose, const TrajectoryPoints & trajectory_points,
  const double threshold_dist, const double threshold_angle)
{
  return static_cast<bool>(tier4_autoware_utils::findNearestIndex(
    trajectory_points, pose, threshold_dist, threshold_angle));
}

bool isPathNearTrajectory(
  const PredictedPath & path, const TrajectoryPoints & trajectory_points,
  const double threshold_dist, const double threshold_angle, const double minmum_overlap_time)
{
  std::vector<size_t> overlap_path_point_index_que;
  for (size_t i = 0; i < path.path.size(); i++) {
    const auto path_pose = path.path.at(i);
    if (isPoseNearTrajectory(path_pose, trajectory_points, threshold_dist, threshold_angle)) {
      overlap_path_point_index_que.emplace_back(i);
    }
  }

  if (overlap_path_point_index_que.size() < 2) {
    // overlap path is too short.
    return false;
  }

  const double timespan_of_overlap_path_point =
    static_cast<double>(
      (overlap_path_point_index_que.back() - overlap_path_point_index_que.front())) *
    rclcpp::Duration(path.time_step).seconds();

  if (timespan_of_overlap_path_point < minmum_overlap_time) {
    // overlap time of path and trajectory is too short.
    return false;
  }

  return true;
}

void convertObjectToBoostPolygon(const PredictedObject & object, Polygon2d & object_polygon)
{
  if (object.shape.type == autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX) {
    const auto & pose = object.kinematics.initial_pose_with_covariance.pose;
    const double yaw = tier4_autoware_utils::normalizeRadian(tf2::getYaw(pose.orientation));
    Eigen::Matrix2d rotation;
    rotation << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw);
    Eigen::Vector2d offset0, offset1, offset2, offset3;
    offset0 = rotation *
              Eigen::Vector2d(object.shape.dimensions.x * 0.5f, object.shape.dimensions.y * 0.5f);
    offset1 = rotation *
              Eigen::Vector2d(object.shape.dimensions.x * 0.5f, -object.shape.dimensions.y * 0.5f);
    offset2 = rotation *
              Eigen::Vector2d(-object.shape.dimensions.x * 0.5f, -object.shape.dimensions.y * 0.5f);
    offset3 = rotation *
              Eigen::Vector2d(-object.shape.dimensions.x * 0.5f, object.shape.dimensions.y * 0.5f);
    object_polygon.outer().push_back(boost::geometry::make<tier4_autoware_utils::Point2d>(
      pose.position.x + offset0.x(), pose.position.y + offset0.y()));
    object_polygon.outer().push_back(boost::geometry::make<tier4_autoware_utils::Point2d>(
      pose.position.x + offset1.x(), pose.position.y + offset1.y()));
    object_polygon.outer().push_back(boost::geometry::make<tier4_autoware_utils::Point2d>(
      pose.position.x + offset2.x(), pose.position.y + offset2.y()));
    object_polygon.outer().push_back(boost::geometry::make<tier4_autoware_utils::Point2d>(
      pose.position.x + offset3.x(), pose.position.y + offset3.y()));
    object_polygon.outer().push_back(object_polygon.outer().front());
  } else if (object.shape.type == autoware_auto_perception_msgs::msg::Shape::CYLINDER) {
    const auto & center = object.kinematics.initial_pose_with_covariance.pose;
    const auto & radius = object.shape.dimensions.x * 0.5;
    constexpr int n = 6;
    for (int i = 0; i < n; ++i) {
      Eigen::Vector2d point;
      point.x() = std::cos(
                    (static_cast<double>(i) / static_cast<double>(n)) * 2.0 * M_PI +
                    M_PI / static_cast<double>(n)) *
                    radius +
                  center.position.x;
      point.y() = std::sin(
                    (static_cast<double>(i) / static_cast<double>(n)) * 2.0 * M_PI +
                    M_PI / static_cast<double>(n)) *
                    radius +
                  center.position.y;
      object_polygon.outer().push_back(
        boost::geometry::make<tier4_autoware_utils::Point2d>(point.x(), point.y()));
    }
    object_polygon.outer().push_back(object_polygon.outer().front());
  } else if (object.shape.type == autoware_auto_perception_msgs::msg::Shape::POLYGON) {
    const auto & pose = object.kinematics.initial_pose_with_covariance.pose;
    for (const auto & point : object.shape.footprint.points) {
      object_polygon.outer().push_back(boost::geometry::make<tier4_autoware_utils::Point2d>(
        pose.position.x + point.x, pose.position.y + point.y));
    }
    object_polygon.outer().push_back(object_polygon.outer().front());
  }
  object_polygon = isClockWise(object_polygon) ? object_polygon : inverseClockWise(object_polygon);
}

void convertcvPointsToBoostPolygon(
  const std::vector<cv::Point2d> & points, Polygon2d & boost_polygon)
{
  for (const auto & point : points) {
    boost_polygon.outer().push_back(bg::make<Point2d>(point.x, point.y));
  }
  boost_polygon.outer().push_back(bg::make<Point2d>(points.front().x, points.front().y));
}

geometry_msgs::msg::Pose lerpByPose(
  const geometry_msgs::msg::Pose & p1, const geometry_msgs::msg::Pose & p2, const double t)
{
  tf2::Transform tf_transform1, tf_transform2;
  tf2::fromMsg(p1, tf_transform1);
  tf2::fromMsg(p2, tf_transform2);
  const auto & tf_point = tf2::lerp(tf_transform1.getOrigin(), tf_transform2.getOrigin(), t);
  const auto & tf_quaternion =
    tf2::slerp(tf_transform1.getRotation(), tf_transform2.getRotation(), t);

  geometry_msgs::msg::Pose pose{};
  pose.position.x = tf_point.x();
  pose.position.y = tf_point.y();
  pose.position.z = tf_point.z();
  pose.orientation = tf2::toMsg(tf_quaternion);
  return pose;
}

bool isClockWise(const Polygon2d & polygon)
{
  const auto n = polygon.outer().size();

  const double x_offset = polygon.outer().at(0).x();
  const double y_offset = polygon.outer().at(0).y();
  double sum = 0.0;
  for (std::size_t i = 0; i < polygon.outer().size(); ++i) {
    sum +=
      (polygon.outer().at(i).x() - x_offset) * (polygon.outer().at((i + 1) % n).y() - y_offset) -
      (polygon.outer().at(i).y() - y_offset) * (polygon.outer().at((i + 1) % n).x() - x_offset);
  }
  return sum < 0.0;
}

Polygon2d inverseClockWise(const Polygon2d & polygon)
{
  const auto & poly = polygon.outer();
  Polygon2d inverted_polygon;
  inverted_polygon.outer().reserve(poly.size());
  std::reverse_copy(poly.begin(), poly.end(), std::back_inserter(inverted_polygon.outer()));
  return inverted_polygon;
}

}  // namespace motion_planning
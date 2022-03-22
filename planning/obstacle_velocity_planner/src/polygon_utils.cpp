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

#include "obstacle_velocity_planner/polygon_utils.hpp"

namespace polygon_utils
{
void appendPointToPolygon(Polygon2d & polygon, const geometry_msgs::msg::Point & geom_point)
{
  Point2d point;
  point.x() = geom_point.x;
  point.y() = geom_point.y;

  bg::append(polygon.outer(), point);
}

void appendPointToPolygon(Polygon2d & polygon, const geometry_msgs::msg::Point32 & geom_point)
{
  Point2d point;
  point.x() = geom_point.x;
  point.y() = geom_point.y;

  bg::append(polygon.outer(), point);
}

Polygon2d convertObstacleToPolygon(
  const geometry_msgs::msg::Pose & pose, const autoware_auto_perception_msgs::msg::Shape & shape)
{
  Polygon2d polygon;

  if (shape.type == autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX) {
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(
                 pose, shape.dimensions.x / 2.0, shape.dimensions.y / 2.0, 0.0)
                 .position);
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(
                 pose, -shape.dimensions.x / 2.0, shape.dimensions.y / 2.0, 0.0)
                 .position);
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(
                 pose, -shape.dimensions.x / 2.0, -shape.dimensions.y / 2.0, 0.0)
                 .position);
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(
                 pose, shape.dimensions.x / 2.0, -shape.dimensions.y / 2.0, 0.0)
                 .position);
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(
                 pose, shape.dimensions.x / 2.0, shape.dimensions.y / 2.0, 0.0)
                 .position);
  } else if (shape.type == autoware_auto_perception_msgs::msg::Shape::CYLINDER) {
    // TODO(murooka)
  } else if (shape.type == autoware_auto_perception_msgs::msg::Shape::POLYGON) {
    for (const auto point : shape.footprint.points) {
      appendPointToPolygon(polygon, point);
    }
  } else {
    throw std::logic_error("The shape type is not supported in obstacle_velocity_planner.");
  }

  return polygon;
}

boost::optional<size_t> getFirstCollisionIndex(
  const std::vector<Polygon2d> & traj_polygons, const Polygon2d & obj_polygon, const double margin)
{
  for (size_t i = 0; i < traj_polygons.size(); ++i) {
    const double dist = bg::distance(traj_polygons.at(i), obj_polygon);

    if (dist < margin) {
      return i;
    }
  }

  return {};
}

boost::optional<size_t> getFirstNonCollisionIndex(
  const std::vector<Polygon2d> & traj_polygons,
  const autoware_auto_perception_msgs::msg::PredictedPath & predicted_path,
  const autoware_auto_perception_msgs::msg::Shape & shape, const size_t start_idx,
  const double dist_margin)
{
  for (size_t i = start_idx; i < traj_polygons.size(); ++i) {
    double min_dist = std::numeric_limits<double>::max();
    for (const auto & path_point : predicted_path.path) {
      const auto obj_polygon = convertObstacleToPolygon(path_point, shape);

      const double dist = bg::distance(traj_polygons.at(i), obj_polygon);
      if (dist < min_dist) {
        min_dist = dist;
      }

      if (min_dist < dist_margin) {
        break;
      }
    }
    if (min_dist > dist_margin) {
      return i;
    }
  }

  return {};
}

bool willCollideWithSurroundObstacle(
  const autoware_auto_planning_msgs::msg::Trajectory & traj,
  const std::vector<Polygon2d> & traj_polygons,
  const autoware_auto_perception_msgs::msg::PredictedPath & predicted_path,
  const autoware_auto_perception_msgs::msg::Shape & shape, const double dist_margin,
  const double max_dist, const double max_ego_obj_overlap_time,
  const double max_prediction_time_for_collision_check)
{
  boost::optional<size_t> start_predicted_path_idx = {};
  for (size_t i = 0; i < predicted_path.path.size(); ++i) {
    const auto & path_point = predicted_path.path.at(i);
    if (
      max_prediction_time_for_collision_check <
      rclcpp::Duration(predicted_path.time_step).seconds() * static_cast<double>(i)) {
      return false;
    }

    for (size_t j = 0; j < traj.points.size(); ++j) {
      const auto & traj_point = traj.points.at(j);
      const double approximated_dist =
        tier4_autoware_utils::calcDistance2d(path_point.position, traj_point.pose.position);
      if (approximated_dist > dist_margin + max_dist) {
        continue;
      }

      const auto & traj_polygon = traj_polygons.at(j);
      const auto obj_polygon = polygon_utils::convertObstacleToPolygon(path_point, shape);
      const double dist = bg::distance(traj_polygon, obj_polygon);

      if (dist < dist_margin) {
        if (!start_predicted_path_idx) {
          start_predicted_path_idx = i;
          std::cerr << "idx " << start_predicted_path_idx.get() << std::endl;
        } else {
          const double overlap_time = static_cast<double>(i - start_predicted_path_idx.get()) *
                                      rclcpp::Duration(predicted_path.time_step).seconds();
          // std::cerr << overlap_time << std::endl;
          if (max_ego_obj_overlap_time < overlap_time) {
            return true;
          }
        }
      } else {
        start_predicted_path_idx = {};
      }
    }
  }

  return false;
}
std::vector<Polygon2d> createOneStepPolygons(
  const autoware_auto_planning_msgs::msg::Trajectory & traj,
  const vehicle_info_util::VehicleInfo & vehicle_info)
{
  std::vector<Polygon2d> polygons;

  for (size_t i = 0; i < traj.points.size(); ++i) {
    const auto polygon = [&]() {
      if (i == 0) {
        return createOneStepPolygon(traj.points.at(i).pose, traj.points.at(i).pose, vehicle_info);
      }
      return createOneStepPolygon(traj.points.at(i - 1).pose, traj.points.at(i).pose, vehicle_info);
    }();

    polygons.push_back(polygon);
  }
  return polygons;
}

Polygon2d createOneStepPolygon(
  const geometry_msgs::msg::Pose & base_step_pose, const geometry_msgs::msg::Pose & next_step_pose,
  const vehicle_info_util::VehicleInfo & vehicle_info)
{
  Polygon2d polygon;

  const double longitudinal_offset = vehicle_info.max_longitudinal_offset_m;
  const double width = vehicle_info.vehicle_width_m / 2.0;
  const double rear_overhang = vehicle_info.rear_overhang_m;

  {  // base step
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(base_step_pose, longitudinal_offset, width, 0.0)
                 .position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(base_step_pose, longitudinal_offset, -width, 0.0)
        .position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(base_step_pose, -rear_overhang, -width, 0.0).position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(base_step_pose, -rear_overhang, width, 0.0).position);
  }

  {  // next step
    appendPointToPolygon(
      polygon, tier4_autoware_utils::calcOffsetPose(next_step_pose, longitudinal_offset, width, 0.0)
                 .position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(next_step_pose, longitudinal_offset, -width, 0.0)
        .position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(next_step_pose, -rear_overhang, -width, 0.0).position);
    appendPointToPolygon(
      polygon,
      tier4_autoware_utils::calcOffsetPose(next_step_pose, -rear_overhang, width, 0.0).position);
  }

  Polygon2d hull_polygon;
  bg::convex_hull(polygon, hull_polygon);

  return hull_polygon;
}
}  // namespace polygon_utils

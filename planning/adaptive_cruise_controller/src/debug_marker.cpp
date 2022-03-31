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

#include <adaptive_cruise_controller/debug_marker.hpp>
#include <tier4_autoware_utils/tier4_autoware_utils.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace motion_planning
{

using tier4_autoware_utils::calcOffsetPose;
using tier4_autoware_utils::createDefaultMarker;
using tier4_autoware_utils::createMarkerColor;
using tier4_autoware_utils::createMarkerOrientation;
using tier4_autoware_utils::createMarkerScale;
using tier4_autoware_utils::createPoint;
using tier4_autoware_utils::createSlowDownVirtualWallMarker;
using tier4_autoware_utils::createStopVirtualWallMarker;

AdaptiveCruiseControllerDebugNode::AdaptiveCruiseControllerDebugNode(rclcpp::Node * node)
: node_(node)
{
  pub_debug_marker_ =
    node_->create_publisher<visualization_msgs::msg::MarkerArray>("~/debug/marker", 1);

  pub_debug_value_ = node_->create_publisher<Float32MultiArrayStamped>("~/debug/debug_values", 1);
}

void AdaptiveCruiseControllerDebugNode::setVirtualWall(
  const geometry_msgs::msg::Pose & stop_pose, const double offset, const bool is_stop_wall)
{
  const auto p = calcOffsetPose(stop_pose, offset, 0.0, 0.0);
  MarkerArray markers;

  if (is_stop_wall) {
    markers = createStopVirtualWallMarker(p, "adaptive cruise", node_->now(), 0);
  } else {
    markers = createSlowDownVirtualWallMarker(p, "adaptive cruise", node_->now(), 0);
  }

  virtual_wall_marker_ptr_ = std::make_shared<MarkerArray>(markers);
}
void AdaptiveCruiseControllerDebugNode::setTargetPolygon(const PredictedObject & object)
{
  auto marker = createDefaultMarker(
    "map", node_->now(), "adaptive_cruise_polygons", 0, Marker::CUBE,
    createMarkerScale(3.0, 1.0, 1.0), createMarkerColor(1.0, 1.0, 0.0, 0.999));
  marker.pose = object.kinematics.initial_pose_with_covariance.pose;
  target_vehicle_marker_ptr_ = std::make_shared<Marker>(marker);
}

void AdaptiveCruiseControllerDebugNode::clearMarker()
{
  clearVirtualWall();
  clearTargetPolygon();
}

void AdaptiveCruiseControllerDebugNode::clearVirtualWall() { virtual_wall_marker_ptr_.reset(); }

void AdaptiveCruiseControllerDebugNode::clearTargetPolygon() { target_vehicle_marker_ptr_.reset(); }

void AdaptiveCruiseControllerDebugNode::publish()
{
  publishMarker();
  publishDebugValues();
}

void AdaptiveCruiseControllerDebugNode::publishMarker()
{
  MarkerArray marker_array;

  // virtual wall marker
  if (virtual_wall_marker_ptr_) {
    for (const auto marker : virtual_wall_marker_ptr_->markers) {
      marker_array.markers.emplace_back(marker);
    }
  }

  // target vehicle marker
  if (target_vehicle_marker_ptr_) {
    marker_array.markers.emplace_back(*target_vehicle_marker_ptr_);
  }

  pub_debug_marker_->publish(marker_array);
}

void AdaptiveCruiseControllerDebugNode::publishDebugValues()
{
  pub_debug_value_->publish(convertDebugValuesToMsg(debug_values_));
}

Float32MultiArrayStamped AdaptiveCruiseControllerDebugNode::convertDebugValuesToMsg(
  const DebugValues debug_value)
{
  Float32MultiArrayStamped debug_msg{};
  debug_msg.stamp = node_->now();
  for (const auto & v : debug_value.getValues()) {
    debug_msg.data.push_back(v);
  }
  return debug_msg;
}

}  // namespace motion_planning

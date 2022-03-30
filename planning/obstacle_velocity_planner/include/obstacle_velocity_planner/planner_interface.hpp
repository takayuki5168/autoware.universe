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

#ifndef OBSTACLE_VELOCITY_PLANNER__PLANNER_INTERFACE_HPP_
#define OBSTACLE_VELOCITY_PLANNER__PLANNER_INTERFACE_HPP_

#include "obstacle_velocity_planner/common_structs.hpp"
#include "vehicle_info_util/vehicle_info_util.hpp"

#include <boost/optional.hpp>

#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/BoundingBox.h>
#include <lanelet2_core/geometry/Lanelet.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <memory>
#include <vector>

class PlannerInterface
{
public:
  PlannerInterface(
    const double max_accel, const double min_accel, const double max_jerk, const double min_jerk,
    const double min_object_accel, const double idling_time,
    const vehicle_info_util::VehicleInfo & vehicle_info)
  : longitudinal_info_(
      RSSLongitudinalInfo(max_accel, min_accel, max_jerk, min_jerk, min_object_accel, idling_time)),
    vehicle_info_(vehicle_info)
  {
  }

  PlannerInterface() = default;

  virtual autoware_auto_planning_msgs::msg::Trajectory generateTrajectory(
    const ObstacleVelocityPlannerData & planner_data, LongitudinalMotion & target_motion) = 0;

  virtual void updateParam(const std::vector<rclcpp::Parameter> & parameters) {}

  // TODO(shimizu) remove this function
  void setMaps(
    const std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr,
    const std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr,
    const std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr)
  {
    lanelet_map_ptr_ = lanelet_map_ptr;
    traffic_rules_ptr_ = traffic_rules_ptr;
    routing_graph_ptr_ = routing_graph_ptr;
  }

  // TODO(shimizu) remove this function
  void setSmoothedTrajectory(const autoware_auto_planning_msgs::msg::Trajectory::SharedPtr traj)
  {
    smoothed_trajectory_ptr_ = traj;
  }

protected:
  // Parameters
  RSSLongitudinalInfo longitudinal_info_;

  // Vehicle Parameters
  vehicle_info_util::VehicleInfo vehicle_info_;

  // TODO(shimizu) remove these parameters
  // Lanelet Map Pointers
  std::shared_ptr<lanelet::LaneletMap> lanelet_map_ptr_;
  std::shared_ptr<lanelet::routing::RoutingGraph> routing_graph_ptr_;
  std::shared_ptr<lanelet::traffic_rules::TrafficRules> traffic_rules_ptr_;
  autoware_auto_planning_msgs::msg::Trajectory::SharedPtr smoothed_trajectory_ptr_;

  double calcRSSDistance(
    const double ego_vel, const double obj_vel, const double margin = 0.0) const
  {
    const auto & i = longitudinal_info_;
    const double rss_dist_with_margin =
      ego_vel * i.idling_time + 0.5 * i.max_accel * std::pow(i.idling_time, 2) +
      std::pow(ego_vel + i.max_accel * i.idling_time, 2) * 0.5 / std::abs(i.min_accel) -
      std::pow(obj_vel, 2) * 0.5 / std::abs(i.min_object_accel) + margin;
    return rss_dist_with_margin;
  }
};

#endif  // OBSTACLE_VELOCITY_PLANNER__PLANNER_INTERFACE_HPP_

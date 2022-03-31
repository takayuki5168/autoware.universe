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

#ifndef ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_HPP_
#define ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_HPP_

#include <adaptive_cruise_controller/adaptive_cruise_control_core.hpp>
#include <adaptive_cruise_controller/debug_marker.hpp>
#include <adaptive_cruise_controller/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tier4_autoware_utils/trajectory/tmp_conversion.hpp>
#include <vehicle_info_util/vehicle_info_util.hpp>

#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_perception_msgs/msg/predicted_objects.hpp>
#include <autoware_auto_planning_msgs/msg/trajectory.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tier4_debug_msgs/msg/float32_multi_array_stamped.hpp>
#include <tier4_planning_msgs/msg/velocity_limit.hpp>
#include <tier4_planning_msgs/msg/velocity_limit_clear_command.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <boost/assert.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/format.hpp>
#include <boost/geometry.hpp>

namespace motion_planning
{
namespace bg = boost::geometry;
using autoware_auto_perception_msgs::msg::ObjectClassification;
using autoware_auto_perception_msgs::msg::PredictedObject;
using autoware_auto_perception_msgs::msg::PredictedObjects;
using autoware_auto_perception_msgs::msg::PredictedPath;
using autoware_auto_planning_msgs::msg::Trajectory;
using autoware_auto_planning_msgs::msg::TrajectoryPoint;
using tier4_planning_msgs::msg::VelocityLimit;
using tier4_planning_msgs::msg::VelocityLimitClearCommand;
using TrajectoryPoints = std::vector<TrajectoryPoint>;
using nav_msgs::msg::Odometry;
using tier4_autoware_utils::Point2d;
using tier4_autoware_utils::Polygon2d;
using vehicle_info_util::VehicleInfo;

enum class CUT_IN_OUT {
  NONE = 0,
  CUT_IN = 1,
  CUT_OUT = 2,
};

class AdaptiveCruiseControllerNode : public rclcpp::Node
{
public:
  explicit AdaptiveCruiseControllerNode(const rclcpp::NodeOptions & options);

private:
  struct Param
  {
    // step length of decimated trajectory
    double decimate_step_length;

    // length to extend trajectory forward for
    double trajectory_extend_length;

    // length to expand rough detection area
    double rough_detection_area_expand_length;

    // length to expand detection area
    double detection_area_expand_length;

    // minimum diff-angle between object and trajectory to be considered as a target objects
    double object_threshold_angle;

    // minimum diff-angle between preidcted_path and trajectory to be considered as a target objects
    double predicted_path_threshold_angle;

    // minimum overlap time of predicted path and trajectory to be considered as a target objects
    double mininum_overlap_time_of_predicted_path;
  };
  Param param_;

  AccParam acc_param_;

  rclcpp::Publisher<Trajectory>::SharedPtr pub_trajectory_;
  rclcpp::Publisher<VelocityLimitClearCommand>::SharedPtr pub_clear_velocity_limit_;
  rclcpp::Publisher<VelocityLimit>::SharedPtr pub_velocity_limit_;
  rclcpp::Subscription<Trajectory>::SharedPtr sub_trajectory_;
  rclcpp::Subscription<PredictedObjects>::SharedPtr sub_objects_;
  rclcpp::Subscription<Odometry>::SharedPtr sub_odometry_;

  tier4_autoware_utils::SelfPoseListener self_pose_listener_;

  void onTrajectory(const Trajectory::ConstSharedPtr msg);
  void onPredictedObjects(const PredictedObjects::ConstSharedPtr msg);
  void onOdometry(const Odometry::ConstSharedPtr msg);

  bool isDataReady();
  TrajectoryPoints trimTrajectoryWithIndexFromSelfPose(
    const TrajectoryPoints & input, const geometry_msgs::msg::PoseStamped & self_pose);
  TrajectoryPoint getExtendTrajectoryPoint(
    const double extend_distance, const TrajectoryPoint & goal_point);
  TrajectoryPoints extendTrajectory(const TrajectoryPoints & input, const double extend_distance);
  TrajectoryPoints decimateTrajectory(const TrajectoryPoints & input, const double step_length);
  TrajectoryPoint getBackwardPointFromBasePoint(
    const TrajectoryPoint & p_from, const TrajectoryPoint & p_to, const TrajectoryPoint & p_base,
    const double backward_length);
  void createPolygonFromTrajectoryPoints(
    const TrajectoryPoints & trajectory_points, const double expand_width,
    std::vector<Polygon2d> & polygon);
  void extractVehicleObjectsInPolygon(
    const PredictedObjects & objects, const std::vector<Polygon2d> & target_polygons,
    PredictedObjects & objects_in_polygon);
  void getTargetObjects(
    const PredictedObjects & objects, const std::vector<Polygon2d> & target_polygons,
    const TrajectoryPoints & target_trajectory, PredictedObjects & target_objects);
  void getNearestObject(
    const PredictedObjects & objects, const TrajectoryPoints & trajectory_points,
    PredictedObject & nearest_object);
  VelocityLimit createVelocityLimitMsg(
    const double velocity, const double acceleration, const double jerk);
  VelocityLimitClearCommand createVelocityLimitClearCommandMsg();

  PredictedPath getHighestConfidencePathFromObject(const PredictedObject & object);
  geometry_msgs::msg::Pose getObjectPose(const PredictedObject & object);
  bool isObjectVelocityHigh(const PredictedObject & object);
  bool isTargetObjectType(const PredictedObject & object);

  Odometry::ConstSharedPtr current_odometry_ptr_;
  PredictedObjects::ConstSharedPtr current_objects_ptr_;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr current_pose_;
  VehicleInfo vehicle_info_;

  std::shared_ptr<AdaptiveCruiseControlCore> controller_ptr_;

  // only for debug
  void fillAndPublishDebugOutput(const PredictedObject & target_object);
  void publishDebugOutputWithNoTarget();
  double calcAcc(
    const geometry_msgs::msg::TwistStamped & twist,
    std::shared_ptr<geometry_msgs::msg::TwistStamped> & prev_twist, const double prev_acc,
    const double lowpass_gain = 0.6, const double timeout = 1.0);

  void resetObjectTwistHistory();
  CUT_IN_OUT detectCutInAndOut(
    AdaptiveCruiseInformation acc_info, const double threshold_length = 10.0);

  geometry_msgs::msg::TwistStamped toTwistStamped(
    const std_msgs::msg::Header & header, const geometry_msgs::msg::Twist & twist);

  std::shared_ptr<AdaptiveCruiseControllerDebugNode> debug_node_ptr_;
  std::shared_ptr<AdaptiveCruiseInformation> prev_acc_info_;
  std::shared_ptr<geometry_msgs::msg::TwistStamped> prev_ego_twist_;
  double ego_accel_ = 0.0;
  std::shared_ptr<geometry_msgs::msg::TwistStamped> prev_object_twist_;
  double obj_accel_ = 0.0;

  bool need_to_clear_velocity_limit_ = false;
};

}  // namespace motion_planning

#endif  // ADAPTIVE_CRUISE_CONTROLLER__ADAPTIVE_CRUISE_CONTROL_HPP_

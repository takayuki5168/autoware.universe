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

#include <adaptive_cruise_controller/adaptive_cruise_control.hpp>
#include <adaptive_cruise_controller/adaptive_cruise_control_core.hpp>
#include <adaptive_cruise_controller/debug_marker.hpp>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace motion_planning
{
AdaptiveCruiseControllerNode::AdaptiveCruiseControllerNode(const rclcpp::NodeOptions & options)
: Node("adaptive_cruise_controller", options), self_pose_listener_(this)
{
  using std::placeholders::_1;

  // set param
  param_.decimate_step_length = declare_parameter("decimate_step_length", 1.0);
  param_.trajectory_extend_length = declare_parameter("trajectory_extend_length", 5.0);
  param_.rough_detection_area_expand_length =
    declare_parameter("rough_detection_area_expand_length", 3.0);
  param_.detection_area_expand_length = declare_parameter("detection_area_expand_length", 1.0);
  param_.object_threshold_angle = declare_parameter("object_threshold_angle", M_PI / 6.0);
  param_.predicted_path_threshold_angle =
    declare_parameter("predicted_path_threshold_angle", M_PI / 6.0);
  param_.mininum_overlap_time_of_predicted_path =
    declare_parameter("mininum_overlap_time_of_predicted_path", 1.0);

  // set acc param
  acc_param_.object_low_velocity_thresh = declare_parameter("object_low_velocity_thresh", 3.0);
  acc_param_.object_stop_velocity_thresh = declare_parameter("object_stop_velocity_thresh", 1.0);
  acc_param_.object_velocity_hysteresis_margin =
    declare_parameter("object_velocity_hysteresis_margin", 1.0);
  acc_param_.reset_time_to_acc_state = declare_parameter("reset_time_to_acc_state", 1.0);
  acc_param_.velocity_to_accleration_weight =
    declare_parameter("velocity_to_accleration_weight", 0.3);
  acc_param_.acc_min_acceleration = declare_parameter("acc_min_acceleration", -1.5);
  acc_param_.acc_min_jerk = declare_parameter("acc_min_jerk", -1.0);
  acc_param_.stop_min_acceleration = declare_parameter("stop_min_acceleration", -3.0);
  acc_param_.object_min_acceleration = declare_parameter("object_min_acceleration", -3.0);
  acc_param_.minimum_margin_distance = declare_parameter("minimum_margin_distance", 5.0);
  acc_param_.idling_time = declare_parameter("idling_time", 1.0);
  acc_param_.breaking_delay_time = declare_parameter("breaking_delay_time", 0.5);
  acc_param_.p_term_in_velocity_pid = declare_parameter("p_term_in_velocity_pid", 0.3);

  // vehicle parameters
  vehicle_info_ = vehicle_info_util::VehicleInfoUtil(*this).getVehicleInfo();

  // debug node
  debug_node_ptr_ = std::make_shared<AdaptiveCruiseControllerDebugNode>(this);

  // controller
  controller_ptr_ = std::make_shared<AdaptiveCruiseControlCore>(
    vehicle_info_.max_longitudinal_offset_m, acc_param_);

  // wait for self pose
  self_pose_listener_.waitForFirstPose();

  // publishers, subscribers
  pub_trajectory_ = create_publisher<Trajectory>("~/output/trajectory", 1);
  pub_velocity_limit_ = create_publisher<VelocityLimit>("~/output/velocity_limit", 1);
  pub_clear_velocity_limit_ =
    create_publisher<VelocityLimitClearCommand>("~/output/clear_velocity_limit", 1);

  sub_trajectory_ = create_subscription<Trajectory>(
    "~/input/trajectory", 1, std::bind(&AdaptiveCruiseControllerNode::onTrajectory, this, _1));
  sub_odometry_ = create_subscription<Odometry>(
    "~/input/odometry", 1, std::bind(&AdaptiveCruiseControllerNode::onOdometry, this, _1));
  sub_objects_ = create_subscription<PredictedObjects>(
    "~/input/objects", 1, std::bind(&AdaptiveCruiseControllerNode::onPredictedObjects, this, _1));
}

void AdaptiveCruiseControllerNode::onTrajectory(const Trajectory::ConstSharedPtr msg)
{
  // initialize
  debug_node_ptr_->resetDebugValues();

  current_pose_ = self_pose_listener_.getCurrentPose();
  if (!isDataReady()) {
    return;
  }

  TrajectoryPoints trajectory_points = tier4_autoware_utils::convertToTrajectoryPointArray(*msg);

  // trim trajectory from self pose
  const auto base_trajectory =
    trimTrajectoryWithIndexFromSelfPose(trajectory_points, *current_pose_);
  // extend trajectory to consider objects after the goal
  const auto extend_trajectory = extendTrajectory(base_trajectory, param_.trajectory_extend_length);
  // decimate trajectory for calculation cost
  const auto decimate_trajectory =
    decimateTrajectory(extend_trajectory, param_.decimate_step_length);

  if (decimate_trajectory.size() < 2) {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "trajectory is too short.");
    pub_trajectory_->publish(*msg);
    return;
  }

  // create rough detection area
  std::vector<Polygon2d> rough_detection_area;
  createPolygonFromTrajectoryPoints(
    decimate_trajectory, param_.rough_detection_area_expand_length, rough_detection_area);
  // extract objects in rough detection area
  PredictedObjects objects_in_rough_detection_area;
  extractVehicleObjectsInPolygon(
    *current_objects_ptr_, rough_detection_area, objects_in_rough_detection_area);

  // create detection area
  std::vector<Polygon2d> detection_area;
  createPolygonFromTrajectoryPoints(
    decimate_trajectory, param_.detection_area_expand_length, detection_area);

  // extract target objects of adaptive cruise controller
  PredictedObjects target_objects;
  getTargetObjects(
    objects_in_rough_detection_area, detection_area, decimate_trajectory, target_objects);

  if (target_objects.objects.empty()) {
    // no target object
    publishDebugOutputWithNoTarget();
    pub_trajectory_->publish(*msg);
    if (need_to_clear_velocity_limit_) {
      pub_clear_velocity_limit_->publish(createVelocityLimitClearCommandMsg());
      need_to_clear_velocity_limit_ = false;
    }
    return;
  }

  // get nearest object from target object
  PredictedObject nearest_target_object;
  getNearestObject(target_objects, decimate_trajectory, nearest_target_object);

  // send velocity / insert velocity to trajectory
  TrajectoryPoints output;
  {
    // get information for adaptive cruise control
    const auto object_time = rclcpp::Time(target_objects.header.stamp);
    controller_ptr_->calcInformationForAdaptiveCruise(
      trajectory_points, *current_pose_, *current_odometry_ptr_, nearest_target_object,
      object_time);

    // calculate target motion (target_velocity, and trajectory)
    controller_ptr_->calculate();

    // send target velocity
    double target_velocity;
    double target_acceleration;
    double target_jerk;
    if (controller_ptr_->getTargetMotion(target_velocity, target_acceleration, target_jerk)) {
      pub_velocity_limit_->publish(
        createVelocityLimitMsg(target_velocity, target_acceleration, target_jerk));
      need_to_clear_velocity_limit_ = true;
    } else {
      if (need_to_clear_velocity_limit_) {
        pub_clear_velocity_limit_->publish(createVelocityLimitClearCommandMsg());
        need_to_clear_velocity_limit_ = false;
      }
    }

    // get trajectory (If necessary, insert stop velocity.)
    bool emergency;
    output = controller_ptr_->getAccTrajectory(emergency);
  }

  // publish trajectory
  auto output_trajectory = tier4_autoware_utils::convertToTrajectory(output);
  output_trajectory.header = msg->header;
  pub_trajectory_->publish(output_trajectory);

  // TODO(kimura) send emergency flag

  // for debug
  fillAndPublishDebugOutput(nearest_target_object);
}

void AdaptiveCruiseControllerNode::onOdometry(const Odometry::ConstSharedPtr msg)
{
  current_odometry_ptr_ = msg;
}

void AdaptiveCruiseControllerNode::onPredictedObjects(const PredictedObjects::ConstSharedPtr msg)
{
  current_objects_ptr_ = msg;
}

bool AdaptiveCruiseControllerNode::isDataReady()
{
  if (!current_pose_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "waiting for current_pose...");
    return false;
  }

  if (!current_odometry_ptr_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "waiting for odometry msg...");
    return false;
  }

  if (!current_objects_ptr_) {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000, "waiting for objects msg...");
    return false;
  }

  return true;
}

TrajectoryPoints AdaptiveCruiseControllerNode::trimTrajectoryWithIndexFromSelfPose(
  const TrajectoryPoints & input, const geometry_msgs::msg::PoseStamped & self_pose)
{
  TrajectoryPoints output{};

  double min_distance = 0.0;
  size_t min_distance_index = 0;
  bool is_init = false;
  for (size_t i = 0; i < input.size(); ++i) {
    const double x = input.at(i).pose.position.x - self_pose.pose.position.x;
    const double y = input.at(i).pose.position.y - self_pose.pose.position.y;
    const double squared_distance = x * x + y * y;
    if (!is_init || squared_distance < min_distance * min_distance) {
      is_init = true;
      min_distance = std::sqrt(squared_distance);
      min_distance_index = i;
    }
  }
  for (size_t i = min_distance_index; i < input.size(); ++i) {
    output.push_back(input.at(i));
  }

  return output;
}

TrajectoryPoint AdaptiveCruiseControllerNode::getExtendTrajectoryPoint(
  const double extend_distance, const TrajectoryPoint & goal_point)
{
  tf2::Transform map2goal;
  tf2::fromMsg(goal_point.pose, map2goal);
  tf2::Transform local_extend_point;
  local_extend_point.setOrigin(tf2::Vector3(extend_distance, 0.0, 0.0));
  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  local_extend_point.setRotation(q);
  const auto map2extend_point = map2goal * local_extend_point;
  geometry_msgs::msg::Pose extend_pose;
  tf2::toMsg(map2extend_point, extend_pose);
  TrajectoryPoint extend_trajectory_point;
  extend_trajectory_point.pose = extend_pose;
  extend_trajectory_point.longitudinal_velocity_mps = goal_point.longitudinal_velocity_mps;
  extend_trajectory_point.lateral_velocity_mps = goal_point.lateral_velocity_mps;
  extend_trajectory_point.acceleration_mps2 = goal_point.acceleration_mps2;
  return extend_trajectory_point;
}

TrajectoryPoints AdaptiveCruiseControllerNode::extendTrajectory(
  const TrajectoryPoints & input, const double extend_distance)
{
  TrajectoryPoints output = input;

  if (extend_distance < std::numeric_limits<double>::epsilon()) {
    return output;
  }

  const auto goal_point = input.back();
  double interpolation_distance = 0.1;

  double extend_sum = 0.0;
  while (extend_sum <= (extend_distance - interpolation_distance)) {
    const auto extend_trajectory_point = getExtendTrajectoryPoint(extend_sum, goal_point);
    output.push_back(extend_trajectory_point);
    extend_sum += interpolation_distance;
  }
  const auto extend_trajectory_point = getExtendTrajectoryPoint(extend_distance, goal_point);
  output.push_back(extend_trajectory_point);

  return output;
}

TrajectoryPoints AdaptiveCruiseControllerNode::decimateTrajectory(
  const TrajectoryPoints & input, const double step_length)
{
  TrajectoryPoints output{};

  if (input.empty()) {
    return output;
  }

  double trajectory_length_sum = 0.0;
  double next_length = 0.0;

  for (int i = 0; i < static_cast<int>(input.size()) - 1; ++i) {
    const auto & p_front = input.at(i);
    const auto & p_back = input.at(i + 1);
    constexpr double epsilon = 1e-3;

    if (next_length <= trajectory_length_sum + epsilon) {
      const auto p_interpolate =
        getBackwardPointFromBasePoint(p_front, p_back, p_back, next_length - trajectory_length_sum);
      output.push_back(p_interpolate);
      next_length += step_length;
      continue;
    }

    trajectory_length_sum += tier4_autoware_utils::calcDistance2d(p_front, p_back);
  }

  output.push_back(input.back());

  return output;
}

TrajectoryPoint AdaptiveCruiseControllerNode::getBackwardPointFromBasePoint(
  const TrajectoryPoint & p_from, const TrajectoryPoint & p_to, const TrajectoryPoint & p_base,
  const double backward_length)
{
  TrajectoryPoint output;
  const double dx = p_to.pose.position.x - p_from.pose.position.x;
  const double dy = p_to.pose.position.y - p_from.pose.position.y;
  const double norm = std::hypot(dx, dy);

  output = p_base;
  output.pose.position.x += backward_length * dx / norm;
  output.pose.position.y += backward_length * dy / norm;

  return output;
}

VelocityLimit AdaptiveCruiseControllerNode::createVelocityLimitMsg(
  const double velocity, const double acceleration, const double jerk)
{
  VelocityLimit msg;
  msg.stamp = this->now();
  msg.sender = "adaptive_cruise_controller";
  msg.max_velocity = velocity;
  msg.use_constraints = true;
  msg.constraints.min_acceleration = std::min(0.0, acceleration);
  msg.constraints.min_jerk = std::min(0.0, jerk);
  msg.constraints.max_jerk = -msg.constraints.min_jerk;
  return msg;
}

VelocityLimitClearCommand AdaptiveCruiseControllerNode::createVelocityLimitClearCommandMsg()
{
  VelocityLimitClearCommand msg;
  msg.stamp = this->now();
  msg.sender = "adaptive_cruise_controller";
  msg.command = true;
  return msg;
}

void AdaptiveCruiseControllerNode::createPolygonFromTrajectoryPoints(
  const TrajectoryPoints & trajectory_points, const double expand_width,
  std::vector<Polygon2d> & polygons)
{
  std::vector<cv::Point2d> vehicle_corner_points;

  const auto & i = vehicle_info_;
  const auto & front_m = i.max_longitudinal_offset_m;
  const auto & width_m = i.vehicle_width_m / 2.0 + expand_width;
  const auto & back_m = i.rear_overhang_m;

  for (size_t i = 0; i < trajectory_points.size() - 1; i++) {
    std::vector<cv::Point2d> polygon;
    vehicle_corner_points.clear();
    // create convex hull polygon from trajectory points i and i+1
    for (size_t j = i; j <= i + 1; j++) {
      const auto point = trajectory_points.at(i);
      const auto yaw = tf2::getYaw(point.pose.orientation);
      vehicle_corner_points.push_back(cv::Point2d(
        point.pose.position.x + std::cos(yaw) * front_m - std::sin(yaw) * width_m,
        point.pose.position.y + std::sin(yaw) * front_m + std::cos(yaw) * width_m));
      vehicle_corner_points.push_back(cv::Point2d(
        point.pose.position.x + std::cos(yaw) * front_m - std::sin(yaw) * -width_m,
        point.pose.position.y + std::sin(yaw) * front_m + std::cos(yaw) * -width_m));
      vehicle_corner_points.push_back(cv::Point2d(
        point.pose.position.x + std::cos(yaw) * -back_m - std::sin(yaw) * -width_m,
        point.pose.position.y + std::sin(yaw) * -back_m + std::cos(yaw) * -width_m));
      vehicle_corner_points.push_back(cv::Point2d(
        point.pose.position.x + std::cos(yaw) * -back_m - std::sin(yaw) * width_m,
        point.pose.position.y + std::sin(yaw) * -back_m + std::cos(yaw) * width_m));
    }
    convexHull(vehicle_corner_points, polygon);
    // convert cv-polygon to boost-polygon
    Polygon2d boost_polygon;
    convertcvPointsToBoostPolygon(polygon, boost_polygon);
    polygons.emplace_back(boost_polygon);
  }
}

void AdaptiveCruiseControllerNode::extractVehicleObjectsInPolygon(
  const PredictedObjects & objects, const std::vector<Polygon2d> & target_polygons,
  PredictedObjects & objects_in_polygon)
{
  objects_in_polygon.header = objects.header;
  for (const auto object : objects.objects) {
    if (!isTargetObjectType(object)) {
      // the object is not target vehicle type
      continue;
    }

    Polygon2d object_polygon;
    convertObjectToBoostPolygon(object, object_polygon);
    if (isObjectWithinPolygon(target_polygons, object_polygon)) {
      objects_in_polygon.objects.emplace_back(object);
    }
  }
}

void AdaptiveCruiseControllerNode::getTargetObjects(
  const PredictedObjects & objects, const std::vector<Polygon2d> & target_polygons,
  const TrajectoryPoints & target_trajectory, PredictedObjects & target_objects)
{
  target_objects.header = objects.header;

  for (const auto object : objects.objects) {
    // check if the object is in the target area
    // and if the object angle is aligned with the trajectory angle.
    Polygon2d object_polygon;
    convertObjectToBoostPolygon(object, object_polygon);
    bool object_in_target_area = isObjectWithinPolygon(target_polygons, object_polygon);
    bool align_object_angle = isAngleAlignedWithTrajectory(
      getObjectPose(object), target_trajectory, param_.object_threshold_angle);

    if (object_in_target_area && align_object_angle) {
      target_objects.objects.emplace_back(object);
      continue;
    }

    // check if the object velocity is high
    // and if the predicted_path and trajectory overlap for a certain period of time.
    bool object_velocity_high = isObjectVelocityHigh(object);
    double threshold_dist_of_path_and_trajectory = vehicle_info_.vehicle_width_m / 2.0 +
                                                   object.shape.dimensions.y / 2.0 +
                                                   param_.detection_area_expand_length;
    bool overlap_predicted_path_with_trajectory = isPathNearTrajectory(
      getHighestConfidencePathFromObject(object), target_trajectory,
      threshold_dist_of_path_and_trajectory, param_.predicted_path_threshold_angle,
      param_.mininum_overlap_time_of_predicted_path);

    if (object_velocity_high && overlap_predicted_path_with_trajectory) {
      target_objects.objects.emplace_back(object);
      continue;
    }
  }
}

void AdaptiveCruiseControllerNode::getNearestObject(
  const PredictedObjects & objects, const TrajectoryPoints & trajectory_points,
  PredictedObject & nearest_object)
{
  double minimum_length_to_object = std::numeric_limits<double>::max();

  for (const auto object : objects.objects) {
    const auto object_point = object.kinematics.initial_pose_with_covariance.pose.position;

    const auto length_to_object = tier4_autoware_utils::calcSignedArcLength(
      trajectory_points, current_pose_->pose.position, object_point);

    if (length_to_object < minimum_length_to_object) {
      minimum_length_to_object = length_to_object;
      nearest_object = object;
    }
  }
}

PredictedPath AdaptiveCruiseControllerNode::getHighestConfidencePathFromObject(
  const PredictedObject & object)
{
  double highest_confidence = 0;
  PredictedPath highest_confidence_path;

  for (const auto path : object.kinematics.predicted_paths) {
    if (highest_confidence < path.confidence) {
      highest_confidence = path.confidence;
      highest_confidence_path = path;
    }
  }
  return highest_confidence_path;
}

geometry_msgs::msg::Pose AdaptiveCruiseControllerNode::getObjectPose(const PredictedObject & object)
{
  const double object_velocity = object.kinematics.initial_twist_with_covariance.twist.linear.x;
  auto obj_pose = object.kinematics.initial_pose_with_covariance.pose;
  double obj_yaw, obj_pitch, obj_roll;
  tf2::getEulerYPR(obj_pose.orientation, obj_yaw, obj_pitch, obj_roll);

  // If the object is stopped, align object orientation to ego vehicle
  if (std::fabs(object_velocity) < acc_param_.object_stop_velocity_thresh) {
    const auto ego_yaw = tf2::getYaw(current_pose_->pose.orientation);
    const auto diff_yaw = tier4_autoware_utils::normalizeRadian(ego_yaw - obj_yaw);
    if (std::fabs(diff_yaw) < M_PI / 2.0) {
      return obj_pose;
    }
    // If the pose of the object and ego-vehicle are in opposite directions,
    // invert yaw-angle
    tf2::Quaternion inv_q;
    inv_q.setRPY(obj_roll, obj_pitch, obj_yaw + M_PI);
    obj_pose.orientation = tf2::toMsg(inv_q);
    return obj_pose;
  }

  if (object_velocity >= 0) {
    return object.kinematics.initial_pose_with_covariance.pose;
  }

  // If the object velocity is negative, invert yaw-angle
  tf2::Quaternion inv_q;
  inv_q.setRPY(obj_roll, obj_pitch, obj_yaw + M_PI);
  obj_pose.orientation = tf2::toMsg(inv_q);
  return obj_pose;
}

bool AdaptiveCruiseControllerNode::isTargetObjectType(const PredictedObject & object)
{
  using Label = ObjectClassification;

  const auto object_label = object.classification.front().label;
  if (
    object_label == Label::BUS || object_label == Label::CAR || object_label == Label::TRAILER ||
    object_label == Label::MOTORCYCLE || object_label == Label::TRUCK) {
    return true;
  }
  return false;
}

bool AdaptiveCruiseControllerNode::isObjectVelocityHigh(const PredictedObject & object)
{
  const auto object_velocity = object.kinematics.initial_twist_with_covariance.twist.linear.x;
  return object_velocity > acc_param_.object_low_velocity_thresh;
}

/* only for debug */

void AdaptiveCruiseControllerNode::publishDebugOutputWithNoTarget()
{
  resetObjectTwistHistory();
  if (prev_acc_info_) {
    debug_node_ptr_->setDebugValues(DebugValues::TYPE::FLAG_CUTOUT_OBJECT, 1.0);
  }
  prev_acc_info_.reset();
  debug_node_ptr_->clearMarker();
  const auto ego_twist_stamped =
    toTwistStamped(current_odometry_ptr_->header, current_odometry_ptr_->twist.twist);
  ego_accel_ = calcAcc(ego_twist_stamped, prev_ego_twist_, ego_accel_);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::CURRENT_VEL, current_odometry_ptr_->twist.twist.linear.x);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::CURRENT_ACC, ego_accel_);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::FLAG_NONE, 1.0);
  debug_node_ptr_->publish();
}

void AdaptiveCruiseControllerNode::fillAndPublishDebugOutput(const PredictedObject & target_object)
{
  const auto acc_info_ptr = controller_ptr_->getAccInfoPtr();
  const auto prev_acc_info_ptr = controller_ptr_->getPrevAccInfoPtr();
  const auto acc_motion = controller_ptr_->getAccMotion();
  const auto acc_state = controller_ptr_->getState();

  // correct object twist
  auto target_object_abs_twist = target_object;
  auto & obj_twist =
    target_object_abs_twist.kinematics.initial_twist_with_covariance.twist.linear.x;
  obj_twist = std::fabs(obj_twist);

  // calculate ego-vehicle acceleration
  const auto ego_twist_stamped =
    toTwistStamped(current_odometry_ptr_->header, current_odometry_ptr_->twist.twist);
  ego_accel_ = calcAcc(ego_twist_stamped, prev_ego_twist_, ego_accel_);

  // calculate target-object acceleration
  const auto obj_twist_stamped = toTwistStamped(
    current_objects_ptr_->header,
    target_object_abs_twist.kinematics.initial_twist_with_covariance.twist);

  const auto cut_in_out = !acc_info_ptr ? CUT_IN_OUT::NONE : detectCutInAndOut(*acc_info_ptr);

  if (cut_in_out == CUT_IN_OUT::NONE) {
    obj_accel_ = calcAcc(obj_twist_stamped, prev_object_twist_, obj_accel_);
  } else {
    prev_object_twist_.reset();
    obj_accel_ = 0.0;
  }

  // calculate target-object velocity by differential of distance to object from ego-vehicle
  double object_vel_by_diff_target_dist = 0.0;
  if (acc_info_ptr && prev_acc_info_ptr) {
    if (cut_in_out == CUT_IN_OUT::NONE) {
      const double diff_dist =
        acc_info_ptr->current_distance_to_object - prev_acc_info_ptr->current_distance_to_object;
      const double dt = (acc_info_ptr->info_time - prev_acc_info_ptr->info_time).seconds();
      // calculate the speed of change of target distance
      object_vel_by_diff_target_dist = dt > 0 ? diff_dist / dt : 0.0;
      // remove the effect of own vehicle speed
      object_vel_by_diff_target_dist += current_odometry_ptr_->twist.twist.linear.x;
    }
  }

  const auto target_velocity = acc_state == State::STOP ? 0.0 : acc_motion.target_velocity;

  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::CURRENT_VEL, current_odometry_ptr_->twist.twist.linear.x);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::CURRENT_ACC, ego_accel_);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::CURRENT_OBJECT_VEL, acc_info_ptr->current_object_velocity);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::CURRENT_OBJECT_VEL_DIFF_DIST, object_vel_by_diff_target_dist);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::CURRENT_OBJECT_ACC, obj_accel_);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::CURRENT_OBJECT_DISTANCE, acc_info_ptr->current_distance_to_object);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::IDEAL_OBJECT_DISTANCE, acc_info_ptr->ideal_distance_to_object);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::FLAG_CUTIN_OBJECT, cut_in_out == CUT_IN_OUT::CUT_IN);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::FLAG_CUTOUT_OBJECT, cut_in_out == CUT_IN_OUT::CUT_OUT);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::FLAG_FIND_OBJECT, 1.0);
  debug_node_ptr_->setDebugValues(
    DebugValues::TYPE::FLAG_ADAPTIVE_CRUISE, (acc_state == State::ACC));
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::FLAG_STOP, (acc_state == State::STOP));
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::FLAG_NONE, 0.0);
  debug_node_ptr_->setDebugValues(DebugValues::TYPE::CURRENT_TARGET_VELOCITY, target_velocity);

  debug_node_ptr_->clearMarker();
  debug_node_ptr_->setTargetPolygon(target_object);
  if (acc_state == State::ACC) {
    const auto target_object_pose = target_object.kinematics.initial_pose_with_covariance.pose;
    const auto closet_pose =
      tier4_autoware_utils::findNearestIndex(acc_info_ptr->original_trajectory, target_object_pose);
    if (closet_pose) {
      const auto offset =
        acc_param_.minimum_margin_distance + target_object.shape.dimensions.x / 2.0;
      debug_node_ptr_->setVirtualWall(
        acc_info_ptr->original_trajectory.at(*closet_pose).pose, -offset, false);
    }

  } else if (acc_state == State::STOP) {
    const auto stop_pose = controller_ptr_->getAccMotion().stop_pose;
    debug_node_ptr_->setVirtualWall(stop_pose, vehicle_info_.max_longitudinal_offset_m, true);
  }
  debug_node_ptr_->publish();
}

double AdaptiveCruiseControllerNode::calcAcc(
  const geometry_msgs::msg::TwistStamped & twist,
  std::shared_ptr<geometry_msgs::msg::TwistStamped> & prev_twist, const double prev_acc,
  const double lowpass_gain, const double timeout)
{
  if (!prev_twist) {
    prev_twist = std::make_shared<geometry_msgs::msg::TwistStamped>(twist);
    return 0.0;
  }

  const double dt =
    (rclcpp::Time(twist.header.stamp) - rclcpp::Time(prev_twist->header.stamp)).seconds();
  const double dv = twist.twist.linear.x - prev_twist->twist.linear.x;
  prev_twist = std::make_shared<geometry_msgs::msg::TwistStamped>(twist);

  if (dt > timeout) {
    return 0.0;
  }

  if (dt <= 0.0) {
    return prev_acc;
  }

  const double acc = dv / dt;
  return prev_acc * lowpass_gain + acc * (1.0 - lowpass_gain);
}

void AdaptiveCruiseControllerNode::resetObjectTwistHistory()
{
  prev_object_twist_.reset();
  obj_accel_ = 0.0;
}

geometry_msgs::msg::TwistStamped AdaptiveCruiseControllerNode::toTwistStamped(
  const std_msgs::msg::Header & header, const geometry_msgs::msg::Twist & twist)
{
  geometry_msgs::msg::TwistStamped twist_stamped;
  twist_stamped.header = header;
  twist_stamped.twist = twist;
  return twist_stamped;
}

CUT_IN_OUT AdaptiveCruiseControllerNode::detectCutInAndOut(
  AdaptiveCruiseInformation acc_info, const double threshold_length)
{
  if (!prev_acc_info_) {
    prev_acc_info_ = std::make_shared<AdaptiveCruiseInformation>(acc_info);
    return CUT_IN_OUT::CUT_IN;
  }

  const double d_dist =
    acc_info.current_distance_to_object - prev_acc_info_->current_distance_to_object;
  prev_acc_info_ = std::make_shared<AdaptiveCruiseInformation>(acc_info);

  const double dt = (this->now() - prev_acc_info_->info_time).seconds();
  if (dt > acc_param_.reset_time_to_acc_state) {
    // previous acc info is too old.
    return CUT_IN_OUT::CUT_IN;
  }

  if (d_dist > threshold_length) return CUT_IN_OUT::CUT_OUT;
  if (d_dist < -threshold_length) return CUT_IN_OUT::CUT_IN;
  return CUT_IN_OUT::NONE;
}

}  // namespace motion_planning

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(motion_planning::AdaptiveCruiseControllerNode)
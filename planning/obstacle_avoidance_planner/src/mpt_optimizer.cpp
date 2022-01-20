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

#include "obstacle_avoidance_planner/mpt_optimizer.hpp"

#include "obstacle_avoidance_planner/utils.hpp"
#include "obstacle_avoidance_planner/vehicle_model/vehicle_model_bicycle_kinematics.hpp"
#include "tf2/utils.h"

#include "nav_msgs/msg/map_meta_data.hpp"

#include "boost/optional.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <tuple>
#include <vector>

namespace
{
geometry_msgs::msg::Pose convertRefPointsToPose(const ReferencePoint & ref_point)
{
  geometry_msgs::msg::Pose pose;
  pose.position = ref_point.p;
  pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(ref_point.yaw);
  return pose;
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> extractBounds(
  const std::vector<ReferencePoint> & ref_points, const size_t l_idx)
{
  Eigen::VectorXd ub_vec(ref_points.size());
  Eigen::VectorXd lb_vec(ref_points.size());
  for (size_t i = 0; i < ref_points.size(); ++i) {
    ub_vec(i) = ref_points.at(i).vehicle_bounds.at(l_idx).upper_bound;
    lb_vec(i) = ref_points.at(i).vehicle_bounds.at(l_idx).lower_bound;
  }
  return {ub_vec, lb_vec};
}

Bounds findWidestBounds(const BoundsCandidates & front_bounds_candidates)
{
  double max_width = std::numeric_limits<double>::min();
  size_t max_width_index = 0;
  if (front_bounds_candidates.size() != 1) {
    for (size_t candidate_idx = 0; candidate_idx < front_bounds_candidates.size();
         ++candidate_idx) {
      const auto & front_bounds_candidate = front_bounds_candidates.at(candidate_idx);
      const double bound_width =
        front_bounds_candidate.upper_bound - front_bounds_candidate.lower_bound;
      if (max_width < bound_width) {
        max_width_index = candidate_idx;
        max_width = bound_width;
      }
    }
  }
  return front_bounds_candidates.at(max_width_index);
}

double calcOverlappedBounds(
  const geometry_msgs::msg::Pose & front_point, const Bounds & front_bounds_candidate,
  const geometry_msgs::msg::Pose & prev_front_point, const Bounds & prev_front_continuous_bounds)
{
  const double avoiding_yaw =
    tier4_autoware_utils::normalizeRadian(tf2::getYaw(front_point.orientation) + M_PI_2);

  geometry_msgs::msg::Point ub_pos;
  ub_pos.x = front_point.position.x + front_bounds_candidate.upper_bound * std::cos(avoiding_yaw);
  ub_pos.y = front_point.position.y + front_bounds_candidate.upper_bound * std::sin(avoiding_yaw);

  geometry_msgs::msg::Point lb_pos;
  lb_pos.x = front_point.position.x + front_bounds_candidate.lower_bound * std::cos(avoiding_yaw);
  lb_pos.y = front_point.position.y + front_bounds_candidate.lower_bound * std::sin(avoiding_yaw);

  const double projected_ub_y =
    geometry_utils::transformToRelativeCoordinate2D(ub_pos, prev_front_point).y;
  const double projected_lb_y =
    geometry_utils::transformToRelativeCoordinate2D(lb_pos, prev_front_point).y;

  const double min_ub = std::min(projected_ub_y, prev_front_continuous_bounds.upper_bound);
  const double max_lb = std::max(projected_lb_y, prev_front_continuous_bounds.lower_bound);

  const double overlapped_length = min_ub - max_lb;
  return overlapped_length;
}

geometry_msgs::msg::Pose calcVehiclePose(
  const ReferencePoint & ref_point, const double lat_error, const double yaw_error,
  const double offset)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = ref_point.p.x - std::sin(ref_point.yaw) * lat_error -
                    std::cos(ref_point.yaw + yaw_error) * offset;
  pose.position.y = ref_point.p.y + std::cos(ref_point.yaw) * lat_error -
                    std::sin(ref_point.yaw + yaw_error) * offset;
  pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(ref_point.yaw + yaw_error);

  return pose;
}

template <typename T>
void trimPoints(std::vector<T> & points)
{
  std::vector<T> trimmed_points;
  constexpr double epsilon = 1e-6;

  auto itr = points.begin();
  while (itr != points.end() - 1) {
    bool is_overlapping = false;
    if (itr != points.begin()) {
      const auto & p_front = tier4_autoware_utils::getPoint(*itr);
      const auto & p_back = tier4_autoware_utils::getPoint(*(itr + 1));

      const double dx = p_front.x - p_back.x;
      const double dy = p_front.y - p_back.y;
      if (dx * dx + dy * dy < epsilon) {
        is_overlapping = true;
      }
    }
    if (is_overlapping) {
      itr = points.erase(itr);
    } else {
      ++itr;
    }
  }
}

/*
MPTOptimizer::MPTMatrix translateMPTMatrix(const MPTOptimizer::MPTMatrix & mpt_mat, const
std::vector<ReferencePoint> ref_points, const double offset, const size_t D_x)
{
  const size_t T_rows = mpt_mat.B.rows();

  // generate T_mat and T_vec to shift a vector
  //   T_mat(X) + T_vec = T_mat * (Bex * U + Wex) + T_vec
  //                    = T_mat * Bex U + T_mat * Wex + T_vec
  Eigen::SparseMatrix<double> T_mat(T_rows, T_rows);
  Eigen::VectorXd T_vec = Eigen::VectorXd::Zero(T_rows);
  std::vector<Eigen::Triplet<double>> triplet_T;

  for (size_t i = 0; i < ref_points.size(); ++i) {
    const double alpha = ref_points.at(i).alpha;

    triplet_T.push_back(Eigen::Triplet<double>(i * D_x, i * D_x, std::cos(alpha)));
    triplet_T.push_back(Eigen::Triplet<double>(i * D_x, i * D_x + 1, offset * std::cos(alpha)));
    triplet_T.push_back(Eigen::Triplet<double>(i * D_x + 1, i * D_x + 1, 1.0));

    T_vec(i * D_x) = -offset * std::sin(alpha);
  }
  T_mat.setFromTriplets(triplet_T.begin(), triplet_T.end());

  MPTOptimizer::MPTMatrix res_mpt_mat;
  res_mpt_mat.B = T_mat * mpt_mat.B
  res_mpt_mat.W = T_mat * mpt_mat.Wex + T_vec;
  return res_mpt_mat;
}
*/

std::vector<double> eigenVectorToStdVector(const Eigen::VectorXd & eigen_vec) {
  return {eigen_vec.data(), eigen_vec.data() + eigen_vec.rows()};
}
}  // namespace

MPTOptimizer::MPTOptimizer(
  const bool is_showing_debug_info, const TrajectoryParam & traj_param,
  const VehicleParam & vehicle_param, const MPTParam & mpt_param)
: is_showing_debug_info_(is_showing_debug_info),
  osqp_solver_ptr_(std::make_unique<autoware::common::osqp::OSQPInterface>(1.0e-3))
{
  traj_param_ptr_ = std::make_unique<TrajectoryParam>(traj_param);
  vehicle_param_ptr_ = std::make_unique<VehicleParam>(vehicle_param);
  mpt_param_ptr_ = std::make_unique<MPTParam>(mpt_param);

  vehicle_model_ptr_ = std::make_unique<KinematicsBicycleModel>(
    vehicle_param_ptr_->wheelbase, mpt_param_ptr_->steer_tau, mpt_param_ptr_->max_steer_rad);
}

boost::optional<MPTOptimizer::MPTTrajs> MPTOptimizer::getModelPredictiveTrajectory(
  const bool enable_avoidance,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & smoothed_points,
  const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
  const std::unique_ptr<Trajectories> & prev_trajs, const CVMaps & maps,
  std::shared_ptr<DebugData> debug_data_ptr)
{
  stop_watch_.tic(__func__);
  if (smoothed_points.empty()) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "return boost::none since smoothed_points is empty");
    return boost::none;
  }
  geometry_msgs::msg::Pose begin_smoothed_point = smoothed_points.front().pose;
  // if (prev_trajs && prev_trajs->model_predictive_trajectory.size() > 0) {
  if (prev_trajs) {
    const size_t prev_nearest_idx = tier4_autoware_utils::findNearestIndex(
      prev_trajs->model_predictive_trajectory, smoothed_points.front().pose.position);
    begin_smoothed_point = prev_trajs->model_predictive_trajectory.at(prev_nearest_idx).pose;
  }

  // NOTE: coordinate of smoothed_points is base center
  //       coordinate of ref_points is offset forward from base link
  std::vector<ReferencePoint> full_ref_points = getReferencePoints(
    begin_smoothed_point, smoothed_points, prev_trajs, enable_avoidance, maps, debug_data_ptr);
  if (full_ref_points.empty()) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "return boost::none since ref_points is empty");
    return boost::none;
  } else if (full_ref_points.size() == 1) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "return boost::none since ref_points.size() == 1");
    return boost::none;
  }

  debug_data_ptr->mpt_fixed_traj = getMPTFixedPoints(full_ref_points);

  std::vector<ReferencePoint> fixed_ref_points;
  std::vector<ReferencePoint> non_fixed_ref_points;
  bool fix = true;
  for (size_t i = 0; i < full_ref_points.size(); ++i) {
    if (i == full_ref_points.size() - 1) {
      if (full_ref_points.at(i).fix_kinematics) {
      } else {
        fix = false;
      }
    } else if (full_ref_points.at(i).fix_kinematics && full_ref_points.at(i + 1).fix_kinematics) {
      // } else if (full_ref_points.at(i).fix_kinematics) {
    } else {
      fix = false;
    }

    if (fix) {
      fixed_ref_points.push_back(full_ref_points.at(i));
      // RCLCPP_ERROR_STREAM(rclcpp::get_logger("fix"), i << " " << full_ref_points.at(i).s);
    } else {
      non_fixed_ref_points.push_back(full_ref_points.at(i));
      // RCLCPP_ERROR_STREAM(rclcpp::get_logger("non_fix"), i << " " << full_ref_points.at(i).s);
    }
  }

  // calculate B and W matrices
  const auto mpt_matrix = generateMPTMatrix(non_fixed_ref_points, prev_trajs, debug_data_ptr);

  // calculate Q and R matrices
  const auto val_matrix =
    generateValueMatrix(non_fixed_ref_points, path_points.back(), debug_data_ptr);

  const auto optimized_control_variables = executeOptimization(prev_trajs,
    enable_avoidance, mpt_matrix, val_matrix, non_fixed_ref_points, debug_data_ptr);
  if (!optimized_control_variables) {
    RCLCPP_INFO_EXPRESSION(
      rclcpp::get_logger("obstacle_avoidance_planner.time"), is_showing_debug_info_,
      "return boost::none since could not solve qp");
    return boost::none;
  }

  const auto mpt_points = getMPTPoints(
    fixed_ref_points, non_fixed_ref_points, optimized_control_variables.get(), mpt_matrix,
    debug_data_ptr);

  auto full_optimized_ref_points = fixed_ref_points;
  full_optimized_ref_points.insert(
    full_optimized_ref_points.end(), non_fixed_ref_points.begin(), non_fixed_ref_points.end());

  debug_data_ptr->msg_stream << "      " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";

  MPTTrajs mpt_trajs;
  mpt_trajs.mpt = mpt_points;
  mpt_trajs.ref_points = full_optimized_ref_points;
  return mpt_trajs;
}

std::vector<ReferencePoint> MPTOptimizer::getReferencePoints(
  [[maybe_unused]] const geometry_msgs::msg::Pose & begin_smoothed_point,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & points,
  [[maybe_unused]] const std::unique_ptr<Trajectories> & prev_trajs, const bool enable_avoidance,
  const CVMaps & maps, std::shared_ptr<DebugData> debug_data_ptr) const
{
  stop_watch_.tic(__func__);

  const auto ref_points = [&]() {
    // assing fixed kinematics
    auto ref_points = [&]() {
      // if plan from ego
      constexpr double epsilon = 1e-04;
      const bool is_planning_from_ego =
        mpt_param_ptr_->plan_from_ego && std::abs(current_vel_) < epsilon;
      if (is_planning_from_ego) {
        // interpolate and crop backward
        const auto interpolated_points = interpolation_utils::getInterpolatedPoints(
          points, mpt_param_ptr_->delta_arc_length_for_mpt_points);
        const auto cropped_interpolated_points = points_utils::clipBackwardPoints(
          interpolated_points, current_pose_.position, traj_param_ptr_->backward_fixing_distance,
          mpt_param_ptr_->delta_arc_length_for_mpt_points);

        auto cropped_ref_points =
          points_utils::convertToReferencePoints(cropped_interpolated_points);

        // assign fix kinematics
        const size_t nearest_ref_idx =
          tier4_autoware_utils::findNearestIndex(cropped_ref_points, current_pose_.position);
        cropped_ref_points.at(nearest_ref_idx).fix_kinematics =
          getState(current_pose_, cropped_ref_points.at(nearest_ref_idx));

        return cropped_ref_points;
      }

      const auto fixed_ref_points = getFixedReferencePoints(points, prev_trajs);

      // if no fixed_ref_points
      if (fixed_ref_points.empty()) {
        // interpolate and crop backward
        const auto interpolated_points = interpolation_utils::getInterpolatedPoints(
          points, mpt_param_ptr_->delta_arc_length_for_mpt_points);
        const auto cropped_interpolated_points = points_utils::clipBackwardPoints(
          interpolated_points, current_pose_.position, traj_param_ptr_->backward_fixing_distance,
          mpt_param_ptr_->delta_arc_length_for_mpt_points);

        return points_utils::convertToReferencePoints(cropped_interpolated_points);
      }

      // calc non fixed traj points
      const size_t seg_idx =
        tier4_autoware_utils::findNearestSegmentIndex(points, fixed_ref_points.back().p);
      const auto non_fixed_traj_points =
        std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint>{
          points.begin() + seg_idx, points.end()};

      const double offset = tier4_autoware_utils::calcLongitudinalOffsetToSegment(
                              non_fixed_traj_points, 0, fixed_ref_points.back().p) +
                            mpt_param_ptr_->delta_arc_length_for_mpt_points;
      const auto non_fixed_interpolated_traj_points = interpolation_utils::getInterpolatedPoints(
        non_fixed_traj_points, mpt_param_ptr_->delta_arc_length_for_mpt_points, offset);
      const auto non_fixed_ref_points =
        points_utils::convertToReferencePoints(non_fixed_interpolated_traj_points);

      // make ref points
      auto ref_points = fixed_ref_points;
      ref_points.insert(ref_points.end(), non_fixed_ref_points.begin(), non_fixed_ref_points.end());

      return ref_points;
    }();

    if (ref_points.empty()) {
      return std::vector<ReferencePoint>{};
    }

    // set some information to reference points considering fix kinematics
    trimPoints(ref_points);
    calcOrientation(ref_points);
    calcVelocity(ref_points, points);
    calcCurvature(ref_points);
    calcArcLength(ref_points);

    // crop trajectory with margin to calculate vehicle bounds at the end point
    const double ref_length_with_margin =
      traj_param_ptr_->num_sampling_points * mpt_param_ptr_->delta_arc_length_for_mpt_points +
      3.0;  // TODO(murooka) magic number
    ref_points = points_utils::clipForwardPoints(ref_points, 0, ref_length_with_margin);

    // set bounds information
    calcBounds(ref_points, enable_avoidance, maps, debug_data_ptr);
    calcVehicleBounds(ref_points, maps, debug_data_ptr, enable_avoidance);

    // set extra information (alpha and has_object_collision)
    // NOTE: This must be after bounds calculation.
    calcExtraPoints(ref_points);

    const double ref_length =
      traj_param_ptr_->num_sampling_points * mpt_param_ptr_->delta_arc_length_for_mpt_points;
    ref_points = points_utils::clipForwardPoints(ref_points, 0, ref_length);

    // bounds information is assigned to debug data after truncating reference points
    debug_data_ptr->ref_points = ref_points;

    return ref_points;
  }();
  if (ref_points.empty()) {
    return std::vector<ReferencePoint>{};
  }

  // TODO(murooka) think about it later
  /*
  // crop ref_points so that ref_points starts from around begin_smoothed_point
  const size_t begin_idx =
    tier4_autoware_utils::findNearestIndex(ref_points, begin_smoothed_point.position);
  const double ref_length =
    traj_param_ptr_->num_sampling_points * traj_param_ptr_->delta_arc_length_for_mpt_points;
  auto truncated_points = points_utils::clipForwardPoints(ref_points, begin_idx, ref_length);
  */

  debug_data_ptr->msg_stream << "        " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
  return ref_points;
}

std::vector<ReferencePoint> MPTOptimizer::getFixedReferencePoints(
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & points,
  const std::unique_ptr<Trajectories> & prev_trajs) const
{
  if (
    !prev_trajs ||
    prev_trajs->model_predictive_trajectory.size() != prev_trajs->mpt_ref_points.size()) {
    return std::vector<ReferencePoint>();
  }

  if (!mpt_param_ptr_->fix_points_around_ego) {
    return std::vector<ReferencePoint>();
  }

  const auto & prev_ref_points = prev_trajs->mpt_ref_points;
  const int nearest_prev_ref_idx =
    tier4_autoware_utils::findNearestIndex(prev_ref_points, current_pose_.position);

  // calculate begin_prev_ref_idx
  const int begin_prev_ref_idx = [&]() {
    const int backward_fixing_num =
      traj_param_ptr_->backward_fixing_distance / mpt_param_ptr_->delta_arc_length_for_mpt_points;

    return std::max(0, nearest_prev_ref_idx - backward_fixing_num);
  }();

  // calculate end_prev_ref_idx
  const int end_prev_ref_idx = [&]() {
    const double forward_fixed_length = std::max(
      current_vel_ * mpt_param_ptr_->forward_fixing_mpt_time,
      mpt_param_ptr_->forward_fixing_mpt_min_distance);

    const int forward_fixing_num =
      forward_fixed_length / mpt_param_ptr_->delta_arc_length_for_mpt_points;
    return std::min(
      static_cast<int>(prev_ref_points.size()) - 1, nearest_prev_ref_idx + forward_fixing_num);
  }();

  bool flag = false;
  std::vector<ReferencePoint> fixed_ref_points;
  for (size_t i = begin_prev_ref_idx; i <= static_cast<size_t>(end_prev_ref_idx); ++i) {
    const auto & prev_ref_point = prev_ref_points.at(i);

    if (!flag) {
      if (tier4_autoware_utils::calcSignedArcLength(points, 0, prev_ref_point.p) < 0) {
        continue;
      }
      flag = true;
    }

    const double lat_distance = tier4_autoware_utils::calcLateralOffset(points, prev_ref_point.p);
    if (std::abs(lat_distance) < 0.05) {
      ReferencePoint fixed_ref_point;
      fixed_ref_point = prev_ref_point;
      fixed_ref_point.fix_kinematics = prev_ref_point.optimized_kinematics;

      /*
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("fix_kinematics"), i << " " << fixed_ref_point.fix_kinematics.get()(0) <<
      " "
                                           << fixed_ref_point.fix_kinematics.get()(1));
      */

      fixed_ref_points.push_back(fixed_ref_point);
    } else {
      break;
    }
  }

  return fixed_ref_points;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> MPTOptimizer::getMPTFixedPoints(
  const std::vector<ReferencePoint> & ref_points) const
{
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> mpt_fixed_traj;
  for (size_t i = 0; i < ref_points.size(); ++i) {
    const auto & ref_point = ref_points.at(i);
    const bool is_fixed = (ref_point.fix_kinematics && i < ref_points.size() - 1) ||
                          (ref_point.fix_kinematics && i == ref_points.size() - 1 &&
                           mpt_param_ptr_->is_hard_fixing_terminal_point);

    if (is_fixed) {
      const double lat_error = ref_point.fix_kinematics.get()(0);
      const double yaw_error = ref_point.fix_kinematics.get()(1);

      autoware_auto_planning_msgs::msg::TrajectoryPoint fixed_traj_point;
      fixed_traj_point.pose = calcVehiclePose(ref_point, lat_error, yaw_error, 0.0);
      mpt_fixed_traj.push_back(fixed_traj_point);
    }
  }

  return mpt_fixed_traj;
}

// predict equation: x = Bex u + Wex (u includes x_0)
// cost function: J = x' Qex x + u' Rex u
MPTOptimizer::MPTMatrix MPTOptimizer::generateMPTMatrix(
  const std::vector<ReferencePoint> & ref_points,
  [[maybe_unused]] const std::unique_ptr<Trajectories> & prev_trajs,
  std::shared_ptr<DebugData> debug_data_ptr) const
{
  if (ref_points.empty()) {
    return MPTMatrix{};
  }

  stop_watch_.tic(__func__);

  // TODO(murooka) we don't use now
  // vehicle_model_ptr_->updateCenterOffset(0.0);

  const size_t N_ref = ref_points.size();
  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t D_v = D_x + D_u * (N_ref - 1);

  Eigen::MatrixXd Bex = Eigen::MatrixXd::Zero(D_x * N_ref, D_v);
  Eigen::VectorXd Wex = Eigen::VectorXd::Zero(D_x * N_ref);

  Eigen::MatrixXd Ad(D_x, D_x);
  Eigen::MatrixXd Bd(D_x, D_u);
  Eigen::MatrixXd Wd(D_x, 1);

  // predict kinematics for N_ref times
  for (size_t i = 0; i < N_ref; ++i) {
    if (i == 0) {
      Bex.block(0, 0, D_x, D_x) = Eigen::MatrixXd::Identity(D_x, D_x);
      continue;
    }

    const int idx_x_i = i * D_x;
    const int idx_x_i_prev = (i - 1) * D_x;
    const int idx_u_i_prev = (i - 1) * D_u;

    const double ds = [&]() {
      if (N_ref == 1) {
        // TODO(murooka)
        return 0.0;
      }
      const size_t prev_idx = (i < N_ref - 1) ? i + 1 : i;
      return ref_points.at(prev_idx).s - ref_points.at(prev_idx - 1).s;
    }();

    // get discrete kinematics matrix A, B, W
    const double ref_k = ref_points.at(std::max(0, static_cast<int>(i) - 1)).k;
    vehicle_model_ptr_->setCurvature(ref_k);
    vehicle_model_ptr_->calculateStateEquationMatrix(Ad, Bd, Wd, ds);

    Bex.block(idx_x_i, 0, D_x, D_x) = Ad * Bex.block(idx_x_i_prev, 0, D_x, D_x);
    Bex.block(idx_x_i, D_x + idx_u_i_prev, D_x, D_u) = Bd;

    for (size_t j = 0; j < i - 1; ++j) {
      size_t idx_u_j = j * D_u;
      Bex.block(idx_x_i, D_x + idx_u_j, D_x, D_u) =
        Ad * Bex.block(idx_x_i_prev, D_x + idx_u_j, D_x, D_u);
    }

    Wex.segment(idx_x_i, D_x) = Ad * Wex.block(idx_x_i_prev, 0, D_x, 1) + Wd;
  }

  MPTMatrix m;
  m.Bex = Bex;
  m.Wex = Wex;

  /*
  if (m.Bex.array().isNaN().any() || m.Wex.array().isNaN().any()) {
    RCLCPP_WARN(
      rclcpp::get_logger(__func__),
      "[ObstacleAvoidance] MPT matrix includes NaN.");
    return boost::none;
  }
  */

  debug_data_ptr->msg_stream << "        " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
  return m;
}

MPTOptimizer::ValueMatrix MPTOptimizer::generateValueMatrix(
  const std::vector<ReferencePoint> & ref_points,
  const autoware_auto_planning_msgs::msg::PathPoint & last_path_pose,
  std::shared_ptr<DebugData> debug_data_ptr) const
{
  if (ref_points.empty()) {
    return ValueMatrix{};
  }

  stop_watch_.tic(__func__);

  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t N_ref = ref_points.size();

  const size_t D_v = D_x + (N_ref - 1) * D_u;

  geometry_msgs::msg::Pose last_ref_pose;
  last_ref_pose.position = ref_points.back().p;
  last_ref_pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(ref_points.back().yaw);
  const auto last_extended_point = points_utils::getLastExtendedPoint(
    last_path_pose, last_ref_pose, traj_param_ptr_->delta_yaw_threshold_for_closest_point,
    traj_param_ptr_->max_dist_for_extending_end_point);

  // update Q
  Eigen::SparseMatrix<double> Qex_sparse_mat(D_x * N_ref, D_x * N_ref);
  std::vector<Eigen::Triplet<double>> Qex_triplet_vec;
  for (size_t i = 0; i < N_ref; ++i) {
    const auto adaptive_error_weight = [&]() -> std::array<double, 2> {
      if (ref_points.at(i).near_objects) {
        return {
          mpt_param_ptr_->obstacle_avoid_lat_error_weight,
          mpt_param_ptr_->obstacle_avoid_yaw_error_weight};
      } else if (std::abs(ref_points[i].k) > 0.3) {  // TODO(murooka)
                                                     // return {0.0, 0.0};
      } else if (i == N_ref - 1 && last_extended_point) {
        return {
          mpt_param_ptr_->terminal_path_lat_error_weight,
          mpt_param_ptr_->terminal_path_yaw_error_weight};
      } else if (i == N_ref - 1) {
        return {
          mpt_param_ptr_->terminal_lat_error_weight, mpt_param_ptr_->terminal_yaw_error_weight};
      }
      return {mpt_param_ptr_->lat_error_weight, mpt_param_ptr_->yaw_error_weight};
    }();
    const double adaptive_lat_error_weight = adaptive_error_weight.at(0);
    const double adaptive_yaw_error_weight = adaptive_error_weight.at(1);

    Qex_triplet_vec.push_back(Eigen::Triplet<double>(i * D_x, i * D_x, adaptive_lat_error_weight));
    Qex_triplet_vec.push_back(
      Eigen::Triplet<double>(i * D_x + 1, i * D_x + 1, adaptive_yaw_error_weight));
  }
  Qex_sparse_mat.setFromTriplets(Qex_triplet_vec.begin(), Qex_triplet_vec.end());

  // update R
  Eigen::SparseMatrix<double> Rex_sparse_mat(D_v, D_v);
  std::vector<Eigen::Triplet<double>> Rex_triplet_vec;
  for (size_t i = 0; i < N_ref - 1; ++i) {
    const double adaptive_steer_weight = ref_points.at(i).near_objects
                                           ? mpt_param_ptr_->obstacle_avoid_steer_input_weight
                                           : mpt_param_ptr_->steer_input_weight;
    Rex_triplet_vec.push_back(
      Eigen::Triplet<double>(D_x + D_u * i, D_x + D_u * i, adaptive_steer_weight));
  }
  addSteerWeightR(Rex_triplet_vec, ref_points);

  Rex_sparse_mat.setFromTriplets(Rex_triplet_vec.begin(), Rex_triplet_vec.end());

  ValueMatrix m;
  m.Qex = Qex_sparse_mat;
  m.Rex = Rex_sparse_mat;

  debug_data_ptr->msg_stream << "        " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
  return m;
}

boost::optional<Eigen::VectorXd> MPTOptimizer::executeOptimization(
  [[maybe_unused]] const std::unique_ptr<Trajectories> & prev_trajs,
  const bool enable_avoidance, const MPTMatrix & mpt_mat, const ValueMatrix & val_mat,
  const std::vector<ReferencePoint> & ref_points, std::shared_ptr<DebugData> debug_data_ptr)
{
  if (ref_points.empty()) {
    return Eigen::VectorXd{};
  }

  stop_watch_.tic(__func__);

  const size_t N_ref = ref_points.size();

  // get matrix
  ObjectiveMatrix obj_m = getObjectiveMatrix(mpt_mat, val_mat, ref_points, debug_data_ptr);
  ConstraintMatrix const_m =
    getConstraintMatrix(enable_avoidance, mpt_mat, ref_points, debug_data_ptr);

  // manual warm start
  const bool manual_warm_start = false;
  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(obj_m.gradient.size());

  if (manual_warm_start) {
    const size_t D_x = vehicle_model_ptr_->getDimX();

    if (prev_trajs && prev_trajs->mpt_ref_points.size() > 1) {
      const size_t seg_idx = tier4_autoware_utils::findNearestSegmentIndex(prev_trajs->mpt_ref_points, ref_points.front().p);
      double offset = tier4_autoware_utils::calcLongitudinalOffsetToSegment(prev_trajs->mpt_ref_points, seg_idx, ref_points.front().p);

      u0(0) = prev_trajs->mpt_ref_points.at(seg_idx).optimized_kinematics(0);
      u0(1) = prev_trajs->mpt_ref_points.at(seg_idx).optimized_kinematics(1);

      for (size_t i = 0; i + 1 < N_ref; ++i) {
        size_t prev_idx = seg_idx + i;
        const size_t prev_N_ref = prev_trajs->mpt_ref_points.size();
        if (prev_idx + 2 > prev_N_ref) {
          prev_idx = static_cast<int>(prev_N_ref) - 2;
          offset = 0.5;
        }

        const double prev_val = prev_trajs->mpt_ref_points.at(prev_idx).optimized_input;
        const double next_val = prev_trajs->mpt_ref_points.at(prev_idx + 1).optimized_input;
        u0(D_x + i) = interpolation::lerp(prev_val, next_val, offset);
      }
    }
  }

  const Eigen::MatrixXd & H = obj_m.hessian;
  const Eigen::MatrixXd & A = const_m.linear;
  std::vector<double> f;
  std::vector<double> upper_bound;
  std::vector<double> lower_bound;
  if (manual_warm_start) {
    f = eigenVectorToStdVector(obj_m.gradient + H * u0);
    Eigen::VectorXd A_times_u0 = A * u0;
    upper_bound = eigenVectorToStdVector(const_m.upper_bound - A_times_u0);
    lower_bound = eigenVectorToStdVector(const_m.lower_bound - A_times_u0);
  } else {
    f = eigenVectorToStdVector(obj_m.gradient);
    upper_bound = eigenVectorToStdVector(const_m.upper_bound);
    lower_bound = eigenVectorToStdVector(const_m.lower_bound);
  }

  // initialize or update solver with warm start
  stop_watch_.tic("initOsqp");
  autoware::common::osqp::CSC_Matrix P_csc =
    autoware::common::osqp::calCSCMatrixTrapezoidal(H);
  autoware::common::osqp::CSC_Matrix A_csc = autoware::common::osqp::calCSCMatrix(A);
  if (prev_mat_n == H.rows() && prev_mat_m == A.rows()) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("obstacle_avoidance_planner.time"), "warm start");

    osqp_solver_ptr_->updateCscP(P_csc);
    osqp_solver_ptr_->updateQ(f);
    osqp_solver_ptr_->updateCscA(A_csc);
    osqp_solver_ptr_->updateL(lower_bound);
    osqp_solver_ptr_->updateU(upper_bound);
  } else {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("obstacle_avoidance_planner.time"), "no warm start");

    osqp_solver_ptr_ = std::make_unique<autoware::common::osqp::OSQPInterface>(
      // obj_m.hessian, const_m.linear, obj_m.gradient, const_m.lower_bound, const_m.upper_bound,
      P_csc, A_csc, f, lower_bound, upper_bound, 1.0e-3);
  }
  prev_mat_n = H.rows();
  prev_mat_m = A.rows();

  // osqp_solver_ptr_ = std::make_unique<autoware::common::osqp::OSQPInterface>(
  //     P_csc, A_csc, f, lower_bound, upper_bound, 1.0e-3);
  // osqp_solver_ptr_ = std::make_unique<autoware::common::osqp::OSQPInterface>(
  //   obj_m.hessian, const_m.linear, obj_m.gradient, const_m.lower_bound, const_m.upper_bound,
  //   1.0e-3);
  // osqp_solver_ptr_->updateEpsRel(1.0e-3);

  debug_data_ptr->msg_stream << "          "
                             << "initOsqp"
                             << ":= " << stop_watch_.toc("initOsqp") << " [ms]\n";

  // solve
  stop_watch_.tic("solveOsqp");
  const auto result = osqp_solver_ptr_->optimize();
  debug_data_ptr->msg_stream << "          "
                             << "solveOsqp"
                             << ":= " << stop_watch_.toc("solveOsqp") << " [ms]\n";

  // check solution status
  const int solution_status = std::get<3>(result);
  if (solution_status != 1) {
    utils::logOSQPSolutionStatus(solution_status);
    return boost::none;
  }

  // print iteration
  const int iteration_status = std::get<4>(result);
  RCLCPP_ERROR_STREAM(rclcpp::get_logger("iteration"), iteration_status);

  // get result
  std::vector<double> result_vec = std::get<0>(result);

  const size_t DIM_U = vehicle_model_ptr_->getDimU();
  const size_t DIM_X = vehicle_model_ptr_->getDimX();
  const Eigen::VectorXd optimized_control_variables =
    Eigen::Map<Eigen::VectorXd>(&result_vec[0], DIM_X + (N_ref - 1) * DIM_U);

  // TODO(murooka)
  // debug_data_ptr->msg_stream << "        " << __func__ <<
  //":= " << stop_watch_.toc(__func__)  << " [ms]\n";

  // RCLCPP_ERROR_STREAM(rclcpp::get_logger("norm"), optimized_control_variables.norm());

  const Eigen::VectorXd optimized_control_variables_with_offset = manual_warm_start ? optimized_control_variables + u0
    : optimized_control_variables;
  return optimized_control_variables_with_offset;
}


MPTOptimizer::ObjectiveMatrix MPTOptimizer::getObjectiveMatrix(
  const MPTMatrix & mpt_mat, const ValueMatrix & val_mat,
  [[maybe_unused]] const std::vector<ReferencePoint> & ref_points,
  std::shared_ptr<DebugData> debug_data_ptr) const
{
  stop_watch_.tic(__func__);

  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t N_ref = ref_points.size();

  const size_t D_xn = D_x * N_ref;
  const size_t D_v = D_x + (N_ref - 1) * D_u;

  // generate T matrix and vector to shift optimization center
  //   define Z as time-series vector of shifted deviation error
  //   Z = sparse_T_mat * (Bex * U + Wex) + T_vec
  Eigen::SparseMatrix<double> sparse_T_mat(D_xn, D_xn);
  Eigen::VectorXd T_vec = Eigen::VectorXd::Zero(D_xn);
  std::vector<Eigen::Triplet<double>> triplet_T_vec;
  const double offset = mpt_param_ptr_->optimization_center_offset;

  for (size_t i = 0; i < N_ref; ++i) {
    const double alpha = ref_points.at(i).alpha;

    triplet_T_vec.push_back(Eigen::Triplet<double>(i * D_x, i * D_x, std::cos(alpha)));
    triplet_T_vec.push_back(Eigen::Triplet<double>(i * D_x, i * D_x + 1, offset * std::cos(alpha)));
    triplet_T_vec.push_back(Eigen::Triplet<double>(i * D_x + 1, i * D_x + 1, 1.0));

    T_vec(i * D_x) = -offset * std::sin(alpha);
  }
  sparse_T_mat.setFromTriplets(triplet_T_vec.begin(), triplet_T_vec.end());

  const Eigen::MatrixXd B = sparse_T_mat * mpt_mat.Bex;
  const Eigen::MatrixXd QB = val_mat.Qex * B;
  const Eigen::MatrixXd R = val_mat.Rex;

  // min J(v) = min (v'Hv + v'f)
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(D_v, D_v);
  H.triangularView<Eigen::Upper>() = B.transpose() * QB + R;
  H.triangularView<Eigen::Lower>() = H.transpose();

  // Eigen::VectorXd f = ((sparse_T_mat * mpt_mat.Wex + T_vec).transpose() * QB).transpose();
  Eigen::VectorXd f = (sparse_T_mat * mpt_mat.Wex + T_vec).transpose() * QB;

  // addSteerWeightF(f);

  const size_t N_avoid = mpt_param_ptr_->avoiding_circle_offsets.size();
  const size_t N_first_slack = [&]() -> size_t {
    if (mpt_param_ptr_->soft_constraint) {
      if (mpt_param_ptr_->l_inf_norm) {
        return 1;
      }
      return N_avoid;
    }
    return 0;
  }();
  const size_t N_second_slack = [&]() -> size_t {
    if (mpt_param_ptr_->two_step_soft_constraint) {
      return N_first_slack;
    }
    return 0;
  }();

  // number of slack variables for one step
  const size_t N_slack = N_first_slack + N_second_slack;

  // extend H for slack variables
  Eigen::MatrixXd full_H = Eigen::MatrixXd::Zero(D_v + N_ref * N_slack, D_v + N_ref * N_slack);
  full_H.block(0, 0, D_v, D_v) = H;

  // extend f for slack variables
  Eigen::VectorXd full_f(D_v + N_ref * N_slack);
  //full_f.segment(0, D_v) = f;
  //full_f.segment(D_v, N_ref * N_first_slack) = mpt_param_ptr_->soft_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_first_slack);
  //full_f.segment(D_v + N_ref * N_first_slack, N_ref * N_second_slack) = mpt_param_ptr_->soft_second_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_second_slack);

  // full_f << f, mpt_param_ptr_->soft_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_first_slack),
  // mpt_param_ptr_->soft_second_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_second_slack);

   full_f.segment(0, D_v) = f;
   if (N_first_slack > 0) {
     full_f.segment(D_v, N_ref * N_first_slack) = mpt_param_ptr_->soft_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_first_slack);
   }
   if (N_second_slack > 0) {
     full_f.segment(D_v + N_ref * N_first_slack, N_ref * N_second_slack) = mpt_param_ptr_->soft_second_avoidance_weight * Eigen::VectorXd::Ones(N_ref * N_second_slack);
   }

  ObjectiveMatrix obj_matrix;
  obj_matrix.hessian = full_H;
  obj_matrix.gradient = full_f; // {full_f.data(), full_f.data() + full_f.rows()};

  debug_data_ptr->msg_stream << "          " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";

  return obj_matrix;
}

// Set constraint: lb <= Ax <= ub
// decision variable
// x := [u0, ..., uN-1 | z00, ..., z0N-1 | z10, ..., z1N-1 | z20, ..., z2N-1]
//   \in \mathbb{R}^{N * (N_vehicle_circle + 1)}
MPTOptimizer::ConstraintMatrix MPTOptimizer::getConstraintMatrix(
  [[maybe_unused]] const bool enable_avoidance, const MPTMatrix & mpt_mat,
  const std::vector<ReferencePoint> & ref_points, [[maybe_unused]] std::shared_ptr<DebugData> debug_data_ptr) const
{
  stop_watch_.tic(__func__);

  // NOTE: currently, add additional length to soft bounds approximately
  //       for soft second and hard bounds
  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t N_ref = ref_points.size();

  const size_t N_u = (N_ref - 1) * D_u;
  const size_t D_v = D_x + N_u;

  const size_t N_avoid = mpt_param_ptr_->avoiding_circle_offsets.size();

  // number of slack variables for one step
  const size_t N_first_slack = [&]() -> size_t {
    if (mpt_param_ptr_->soft_constraint) {
      if (mpt_param_ptr_->l_inf_norm) {
        return 1;
      }
      return N_avoid;
    }
    return 0;
  }();
  const size_t N_second_slack = [&]() -> size_t {
    if (mpt_param_ptr_->soft_constraint && mpt_param_ptr_->two_step_soft_constraint) {
      return N_first_slack;
    }
    return 0;
  }();

  // number of all slack variables is N_ref * N_slack
  const size_t N_slack = N_first_slack + N_second_slack;
  const size_t N_soft = mpt_param_ptr_->two_step_soft_constraint ? 2 : 1;

  const size_t A_cols = [&] {
    if (mpt_param_ptr_->soft_constraint) {
      return D_v + N_ref * N_slack;  // initial_state + steer + soft
    }
    return D_v;  // initial state + steer
  }();

  // calculate indices of fixed points
  std::vector<size_t> fixed_points_indices;
  for (size_t i = 0; i < N_ref; ++i) {
    if (ref_points.at(i).fix_kinematics) {
      // RCLCPP_ERROR_STREAM(rclcpp::get_logger("fixed"), i);
      fixed_points_indices.push_back(i);
    }
    //else if (
    //           // TODO(murooka)
    //  ref_points.at(i).fix_kinematics && i == ref_points.size() - 1 &&
    //  mpt_param_ptr_->is_hard_fixing_terminal_point) {
    //  fixed_points_indices.push_back(i);
    //}
  }

  // calculate rows of A
  size_t A_rows = 0;
  if (mpt_param_ptr_->soft_constraint) {
    // 3 means slack variable constraints to be between lower and upper bounds, and positive.
    A_rows += 3 * N_ref * N_avoid * N_soft;
  }
  if (mpt_param_ptr_->hard_constraint) {
    A_rows += N_ref * N_avoid;
  }
  A_rows += fixed_points_indices.size() * D_x;
  if (mpt_param_ptr_->steer_limit_constraint) {
    A_rows += N_u;
  }

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(A_rows, A_cols);
  Eigen::VectorXd lb = Eigen::VectorXd::Constant(A_rows, -autoware::common::osqp::INF);
  Eigen::VectorXd ub = Eigen::VectorXd::Constant(A_rows, autoware::common::osqp::INF);
  size_t A_rows_end = 0;

  // CX = C(Bv + w) + C \in R^{N_ref, N_ref * D_x}
  for (size_t l_idx = 0; l_idx < N_avoid; ++l_idx) {
    // create C := [1 | l | O]
    Eigen::SparseMatrix<double> C_sparse_mat(N_ref, N_ref * D_x);
    std::vector<Eigen::Triplet<double>> C_triplet_vec;
    Eigen::VectorXd C_vec = Eigen::VectorXd::Zero(N_ref);

    // calculate C mat and vec
    for (size_t i = 0; i < N_ref; ++i) {
      const double beta = ref_points.at(i).beta.at(l_idx).get();
      const double avoid_offset = mpt_param_ptr_->avoiding_circle_offsets.at(l_idx);

      C_triplet_vec.push_back(Eigen::Triplet<double>(i, i * D_x, 1.0 * std::cos(beta)));
      C_triplet_vec.push_back(
        Eigen::Triplet<double>(i, i * D_x + 1, avoid_offset * std::cos(beta)));
      C_vec(i) = avoid_offset * std::sin(beta);
    }
    C_sparse_mat.setFromTriplets(C_triplet_vec.begin(), C_triplet_vec.end());

    // calculate CB, and CW
    const Eigen::MatrixXd CB = C_sparse_mat * mpt_mat.Bex;
    const Eigen::VectorXd CW = C_sparse_mat * mpt_mat.Wex + C_vec;

    // calculate bounds
    const auto & [part_ub, part_lb] = extractBounds(ref_points, l_idx);

    // soft constraints
    if (mpt_param_ptr_->soft_constraint) {
      size_t A_offset_cols = D_v;
      for (size_t s_idx = 0; s_idx < N_soft; ++s_idx) {
        const size_t A_blk_rows = 3 * N_ref;

        // A := [C * Bex | O | ... | O | I | O | ...
        //      -C * Bex | O | ... | O | I | O | ...
        //          O    | O | ... | O | I | O | ... ]
        Eigen::MatrixXd A_blk = Eigen::MatrixXd::Zero(A_blk_rows, A_cols);
        A_blk.block(0, 0, N_ref, D_v) = CB;
        A_blk.block(N_ref, 0, N_ref, D_v) = -CB;

        size_t local_A_offset_cols = A_offset_cols;
        if (!mpt_param_ptr_->l_inf_norm) {
          local_A_offset_cols += N_ref * l_idx;
        }
        A_blk.block(0, local_A_offset_cols, N_ref, N_ref) = Eigen::MatrixXd::Identity(N_ref, N_ref);
        A_blk.block(N_ref, local_A_offset_cols, N_ref, N_ref) =
          Eigen::MatrixXd::Identity(N_ref, N_ref);
        A_blk.block(2 * N_ref, local_A_offset_cols, N_ref, N_ref) =
          Eigen::MatrixXd::Identity(N_ref, N_ref);

        // lb := [lower_bound - CW
        //        CW - upper_bound
        //               O        ]
        Eigen::VectorXd lb_blk = Eigen::VectorXd::Zero(A_blk_rows);
        lb_blk.segment(0, N_ref) = -CW + part_lb;
        lb_blk.segment(N_ref, N_ref) = CW - part_ub;

        if (s_idx == 1) {
          // add additional clearance
          lb_blk.segment(0, N_ref) -=
            Eigen::MatrixXd::Constant(N_ref, 1, mpt_param_ptr_->soft_second_clearance_from_road);
          lb_blk.segment(N_ref, N_ref) -=
            Eigen::MatrixXd::Constant(N_ref, 1, mpt_param_ptr_->soft_second_clearance_from_road);
        }

        A_offset_cols += N_ref * N_first_slack;

        A.block(A_rows_end, 0, A_blk_rows, A_cols) = A_blk;
        lb.segment(A_rows_end, A_blk_rows) = lb_blk;

        A_rows_end += A_blk_rows;
      }
    }

    // hard constraints
    if (mpt_param_ptr_->hard_constraint) {
      const size_t A_blk_rows = N_ref;

      Eigen::MatrixXd A_blk = Eigen::MatrixXd::Zero(A_blk_rows, A_cols);
      A_blk.block(0, 0, N_ref, N_ref) = CB;

      A.block(A_rows_end, 0, A_blk_rows, A_cols) = A_blk;
      lb.segment(A_rows_end, A_blk_rows) = part_lb - CW;
      ub.segment(A_rows_end, A_blk_rows) = part_ub - CW;

      A_rows_end += A_blk_rows;
    }
  }

  // fixed points constraint
  // CX = C(B v + w) where C extracts fixed points
  if (fixed_points_indices.size() > 0) {
    for (const size_t i : fixed_points_indices) {
      A.block(A_rows_end, 0, D_x, N_ref) = mpt_mat.Bex.block(i * D_x, 0, D_x, N_ref);

      lb.segment(A_rows_end, D_x) =
        ref_points[i].fix_kinematics.get() - mpt_mat.Wex.segment(i * D_x, D_x);
      ub.segment(A_rows_end, D_x) =
        ref_points[i].fix_kinematics.get() - mpt_mat.Wex.segment(i * D_x, D_x);

      //RCLCPP_ERROR_STREAM(
      //  rclcpp::get_logger("fixed_condition"),
      //  i << " " << ref_points[i].fix_kinematics.get()(0) - mpt_mat.Wex(i * D_x) << " "
      //    << ref_points[i].fix_kinematics.get()(1) - mpt_mat.Wex(i * D_x + 1));
      //
      //// TODO(murooka)
      //if (i + 1 < N_ref) {
      //  lb(A_rows_end + i) = ref_points[i + 1].fix_kinematics.get()(0) - bias(i);
      //  ub(A_rows_end + i) = ref_points[i + 1].fix_kinematics.get()(0) - bias(i);
      //} else {
      //  lb(A_rows_end + i) = -bias(i);
      //  ub(A_rows_end + i) = -bias(i);
      //}
      A_rows_end += D_x;
    }
  }

  // steer max limit
  if (mpt_param_ptr_->steer_limit_constraint) {
    A.block(A_rows_end, D_x, N_u, N_u) = Eigen::MatrixXd::Identity(N_u, N_u);
    lb.segment(A_rows_end, N_u) = Eigen::MatrixXd::Constant(N_u, 1, -mpt_param_ptr_->max_steer_rad);
    ub.segment(A_rows_end, N_u) = Eigen::MatrixXd::Constant(N_u, 1, mpt_param_ptr_->max_steer_rad);

    A_rows_end += N_u;
  }

  ConstraintMatrix constraint_matrix;
  constraint_matrix.linear = A;
  constraint_matrix.lower_bound = lb;
  constraint_matrix.upper_bound = ub;

  debug_data_ptr->msg_stream << "          " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
  return constraint_matrix;
}

std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> MPTOptimizer::getMPTPoints(
  std::vector<ReferencePoint> & fixed_ref_points,
  std::vector<ReferencePoint> & non_fixed_ref_points, const Eigen::VectorXd & Uex,
  const MPTMatrix & mpt_mat, std::shared_ptr<DebugData> debug_data_ptr)
{
  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t N_ref = static_cast<size_t>(Uex.rows() - D_x) + 1;

  for (size_t i = 0; i < static_cast<size_t>(Uex.rows()); ++i) {
    // RCLCPP_ERROR_STREAM(rclcpp::get_logger("po"), i << " " << Uex(i));
  }

  stop_watch_.tic(__func__);

  std::vector<double> lat_error_vec;
  std::vector<double> yaw_error_vec;
  for (size_t i = 0; i < fixed_ref_points.size(); ++i) {
    const auto & ref_point = fixed_ref_points.at(i);

    lat_error_vec.push_back(ref_point.fix_kinematics.get()(0));
    yaw_error_vec.push_back(ref_point.fix_kinematics.get()(1));

    // RCLCPP_ERROR_STREAM(
    //   rclcpp::get_logger("fixed"), i << " " << lat_error_vec.back() << " " <<
    //   yaw_error_vec.back());
  }

  const size_t N_kinematics = vehicle_model_ptr_->getDimX();
  const Eigen::VectorXd Xex = mpt_mat.Bex * Uex + mpt_mat.Wex;

  for (size_t i = 0; i < non_fixed_ref_points.size(); ++i) {
    lat_error_vec.push_back(Xex(i * N_kinematics));
    yaw_error_vec.push_back(Xex(i * N_kinematics + 1));

    /*
    if (non_fixed_ref_points.at(i).fix_kinematics) {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("non_fixed"), i << " " << lat_error_vec.back() << " "
                                           << yaw_error_vec.back() << " "
                                           << non_fixed_ref_points.at(i).fix_kinematics.get()(0) <<
    " "
                                           << non_fixed_ref_points.at(i).fix_kinematics.get()(1));
    } else {
      RCLCPP_ERROR_STREAM(
        rclcpp::get_logger("non_fixed"),
        i << " " << lat_error_vec.back() << " " << yaw_error_vec.back());
    }
    */
  }

  // calculate trajectory from optimization result
  std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> traj_points;
  debug_data_ptr->vehicle_circles_pose.resize(lat_error_vec.size());
  for (size_t i = 0; i < lat_error_vec.size(); ++i) {
    auto & ref_point = (i < fixed_ref_points.size())
                         ? fixed_ref_points.at(i)
                         : non_fixed_ref_points.at(i - fixed_ref_points.size());
    const double lat_error = lat_error_vec.at(i);
    const double yaw_error = yaw_error_vec.at(i);

    /*
    RCLCPP_ERROR_STREAM(
      rclcpp::get_logger("full"),
      i << " " << lat_error << " " << yaw_error << " " << ref_point.yaw);
    */

    geometry_msgs::msg::Pose ref_pose;
    ref_pose.position = ref_point.p;
    ref_pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(ref_point.yaw);
    debug_data_ptr->mpt_ref_poses.push_back(ref_pose);
    debug_data_ptr->lateral_errors.push_back(lat_error);

    ref_point.optimized_kinematics << lat_error_vec.at(i), yaw_error_vec.at(i);
    if (i == N_ref - 1) {
      ref_point.optimized_input = 0.0;
    } else {
      ref_point.optimized_input = Uex(D_x + i * D_u);
    }

    autoware_auto_planning_msgs::msg::TrajectoryPoint traj_point;
    traj_point.pose = calcVehiclePose(ref_point, lat_error, yaw_error, 0.0);

    traj_point.longitudinal_velocity_mps = ref_point.v;
    traj_points.push_back(traj_point);

    {  // for debug visualization
      const double base_x = ref_point.p.x - std::sin(ref_point.yaw) * lat_error;
      const double base_y = ref_point.p.y + std::cos(ref_point.yaw) * lat_error;

      // NOTE: coordinate of avoiding_circle_offsets is back wheel center
      for (const double d : mpt_param_ptr_->avoiding_circle_offsets) {
        geometry_msgs::msg::Pose vehicle_circle_pose;

        vehicle_circle_pose.position.x = base_x + d * std::cos(ref_point.yaw + yaw_error);
        vehicle_circle_pose.position.y = base_y + d * std::sin(ref_point.yaw + yaw_error);

        vehicle_circle_pose.orientation =
          tier4_autoware_utils::createQuaternionFromYaw(ref_point.yaw + ref_point.alpha);

        debug_data_ptr->vehicle_circles_pose.at(i).push_back(vehicle_circle_pose);
      }
    }
  }

  /*
  for (size_t i = 0; i < lat_error_vec.size(); ++i) {
    auto & ref_point = (i < fixed_ref_points.size())
                         ? fixed_ref_points.at(i)
                         : non_fixed_ref_points.at(i - fixed_ref_points.size());

    if (i > 0 && traj_points.size() > 1) {
      traj_points.at(i).pose.orientation = geometry_utils::getQuaternionFromPoints(
                                                                         traj_points.at(i).pose.position, traj_points.at(i - 1).pose.position);
    } else if (traj_points.size() > 1) {
      traj_points.at(i).pose.orientation = geometry_utils::getQuaternionFromPoints(
                                                                         traj_points.at(i + 1).pose.position, traj_points.at(i).pose.position);
    } else {
      traj_points.at(i).pose.orientation = tier4_autoware_utils::createQuaternionFromYaw(ref_point.yaw);
    }
  }
  */

  debug_data_ptr->msg_stream << "        " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";

  return traj_points;
}





void MPTOptimizer::calcOrientation(std::vector<ReferencePoint> & ref_points) const
{
  const auto yaw_angles = SplineInterpolation2d::getYawAngles(ref_points);
  for (size_t i = 0; i < ref_points.size(); ++i) {
    if (ref_points.at(i).fix_kinematics) {
      // RCLCPP_ERROR_STREAM(rclcpp::get_logger("yaw"), i << " " << ref_points.at(i).yaw);
      continue;
    }

    ref_points.at(i).yaw = yaw_angles.at(i);
    // RCLCPP_ERROR_STREAM(rclcpp::get_logger("yaw"), i << " " << ref_points.at(i).yaw);
  }
}

void MPTOptimizer::calcVelocity(
  std::vector<ReferencePoint> & ref_points,
  const std::vector<autoware_auto_planning_msgs::msg::TrajectoryPoint> & points) const
{
  for (size_t i = 0; i < ref_points.size(); i++) {
    ref_points.at(i).v = points[tier4_autoware_utils::findNearestIndex(points, ref_points.at(i).p)]
                           .longitudinal_velocity_mps;
  }
}

void MPTOptimizer::calcCurvature(std::vector<ReferencePoint> & ref_points) const
{
  const size_t num_points = static_cast<int>(ref_points.size());

  /* calculate curvature by circle fitting from three points */
  size_t max_smoothing_num = static_cast<size_t>(std::floor(0.5 * (num_points - 1)));
  size_t L =
    std::min(static_cast<size_t>(mpt_param_ptr_->num_curvature_sampling_points), max_smoothing_num);
  auto curvatures = points_utils::calcCurvature(
    ref_points, static_cast<size_t>(mpt_param_ptr_->num_curvature_sampling_points));
  for (size_t i = L; i < num_points - L; ++i) {
    if (!ref_points.at(i).fix_kinematics) {
      ref_points.at(i).k = curvatures.at(i);
    }
  }
  /* first and last curvature is copied from next value */
  for (size_t i = 0; i < std::min(L, num_points); ++i) {
    if (!ref_points.at(i).fix_kinematics) {
      ref_points.at(i).k = ref_points.at(std::min(L, num_points - 1)).k;
    }
    if (!ref_points.at(num_points - i - 1).fix_kinematics) {
      ref_points.at(num_points - i - 1).k =
        ref_points.at(std::max(static_cast<int>(num_points) - static_cast<int>(L) - 1, 0)).k;
    }
  }
}

void MPTOptimizer::calcArcLength(std::vector<ReferencePoint> & ref_points) const
{
  for (size_t i = 0; i < ref_points.size(); i++) {
    if (i > 0) {
      geometry_msgs::msg::Point a, b;
      a = ref_points.at(i).p;
      b = ref_points.at(i - 1).p;
      ref_points.at(i).s = ref_points.at(i - 1).s + tier4_autoware_utils::calcDistance2d(a, b);
    }
  }
}

void MPTOptimizer::calcExtraPoints(std::vector<ReferencePoint> & ref_points) const
{
  for (size_t i = 0; i < ref_points.size(); ++i) {
    // alpha
    const double front_wheel_s = ref_points.at(i).s + vehicle_param_ptr_->wheelbase; // TODO(murooka) use offset?
    const int front_wheel_nearest_idx = points_utils::getNearestIdx(ref_points, front_wheel_s, i);
    const auto front_wheel_pos = ref_points.at(front_wheel_nearest_idx).p;

    const bool are_too_close_points =
      tier4_autoware_utils::calcDistance2d(front_wheel_pos, ref_points.at(i).p) < 1e-03;
    const auto front_wheel_yaw = are_too_close_points ? ref_points.at(i).yaw
                                                      : tier4_autoware_utils::calcAzimuthAngle(
                                                          ref_points.at(i).p, front_wheel_pos);
    ref_points.at(i).alpha =
      tier4_autoware_utils::normalizeRadian(front_wheel_yaw - ref_points.at(i).yaw);

    // RCLCPP_ERROR_STREAM(rclcpp::get_logger("alpha"), i << " " << ref_points.at(i).alpha);

    // near objects
    ref_points.at(i).near_objects = [&]() {
      const int avoidance_check_steps =
        mpt_param_ptr_->near_objects_length /
        mpt_param_ptr_->delta_arc_length_for_mpt_points;  // TODO(murooka) use ros param

      const int avoidance_check_begin_idx =
        std::max(0, static_cast<int>(i) - avoidance_check_steps);
      const int avoidance_check_end_idx =
        std::min(static_cast<int>(ref_points.size()), static_cast<int>(i) + avoidance_check_steps);

      for (int a_idx = avoidance_check_begin_idx; a_idx < avoidance_check_end_idx; ++a_idx) {
        if (ref_points.at(a_idx).vehicle_bounds.at(0).hasCollisionWithObject()) {
          return true;
        }
      }
      return false;
    }();
  }
}

void MPTOptimizer::addSteerWeightR(
  std::vector<Eigen::Triplet<double>> & Rex_triplet_vec, const std::vector<ReferencePoint> & ref_points) const
{
  const size_t D_x = vehicle_model_ptr_->getDimX();
  const size_t D_u = vehicle_model_ptr_->getDimU();
  const size_t N_ref = ref_points.size();
  const size_t N_u = (N_ref - 1) * D_u;
  const size_t D_v = D_x + N_u;

  // add steering rate : weight for (u(i) - u(i-1))^2
  for (size_t i = D_x; i < D_v - 1; ++i) {
    Rex_triplet_vec.push_back(
      Eigen::Triplet<double>(i, i, mpt_param_ptr_->steer_rate_weight));
    Rex_triplet_vec.push_back(
      Eigen::Triplet<double>(i + 1, i, -mpt_param_ptr_->steer_rate_weight));
    Rex_triplet_vec.push_back(
      Eigen::Triplet<double>(i, i + 1, -mpt_param_ptr_->steer_rate_weight));
    Rex_triplet_vec.push_back(
      Eigen::Triplet<double>(i + 1, i + 1, mpt_param_ptr_->steer_rate_weight));
  }
  /*
  if (N > 1) {
    // steer rate i = 0
    R(0, 0) += mpt_param_ptr_->steer_rate_weight / (ctrl_period * ctrl_period);
  }
  */

  /*
  // add steering acceleration : weight for { (u(i+1) - 2*u(i) + u(i-1)) / dt^2 }^2
  const double steer_acc_r = mpt_param_ptr_->steer_acc_weight / std::pow(DT, 4);
  const double steer_acc_r_cp1 = mpt_param_ptr_->steer_acc_weight / (std::pow(DT, 3) * ctrl_period);
  const double steer_acc_r_cp2 =
    mpt_param_ptr_->steer_acc_weight / (std::pow(DT, 2) * std::pow(ctrl_period, 2));
  const double steer_acc_r_cp4 = mpt_param_ptr_->steer_acc_weight / std::pow(ctrl_period, 4);
  for (size_t i = 1; i < N - 1; ++i) {
    R(i - 1, i - 1) += (steer_acc_r);
    R(i - 1, i + 0) += (steer_acc_r * -2.0);
    R(i - 1, i + 1) += (steer_acc_r);
    R(i + 0, i - 1) += (steer_acc_r * -2.0);
    R(i + 0, i + 0) += (steer_acc_r * 4.0);
    R(i + 0, i + 1) += (steer_acc_r * -2.0);
    R(i + 1, i - 1) += (steer_acc_r);
    R(i + 1, i + 0) += (steer_acc_r * -2.0);
    R(i + 1, i + 1) += (steer_acc_r);
  }
  if (N > 1) {
    // steer acc i = 1
    R(0, 0) += steer_acc_r * 1.0 + steer_acc_r_cp2 * 1.0 + steer_acc_r_cp1 * 2.0;
    R(1, 0) += steer_acc_r * -1.0 + steer_acc_r_cp1 * -1.0;
    R(0, 1) += steer_acc_r * -1.0 + steer_acc_r_cp1 * -1.0;
    R(1, 1) += steer_acc_r * 1.0;
    // steer acc i = 0
    R(0, 0) += steer_acc_r_cp4 * 1.0;
  }
  */
}

void MPTOptimizer::addSteerWeightF(Eigen::VectorXd & f) const
{
  constexpr double DT = 0.1;
  constexpr double ctrl_period = 0.03;
  constexpr double raw_steer_cmd_prev = 0;
  constexpr double raw_steer_cmd_pprev = 0;

  if (f.rows() < 2) {
    return;
  }

  // steer rate for i = 0
  f(0) += -2.0 * mpt_param_ptr_->steer_rate_weight / (std::pow(DT, 2)) * 0.5;

  // const double steer_acc_r = mpt_param_.weight_steer_acc / std::pow(DT, 4);
  const double steer_acc_r_cp1 = mpt_param_ptr_->steer_acc_weight / (std::pow(DT, 3) * ctrl_period);
  const double steer_acc_r_cp2 =
    mpt_param_ptr_->steer_acc_weight / (std::pow(DT, 2) * std::pow(ctrl_period, 2));
  const double steer_acc_r_cp4 = mpt_param_ptr_->steer_acc_weight / std::pow(ctrl_period, 4);

  // steer acc  i = 0
  f(0) += ((-2.0 * raw_steer_cmd_prev + raw_steer_cmd_pprev) * steer_acc_r_cp4) * 0.5;

  // steer acc for i = 1
  f(0) += (-2.0 * raw_steer_cmd_prev * (steer_acc_r_cp1 + steer_acc_r_cp2)) * 0.5;
  f(1) += (2.0 * raw_steer_cmd_prev * steer_acc_r_cp1) * 0.5;
}

double MPTOptimizer::calcLateralError(
  const geometry_msgs::msg::Point & target_point, const ReferencePoint & ref_point) const
{
  const double err_x = target_point.x - ref_point.p.x;
  const double err_y = target_point.y - ref_point.p.y;
  const double ref_yaw = ref_point.yaw;
  const double lat_err = -std::sin(ref_yaw) * err_x + std::cos(ref_yaw) * err_y;
  return lat_err;
}

Eigen::Vector2d MPTOptimizer::getState(
  const geometry_msgs::msg::Pose & target_pose, const ReferencePoint & nearest_ref_point) const
{
  const double lat_error = calcLateralError(target_pose.position, nearest_ref_point);
  const double yaw_error = tier4_autoware_utils::normalizeRadian(
    tf2::getYaw(target_pose.orientation) - nearest_ref_point.yaw);
  Eigen::VectorXd kinematics = Eigen::VectorXd::Zero(2);
  kinematics << lat_error, yaw_error;
  return kinematics;
}

void MPTOptimizer::calcBounds(
  std::vector<ReferencePoint> & ref_points, const bool enable_avoidance, const CVMaps & maps,
  std::shared_ptr<DebugData> debug_data_ptr) const
{
  stop_watch_.tic(__func__);

  // search bounds candidate for each ref points
  SequentialBoundsCandidates sequential_bounds_candidates;
  for (const auto & ref_point : ref_points) {
    const auto bounds_candidates = getBoundsCandidates(
      enable_avoidance, convertRefPointsToPose(ref_point), maps, debug_data_ptr);
    sequential_bounds_candidates.push_back(bounds_candidates);
  }
  // debug_data_ptr->sequential_bounds_candidates = sequential_bounds_candidates;

  // search continuous and widest bounds only for front point
  for (size_t i = 0; i < sequential_bounds_candidates.size(); ++i) {
    // NOTE: back() is the front avoiding circle
    const auto & bounds_candidates = sequential_bounds_candidates.at(i);
    const auto & ref_point = ref_points.at(i);

    // extract only continuous bounds;
    if (i == 0) {  // TODO(murooka) use previous bounds, not widest bounds
      const auto & widest_bounds = findWidestBounds(bounds_candidates);
      ref_points.at(i).bounds = widest_bounds;
    } else {
      const auto & prev_ref_point = ref_points.at(i - 1);
      const auto & prev_continuous_bounds = prev_ref_point.bounds;

      // search continuous bounds
      double max_length = std::numeric_limits<double>::min();
      int max_idx = -1;
      for (size_t c_idx = 0; c_idx < bounds_candidates.size(); ++c_idx) {
        const auto & bounds_candidate = bounds_candidates.at(c_idx);
        const double overlapped_length = calcOverlappedBounds(
          convertRefPointsToPose(ref_point), bounds_candidate,
          convertRefPointsToPose(prev_ref_point), prev_continuous_bounds);
        if (overlapped_length > 0 && max_length < overlapped_length) {
          max_length = overlapped_length;
          max_idx = c_idx;
        }
      }

      // find widest bounds
      if (max_idx == -1) {
        // NOTE: set invalid bounds so that MPT won't be solved
        // TODO(murooka) this invalid bounds even makes optimization solved
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("getBounds: front"), "invalid bounds");
        const auto invalid_bounds =
          Bounds{-5.0, 5.0, CollisionType::OUT_OF_ROAD, CollisionType::OUT_OF_ROAD};
        ref_points.at(i).bounds = invalid_bounds;
      } else {
        ref_points.at(i).bounds = bounds_candidates.at(max_idx);
      }
    }
  }

  debug_data_ptr->msg_stream << "            " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
  return;
}

void MPTOptimizer::calcVehicleBounds(
  std::vector<ReferencePoint> & ref_points, [[maybe_unused]] const CVMaps & maps,
  [[maybe_unused]] std::shared_ptr<DebugData> debug_data_ptr,
  [[maybe_unused]] const bool enable_avoidance) const
{
  stop_watch_.tic(__func__);

  SplineInterpolation2d ref_points_spline_interpolation;
  ref_points_spline_interpolation.calcSplineCoefficients(ref_points);

  for (size_t p_idx = 0; p_idx < ref_points.size(); ++p_idx) {
    const auto & ref_point = ref_points.at(p_idx);
    ref_points.at(p_idx).vehicle_bounds.clear();
    ref_points.at(p_idx).beta.clear();

    for (const double d : mpt_param_ptr_->avoiding_circle_offsets) {
      geometry_msgs::msg::Pose avoid_traj_pose;
      avoid_traj_pose.position =
        ref_points_spline_interpolation.getSplineInterpolatedValues(p_idx, d);
      const double vehicle_bounds_pose_yaw = ref_points_spline_interpolation.getYawAngle(p_idx, d);
      avoid_traj_pose.orientation =
        tier4_autoware_utils::createQuaternionFromYaw(vehicle_bounds_pose_yaw);

      const double avoid_yaw = std::atan2(
        avoid_traj_pose.position.y - ref_point.p.y, avoid_traj_pose.position.x - ref_point.p.x);
      const double beta = ref_point.yaw - vehicle_bounds_pose_yaw;
      ref_points.at(p_idx).beta.push_back(beta);

      const double offset_y = -tier4_autoware_utils::calcDistance2d(ref_point, avoid_traj_pose) *
                              std::sin(avoid_yaw - vehicle_bounds_pose_yaw);

      const auto vehicle_bounds_pose =
        tier4_autoware_utils::calcOffsetPose(avoid_traj_pose, 0.0, offset_y, 0.0);

      // interpolate bounds
      const double avoid_s = ref_points_spline_interpolation.getAccumulatedDistance(p_idx) + d;
      for (size_t cp_idx = p_idx; cp_idx < ref_points.size(); ++cp_idx) {
        const double current_s = ref_points_spline_interpolation.getAccumulatedDistance(cp_idx);
        if (avoid_s <= current_s) {
          double prev_avoid_idx;
          if (cp_idx == 0) {
            prev_avoid_idx = cp_idx;
          } else {
            prev_avoid_idx = cp_idx - 1;
          }

          const double prev_s =
            ref_points_spline_interpolation.getAccumulatedDistance(prev_avoid_idx);
          const double next_s =
            ref_points_spline_interpolation.getAccumulatedDistance(prev_avoid_idx + 1);
          const double ratio = (avoid_s - prev_s) / (next_s - prev_s);

          const auto prev_bounds = ref_points.at(prev_avoid_idx).bounds;
          const auto next_bounds = ref_points.at(prev_avoid_idx + 1).bounds;

          auto bounds = Bounds::lerp(prev_bounds, next_bounds, ratio);
          bounds.translate(offset_y);

          ref_points.at(p_idx).vehicle_bounds.push_back(bounds);
          break;
        }

        if (cp_idx == ref_points.size() - 1) {
          ref_points.at(p_idx).vehicle_bounds.push_back(ref_points.back().bounds);
        }
      }

      ref_points.at(p_idx).vehicle_bounds_poses.push_back(vehicle_bounds_pose);
    }
  }

  debug_data_ptr->msg_stream << "            " << __func__ << ":= " << stop_watch_.toc(__func__)
                             << " [ms]\n";
}

BoundsCandidates MPTOptimizer::getBoundsCandidates(
  const bool enable_avoidance, const geometry_msgs::msg::Pose & avoiding_point, const CVMaps & maps,
  [[maybe_unused]] std::shared_ptr<DebugData> debug_data_ptr) const
{
  // stop_watch_.tic(__func__);

  BoundsCandidates bounds_candidate;

  const double lane_width = 5.0;
  const std::vector<double> ds_vec{0.45, 0.15, 0.05};

  // search right to left
  const double bound_angle =
    tier4_autoware_utils::normalizeRadian(tf2::getYaw(avoiding_point.orientation) + M_PI_2);

  double traversed_dist = -lane_width;
  double current_right_bound = -lane_width;

  // calculate the initial position is empty or not
  // 0.drivable, 1.out of map 2.out of road or object
  CollisionType previous_collision_type =
    getCollisionType(maps, enable_avoidance, avoiding_point, traversed_dist, bound_angle);

  const auto has_collision = [&](const CollisionType & collision_type) -> bool {
    return collision_type == CollisionType::OUT_OF_ROAD || collision_type == CollisionType::OBJECT;
  };
  CollisionType latest_right_bound_collision_type = CollisionType::OUT_OF_ROAD;

  while (traversed_dist < lane_width) {
    for (size_t ds_idx = 0; ds_idx < ds_vec.size(); ++ds_idx) {
      const double ds = ds_vec.at(ds_idx);
      while (true) {
        const CollisionType current_collision_type =
          getCollisionType(maps, enable_avoidance, avoiding_point, traversed_dist, bound_angle);

        // return only full bounds whenever finding out of map
        if (current_collision_type == CollisionType::OUT_OF_SIGHT) {
          const auto full_bounds = Bounds{
            -lane_width, lane_width, CollisionType::OUT_OF_SIGHT, CollisionType::OUT_OF_SIGHT};
          return BoundsCandidates({full_bounds});
        }

        if (has_collision(previous_collision_type)) {
          if (!has_collision(current_collision_type)) {  // if target_position becomes no collision
            if (ds_idx == ds_vec.size() - 1) {
              current_right_bound = traversed_dist - ds / 2.0;
              latest_right_bound_collision_type = previous_collision_type;
              previous_collision_type = current_collision_type;
            }
            break;
          }
        } else {
          if (has_collision(current_collision_type)) {
            if (ds_idx == ds_vec.size() - 1) {
              const double left_bound = traversed_dist - ds / 2.0;
              bounds_candidate.push_back(Bounds{
                current_right_bound, left_bound, latest_right_bound_collision_type,
                current_collision_type});
              previous_collision_type = current_collision_type;
            }
            break;
          }
        }

        // if target_position is out of lane
        if (traversed_dist >= lane_width) {
          if (!has_collision(previous_collision_type) && ds_idx == ds_vec.size() - 1) {
            const double left_bound = traversed_dist - ds / 2.0;
            bounds_candidate.push_back(Bounds{
              current_right_bound, left_bound, latest_right_bound_collision_type,
              CollisionType::OUT_OF_ROAD});
          }
          break;
        }

        // go forward with ds
        traversed_dist += ds;
        previous_collision_type = current_collision_type;
      }

      if (ds_idx != ds_vec.size() - 1) {
        // go back with ds since target_position became empty or road/object
        // NOTE: if ds is the last of ds_vec, don't have to go back
        traversed_dist -= ds;
      }
    }
  }

  // if empty
  // TODO(murooka) sometimes this condition realizes in odaiba
  //               and if we use {1.0, -1.0}, MPT can be solved
  if (bounds_candidate.empty()) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("getBoundsCandidate"), "empty");
    // NOTE: set invalid bounds so that MPT won't be solved
    const auto invalid_bounds = Bounds{
      -5.0, 5.0,  // 1.0, -1.0,  // TODO
      CollisionType::OUT_OF_ROAD, CollisionType::OUT_OF_ROAD};
    bounds_candidate.push_back(invalid_bounds);
  }

  // debug_data_ptr->msg_stream << "          " << __func__ <<
  // ":= " << stop_watch_.toc(__func__)  << " [ms]\n";

  return bounds_candidate;
}

// 0.drivable, 1.out of drivable area 2.out of road or object
// 0.NO_COLLISION, 1.OUT_OF_SIGHT, 2.OUT_OF_ROAD, 3.OBJECT
CollisionType MPTOptimizer::getCollisionType(
  const CVMaps & maps, const bool enable_avoidance, const geometry_msgs::msg::Pose & avoiding_point,
  const double traversed_dist, const double bound_angle) const
{
  // calculate clearance
  const double min_soft_road_clearance = mpt_param_ptr_->avoiding_circle_radius +
                                         mpt_param_ptr_->soft_clearance_from_road +
                                         mpt_param_ptr_->extra_desired_clearance_from_road;
  const double min_obj_clearance = mpt_param_ptr_->avoiding_circle_radius +
                                   mpt_param_ptr_->clearance_from_object +
                                   mpt_param_ptr_->soft_clearance_from_road;

  // calculate target position
  const geometry_msgs::msg::Point target_pos = tier4_autoware_utils::createPoint(
    avoiding_point.position.x + traversed_dist * std::cos(bound_angle),
    avoiding_point.position.y + traversed_dist * std::sin(bound_angle), 0.0);

  const auto opt_road_clearance = getClearance(maps.clearance_map, target_pos, maps.map_info);
  const auto opt_obj_clearance =
    getClearance(maps.only_objects_clearance_map, target_pos, maps.map_info);

  // object has more priority than road, so its condition exists first
  if (enable_avoidance && opt_obj_clearance) {
    const bool is_obj = opt_obj_clearance.get() < min_obj_clearance;
    if (is_obj) {
      return CollisionType::OBJECT;
    }
  }

  if (opt_road_clearance) {
    const bool out_of_road = opt_road_clearance.get() < min_soft_road_clearance;
    if (out_of_road) {
      return CollisionType::OUT_OF_ROAD;
    } else {
      return CollisionType::NO_COLLISION;
    }
  }

  return CollisionType::OUT_OF_SIGHT;
}

boost::optional<double> MPTOptimizer::getClearance(
  const cv::Mat & clearance_map, const geometry_msgs::msg::Point & map_point,
  const nav_msgs::msg::MapMetaData & map_info) const
{
  const auto image_point = geometry_utils::transformMapToOptionalImage(map_point, map_info);
  if (!image_point) {
    return boost::none;
  }
  const float clearance = clearance_map.ptr<float>(static_cast<int>(
                            image_point.get().y))[static_cast<int>(image_point.get().x)] *
                          map_info.resolution;
  return clearance;
}


void MPTOptimizer::setEgoData(
  const geometry_msgs::msg::Pose & current_pose, const double current_vel)
{
  current_pose_ = current_pose;
  current_vel_ = current_vel;
}

MPTOptimizer::MPTMatrix MPTOptimizer::translateMPTMatrix(
  const MPTMatrix & mat, const std::vector<double> alpha_vec, const double offset, const size_t D_x,
  const bool only_y) const
{
  // generate T_mat and T_vec to shift a vector
  //   T_mat(X) + T_vec = T_mat * (Bex * U + Wex) + T_vec
  //                    = T_mat * Bex U + T_mat * Wex + T_vec
  const size_t N_ref = alpha_vec.size();

  Eigen::SparseMatrix<double> T_mat(N_ref * (only_y ? 1 : D_x), N_ref * D_x);
  std::vector<Eigen::Triplet<double>> T_triplet;
  Eigen::VectorXd T_vec = Eigen::VectorXd::Zero(N_ref);

  // calculate C mat and vec
  for (size_t i = 0; i < N_ref; ++i) {
    const double alpha = alpha_vec.at(i);
    T_triplet.push_back(Eigen::Triplet<double>(i, i * D_x, 1.0 * std::cos(alpha)));
    T_triplet.push_back(Eigen::Triplet<double>(i, i * D_x + 1, offset * std::cos(alpha)));
    if (!only_y) {
      T_triplet.push_back(Eigen::Triplet<double>(i * D_x + 1, i * D_x + 1, 1.0));
    }

    T_vec(i) = -offset * std::sin(alpha);
  }
  T_mat.setFromTriplets(T_triplet.begin(), T_triplet.end());

  // calculate CB, and CW
  MPTMatrix res_mat;
  res_mat.Bex = T_mat * mat.Bex;
  res_mat.Wex = T_mat * mat.Wex + T_vec;

  return res_mat;
}

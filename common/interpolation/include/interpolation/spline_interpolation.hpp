// Copyright 2021 Tier IV, Inc.
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

#ifndef INTERPOLATION__SPLINE_INTERPOLATION_HPP_
#define INTERPOLATION__SPLINE_INTERPOLATION_HPP_

#include "interpolation/interpolation_utils.hpp"
#include "tier4_autoware_utils/geometry/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace interpolation
{
// NOTE: X(s) = a_i (s - s_i)^3 + b_i (s - s_i)^2 + c_i (s - s_i) + d_i : (i = 0, 1, ... N-1)
struct MultiSplineCoef
{
  MultiSplineCoef() = default;

  explicit MultiSplineCoef(const size_t num_spline)
  {
    a.resize(num_spline);
    b.resize(num_spline);
    c.resize(num_spline);
    d.resize(num_spline);
  }

  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> d;
};

std::vector<double> slerp(
  const std::vector<double> & base_keys, const std::vector<double> & base_values,
  const std::vector<double> & query_keys);

std::vector<double> slerpDiff(
  const std::vector<double> & base_keys, const std::vector<double> & base_values,
  const std::vector<double> & query_keys);
}  // namespace interpolation

class SplineInterpolation
{
public:
  SplineInterpolation() = default;

  static std::vector<double> slerp(
    const std::vector<double> & base_keys, const std::vector<double> & base_values,
    const std::vector<double> & query_keys)
  {
    // calculate spline coefficients
    const auto multi_spline_coef = getSplineCoefficients(base_keys, base_values);

    // interpolate base_keys at query_keys
    return getSplineInterpolatedValues(base_keys, query_keys, multi_spline_coef);
  }

  static std::vector<double> slerpDiff(
    const std::vector<double> & base_keys, const std::vector<double> & base_values,
    const std::vector<double> & query_keys)
  {
    // calculate spline coefficients
    const auto multi_spline_coef = getSplineCoefficients(base_keys, base_values);

    // interpolate base_keys at query_keys
    return getSplineInterpolatedDiffValues(base_keys, query_keys, multi_spline_coef);
  }

protected:
  static interpolation::MultiSplineCoef getSplineCoefficients(
    const std::vector<double> & base_keys, const std::vector<double> & base_values)
  {
    // throw exceptions for invalid arguments
    interpolation_utils::validateKeysAndValues(base_keys, base_values);

    const size_t num_base = base_keys.size();  // N+1

    std::vector<double> diff_keys;    // N
    std::vector<double> diff_values;  // N
    for (size_t i = 0; i < num_base - 1; ++i) {
      diff_keys.push_back(base_keys.at(i + 1) - base_keys.at(i));
      diff_values.push_back(base_values.at(i + 1) - base_values.at(i));
    }

    std::vector<double> v = {0.0};
    if (num_base > 2) {
      // solve tridiagonal matrix algorithm
      interpolation_utils::TDMACoef tdma_coef(num_base - 2);  // N-1

      for (size_t i = 0; i < num_base - 2; ++i) {
        tdma_coef.b[i] = 2 * (diff_keys[i] + diff_keys[i + 1]);
        if (i != num_base - 3) {
          tdma_coef.a[i] = diff_keys[i + 1];
          tdma_coef.c[i] = diff_keys[i + 1];
        }
        tdma_coef.d[i] =
          6.0 * (diff_values[i + 1] / diff_keys[i + 1] - diff_values[i] / diff_keys[i]);
      }

      const std::vector<double> tdma_res =
        interpolation_utils::solveTridiagonalMatrixAlgorithm(tdma_coef);

      // calculate v
      v.insert(v.end(), tdma_res.begin(), tdma_res.end());
    }
    v.push_back(0.0);

    // calculate a, b, c, d of spline coefficients
    interpolation::MultiSplineCoef multi_spline_coef(num_base - 1);  // N
    for (size_t i = 0; i < num_base - 1; ++i) {
      multi_spline_coef.a[i] = (v[i + 1] - v[i]) / 6.0 / diff_keys[i];
      multi_spline_coef.b[i] = v[i] / 2.0;
      multi_spline_coef.c[i] =
        diff_values[i] / diff_keys[i] - diff_keys[i] * (2 * v[i] + v[i + 1]) / 6.0;
      multi_spline_coef.d[i] = base_values[i];
    }

    return multi_spline_coef;
  }

  static std::vector<double> getSplineInterpolatedValues(
    const std::vector<double> & base_keys, const std::vector<double> & query_keys,
    const interpolation::MultiSplineCoef & multi_spline_coef)
  {
    // throw exceptions for invalid arguments
    interpolation_utils::validateKeys(base_keys, query_keys);

    const auto & a = multi_spline_coef.a;
    const auto & b = multi_spline_coef.b;
    const auto & c = multi_spline_coef.c;
    const auto & d = multi_spline_coef.d;

    std::vector<double> res;
    size_t j = 0;
    for (const auto & query_key : query_keys) {
      while (base_keys.at(j + 1) < query_key) {
        ++j;
      }

      const double ds = query_key - base_keys.at(j);
      res.push_back(d.at(j) + (c.at(j) + (b.at(j) + a.at(j) * ds) * ds) * ds);
    }

    return res;
  }

  static std::vector<double> getSplineInterpolatedDiffValues(
    const std::vector<double> & base_keys, const std::vector<double> & query_keys,
    const interpolation::MultiSplineCoef & multi_spline_coef)
  {
    // throw exceptions for invalid arguments
    interpolation_utils::validateKeys(base_keys, query_keys);

    const auto & a = multi_spline_coef.a;
    const auto & b = multi_spline_coef.b;
    const auto & c = multi_spline_coef.c;

    std::vector<double> res;
    size_t j = 0;
    for (const auto & query_key : query_keys) {
      while (base_keys.at(j + 1) < query_key) {
        ++j;
      }

      const double ds = query_key - base_keys.at(j);
      res.push_back(c.at(j) + (2.0 * b.at(j) + 3.0 * a.at(j) * ds) * ds);
    }

    return res;
  }
};

class SplineInterpolation1d : public SplineInterpolation
{
public:
  SplineInterpolation1d() = default;

  void calcSplineCoefficients(
    const std::vector<double> & base_keys, const std::vector<double> & base_values)
  {
    multi_spline_coef_ = getSplineCoefficients(base_keys, base_values);
  }

  std::vector<double> getSplineInterpolatedValues(
    const std::vector<double> & base_keys, const std::vector<double> & query_keys) const
  {
    return SplineInterpolation::getSplineInterpolatedValues(
      base_keys, query_keys, multi_spline_coef_);
  }

private:
  interpolation::MultiSplineCoef multi_spline_coef_;
};

class SplineInterpolation2d : public SplineInterpolation
{
public:
  SplineInterpolation2d() = default;

  template <typename T>
  static std::vector<double> getYawAngles(const std::vector<T> & points)
  {
    const auto base = getBaseValues(points);

    const auto & base_s = base.at(0);
    const auto & base_x = base.at(1);
    const auto & base_y = base.at(2);

    // calculate spline coefficients
    const auto multi_spline_coef_x = getSplineCoefficients(base_s, base_x);
    const auto multi_spline_coef_y = getSplineCoefficients(base_s, base_y);

    // interpolate base_keys at query_keys
    const auto diff_x_vec = getSplineInterpolatedDiffValues(base_s, base_s, multi_spline_coef_x);
    const auto diff_y_vec = getSplineInterpolatedDiffValues(base_s, base_s, multi_spline_coef_y);

    std::vector<double> yaw_angle_vec;
    for (size_t i = 0; i < diff_x_vec.size(); ++i) {
      const double diff_x = diff_x_vec.at(i);
      const double diff_y = diff_y_vec.at(i);

      const double yaw = std::atan2(diff_y, diff_x);
      yaw_angle_vec.push_back(yaw);
    }

    return yaw_angle_vec;
  }

  template <typename T>
  void calcSplineCoefficients(const std::vector<T> & points)
  {
    const auto base = getBaseValues(points);

    base_s_vec_ = base.at(0);
    const auto & base_x_vec = base.at(1);
    const auto & base_y_vec = base.at(2);

    // calculate spline coefficients
    multi_spline_coef_x_ = getSplineCoefficients(base_s_vec_, base_x_vec);
    multi_spline_coef_y_ = getSplineCoefficients(base_s_vec_, base_y_vec);
  }

  geometry_msgs::msg::Point getSplineInterpolatedValues(const size_t idx, const double s)
  {
    double whole_s = base_s_vec_.at(idx) + s;
    if (whole_s < base_s_vec_.front()) {
      whole_s = base_s_vec_.front();
    }
    if (whole_s > base_s_vec_.back()) {
      whole_s = base_s_vec_.back();
    }

    const double x =
      SplineInterpolation::getSplineInterpolatedValues(base_s_vec_, {whole_s}, multi_spline_coef_x_)
        .at(0);
    const double y =
      SplineInterpolation::getSplineInterpolatedValues(base_s_vec_, {whole_s}, multi_spline_coef_y_)
        .at(0);

    geometry_msgs::msg::Point geom_point;
    geom_point.x = x;
    geom_point.y = y;
    return geom_point;
  }

  double getYawAngle(const size_t idx, const double s)
  {
    double whole_s = base_s_vec_.at(idx) + s;
    /*
    if (whole_s < base_s_vec_.front() || whole_s > base_s_vec_.back()) {
      return {};
    }
    */
    if (whole_s < base_s_vec_.front()) {
      whole_s = base_s_vec_.front();
    }
    if (whole_s > base_s_vec_.back()) {
      whole_s = base_s_vec_.back();
    }

    const double diff_x =
      getSplineInterpolatedDiffValues(base_s_vec_, {whole_s}, multi_spline_coef_x_).at(0);
    const double diff_y =
      getSplineInterpolatedDiffValues(base_s_vec_, {whole_s}, multi_spline_coef_y_).at(0);

    return std::atan2(diff_y, diff_x);
  }

  double getAccumulatedDistance(const size_t idx) const { return base_s_vec_.at(idx); }

  // void setPoint(T & point, const geometry_msgs::msg::Point & geom_point) { point = geom_point; }

private:
  interpolation::MultiSplineCoef multi_spline_coef_x_;
  interpolation::MultiSplineCoef multi_spline_coef_y_;
  std::vector<double> base_s_vec_;

  template <typename T>
  static std::array<std::vector<double>, 3> getBaseValues(const std::vector<T> & points)
  {
    // calculate x, y
    std::vector<double> base_x;
    std::vector<double> base_y;
    for (size_t i = 0; i < points.size(); i++) {
      const auto & current_pos = tier4_autoware_utils::getPoint(points.at(i));
      if (i > 0) {
        const auto & prev_pos = tier4_autoware_utils::getPoint(points.at(i - 1));
        if (
          std::fabs(current_pos.x - prev_pos.x) < 1e-6 &&
          std::fabs(current_pos.y - prev_pos.y) < 1e-6) {
          continue;
        }
      }
      base_x.push_back(current_pos.x);
      base_y.push_back(current_pos.y);
    }

    // calculate base_keys, base_values
    if (base_x.empty() || base_y.empty()) {
      // return std::vector<geometry_msgs::msg::Point>{};
    }
    const std::vector<double> base_s = calcEuclidDist(base_x, base_y);
    if (base_s.empty() || base_s.size() == 1) {
      // return std::vector<geometry_msgs::msg::Point>{};
    }

    return {base_s, base_x, base_y};
  }

  static std::vector<double> calcEuclidDist(
    const std::vector<double> & x, const std::vector<double> & y)
  {
    if (x.size() != y.size()) {
      return std::vector<double>{};
    }

    std::vector<double> dist_v;
    dist_v.push_back(0.0);
    for (size_t i = 0; i < x.size() - 1; ++i) {
      const double dx = x.at(i + 1) - x.at(i);
      const double dy = y.at(i + 1) - y.at(i);
      dist_v.push_back(dist_v.at(i) + std::hypot(dx, dy));
    }

    return dist_v;
  }
};

#endif  // INTERPOLATION__SPLINE_INTERPOLATION_HPP_

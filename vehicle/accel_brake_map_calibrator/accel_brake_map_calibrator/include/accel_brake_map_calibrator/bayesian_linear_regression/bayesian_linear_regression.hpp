//
// Copyright 2021 Tier IV, Inc. All rights reserved.
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
//

#ifndef ACCEL_BRAKE_MAP_CALIBRATOR__BAYESIAN_LINEAR_REGRESSION_HPP_
#define ACCEL_BRAKE_MAP_CALIBRATOR__BAYESIAN_LINEAR_REGRESSION_HPP_

#include <vector>
#include "eigen3/Eigen/Core"

class KernelInterface
{
public:
  KernelInterface(const double new_data_cov)
    : new_data_cov_(new_data_cov)
  {
  }

  void updateWeight(const Eigen::MatrixXd x, const Eigen::VectorXd t)
  {
    const Eigen::MatrixXd phi = calcKernelMatrix(x);

    const Eigen::MatrixXd weight_sigma_inv = weight_sigma_.inverse();

    const Eigen::VectorXd tmp_weight_mu = weight_sigma_ * (weight_sigma_inv * weight_mu_ + new_data_cov_ * phi.transpose() * t);
    const Eigen::MatrixXd tmp_weight_sigma = weight_sigma_inv + new_data_cov_ * phi.transpose() * phi;

    weight_mu_ = tmp_weight_mu;
    weight_sigma_ = tmp_weight_sigma;
  }

  Eigen::VectorXd predict(const Eigen::MatrixXd x) const
  {
    return calcKernelMatrix(x) * weight_mu_;
  }

  void initializeWeight()
  {
    constexpr double const_large_sigma = 1000.0;
    weight_mu_ = Eigen::VectorXd::Zero(weight_mu_.rows());
    weight_sigma_ = Eigen::MatrixXd::Constant(weight_sigma_.rows(), weight_sigma_.cols(), const_large_sigma);
  }

protected:
  double new_data_cov_;

  Eigen::VectorXd weight_mu_;
  Eigen::MatrixXd weight_sigma_;

  virtual Eigen::MatrixXd calcSingleKernelMatrix(const Eigen::MatrixXd x) const;
  virtual Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd x) const;
};

class PolynomialKernel : public KernelInterface
{
public:
  PolynomialKernel(const size_t dim_x, const size_t dim_poly, const double new_data_cov)
    : KernelInterface(new_data_cov), dim_x_(dim_x), dim_poly_(dim_poly)
  {
    const size_t dim_weight = dim_x_ + dim_poly + 1;   // NOTE: (1, x_1, x_2, x_1^2, x_2^2, ...)
    weight_mu_.resize(dim_weight);
    weight_sigma_.resize(dim_weight, dim_weight);

    initializeWeight();
  }

private:
  const size_t dim_x_;
  const size_t dim_poly_;

  // When dim_ = 3 and dim_x = 2,
  // single kernel matrix will be (1, x_1, x_2, x_1^2, x_2^2, x_1^3, x_2^3).
  // NOTE: rows, cols of x is 1, dim_x.
  Eigen::MatrixXd calcSingleKernelMatrix(const Eigen::MatrixXd x) const override
  {
    if (static_cast<int>(dim_x_) != x.cols()) {
    }

    Eigen::MatrixXd single_kernel(1, dim_x_ * dim_poly_ + 1);
    single_kernel.block(0, 0, 1, 1) = Eigen::MatrixXd::Identity(1, 1);

    Eigen::MatrixXd mul_x = x;
    for (size_t i = 0; i < dim_poly_; ++i) {
      single_kernel.block(0, dim_x_ * i + 1, 1, dim_x_) = mul_x;
      mul_x = mul_x.diagonal() * x;
    }

    return single_kernel;
  }

  Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd x) const override
  {
    if (static_cast<int>(dim_x_) != x.cols()) {
    }

    Eigen::MatrixXd kernel_mat(x.rows(), dim_x_ * dim_poly_ + 1);

    Eigen::VectorXd mul_x = x;
    for (size_t i = 0; static_cast<int>(i) < x.rows(); ++i) {
      kernel_mat.block(i, 0, 1, kernel_mat.cols()) = calcSingleKernelMatrix(x);
    }

    return kernel_mat;
  }
};

enum class KernelType
{
 Polynomial = 0,
 RBF
};

struct SamplingInfo
{
  double grid_num;
  double min_val;
  double max_val;
};

class BayesianLinearRegression
{
public:
  BayesianLinearRegression(KernelType kernel_type, const size_t dim_x, const std::vector<SamplingInfo> sampling_info)
    : dim_x_(dim_x)
  {
    if (kernel_type == KernelType::RBF) {
      // kernel_ = std::make_shared<RBFKernel>(new_data_cov);
    } else if (kernel_type == KernelType::Polynomial) {
      const double new_data_cov = 1.0;
      kernel_ = std::make_shared<PolynomialKernel>(dim_x, 3, new_data_cov);
    } else {
    }

    if (dim_x_ != sampling_info.size()) {
    }
  }

  void reset() const
  {
    kernel_->initializeWeight();
  }

  void fit(std::vector<std::vector<double>> data)
  {
    Eigen::MatrixXd x_eigen_mat(data.at(0).size(), data.size() - 1);
    Eigen::VectorXd t_eigen_vec(data.at(0).size());

    // convert std vector to eigen vector
    for (size_t i = 0; i < data.size(); ++i) {
      if (i == data.size() - 1) {
        t_eigen_vec.segment(i, data.at(0).size()) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data.at(i).data(), data.at(i).size());
      } else {
        x_eigen_mat.block(0, i, data.at(0).size(), 1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(data.at(i).data(), data.at(i).size());
      }
    }

    // update kernel weight by regression
    kernel_->updateWeight(x_eigen_mat, t_eigen_vec);
  }

  Eigen::VectorXd predict(std::vector<std::vector<double>> x_data) const
  {
    Eigen::MatrixXd x_eigen_mat(x_data.at(0).size(), x_data.size() - 1);

    // convert std vector to eigen vector
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_eigen_mat.block(0, i, x_data.at(0).size(), 1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x_data.at(i).data(), x_data.at(i).size());
    }

    // predict and return
    return kernel_->predict(x_eigen_mat);
  }

private:
  const size_t dim_x_;
  std::shared_ptr<KernelInterface> kernel_;
};

#endif  // ACCEL_BRAKE_MAP_CALIBRATOR__BAYESIAN_LINEAR_REGRESSION_HPP_

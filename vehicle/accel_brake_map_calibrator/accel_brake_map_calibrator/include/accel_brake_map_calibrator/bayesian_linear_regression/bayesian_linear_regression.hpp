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

struct SamplingInfo
{
  std::vector<int> grid_num_vec;
  std::vector<double> min_val_vec;
  std::vector<double> max_val_vec;
};

class KernelInterface
{
public:
  KernelInterface() {}
  virtual Eigen::VectorXd calc(const Eigen::MatrixXd x);

protected:
  virtual Eigen::MatrixXd calcSingleKernelMatrix(const Eigen::MatrixXd x) const;
  virtual Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd x) const;
};

class RBFKernel : public KernelInterface
{
public:
  RBFKernel(const SamplingInfo & sampling_info)
    : KernelInterface(), sampling_info_(sampling_info)
  {
    // calculate dimension of feature (the number of RBF)
    dim_feature_ = 0;
    for (const size_t grid_num : grid_num_vec) {
      dim_feature_ += grid_num;
    }

    std::vector<std::vector<double>> base_x_vec;
    const double dim_x = x.rows();
    for (size_t x_idx = 0; x_idx < dim_x, ++x_idx) {
      const int grid_num = sampling_info_.grid_num_vec.at(x_idx);
      const double max_val = sampling_info_.max_val_vec.at(x_idx);
      const double min_val = sampling_info_.min_val_vec.at(x_idx);

      std::vector<double> single_x_vec;
      for (size_t g_idx = 0; g_idx < grid_num; ++g_idx) {
        const double current_val = min_val + (max_val - min_val) * g_idx / (grid_num - 1.0);
        single_x_vec.push_back(current_val);
      }

      base_x_vec.push_back(single_x_vec);
    }

    Eigen::MatrixXd single_kernel_mat(sampling_info_.grid_num_vec.size(), dim_feature_);
    for (size_t b_idx = 0; b_idx < base_x_vec.size(); ++b_ix) {
      for (size_t g_idx = 0; g_idx < base_x_vec.at(b_idx); ++g_idx) {
        const size_t m_idx = 
        single_kernel_mat(b_idx, m_idx) = base_x_vec.at(b_idx).at(g_idx);
      }
    }
  }

  Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd x) const override
  {
    Eigen::MatrixXd kernel_mat(dim_feature_, x.cols());

    for (size_t c_idx = 0; c_idx < x.cols(); ++c_idx) {
      const Eigen::VectorXd single_x = x.block(0, c_idx, x.rows(), 1);
      const Eigen::VectorXd single_phi = calcSingleKernelMatrix(single_x);

      kernel_mat.block(0, c_idx, dim_feature_, 1) = single_phi;
    }

    return kernel_mat;
  }

private:
  size_t dim_feature_;
  Eigen::MatrixXd base_state_vec_;

  Eigen::MatrixXd calcSingleKernelMatrix(const Eigen::VectorXd x) const override
  {
    return single_kernel_mat;
  }
};

enum class KernelType
{
 Polynomial = 0,
 RBF
};

class BayesianLinearRegression
{
public:
  BayesianLinearRegression(KernelType kernel_type, const std::vector<SamplingInfo> sampling_info, const double initial_weight_sigma, const double observe_sigma)
    : dim_x_(sampling_info.size()), observe_sigma_(observe_sigma)
  {
    if (kernel_type == KernelType::RBF) {
      kernel_ = std::make_shared<RBFKernel>(new_data_cov);
    } else if (kernel_type == KernelType::Polynomial) {
      const double new_data_cov = 1.0;
      kernel_ = std::make_shared<PolynomialKernel>(dim_x, 3, new_data_cov);
    } else {
    }

    dim_feature_ = kenel_->getDimFeature();
    initializeWeight(initial_weight_sigma);
  }

  void initializeWeight(const double initial_weight_sigma)
  {
    weight_mu_ = Eigen::VectorXd::Zero(dim_feature_);
    weight_sigma_ = Eigen::MatrixXd::Identity(dim_feature_, dim_feature_) * initial_weight_sigma;
  }

  void reset(const double initial_weight_sigma)
  {
    kernel_->initializeWeight(initial_weight_sigma);
  }

  void updateWeight(const Eigen::MatrixXd x, const Eigen::VectorXd t)
  {
    const Eigen::MatrixXd phi = calcKernelMatrix(x);

    const Eigen::MatrixXd weight_sigma_inv = weight_sigma_.inverse();

    const Eigen::MatrixXd next_weight_sigma = weight_sigma_inv + std::pow(observe_sigma_, -2.0) * phi.transpose() * phi;
    const Eigen::VectorXd next_weight_mu = next_weight_sigma_ * (weight_sigma_inv * weight_mu_ + std::pow(observe_sigma_, -2.0) * phi.transpose() * t);

    weight_mu_ = next_weight_mu;
    weight_sigma_ = next_weight_sigma;
  }

  void fit(const std::vector<std::vector<double>> x_vec, const std::vector<double> t_vec)
  {
    if (x_vec.size() != t_vec.size()) {
      throw std::logic_error("The sizes of x_vec and t_vec are different");
    }

    Eigen::MatrixXd x_eigen_mat(x_vec.size(), x_vec.at(0).size());
    Eigen::MatrixXd t_eigen_mat(t_vec.size());

    // convert std vector to eigen vector
    for (size_t i = 0; i < x_vec.size(); ++i) {
      x_eigen_mat.block(0, i, x_vec.at(i).size(), 1) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x_vec.at(i).data(), x_vec.at(i).size());
    }
    t_eigen_mat.block(0, 0, 1, t_vec.size()) = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(t_vec.data(), t_vec.size());


    // update kernel weight by regression
    kernel_->updateWeight(x_eigen_mat, t_eigen_vec);
  }

  /*
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
  */

private:
  // const size_t dim_x_;
  std::shared_ptr<KernelInterface> kernel_;
  size_t dim_feature_;

  double observe_sigma_;

  Eigen::VectorXd weight_mu_;
  Eigen::MatrixXd weight_sigma_;

};

#endif  // ACCEL_BRAKE_MAP_CALIBRATOR__BAYESIAN_LINEAR_REGRESSION_HPP_

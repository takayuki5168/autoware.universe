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

#include <memory>
#include <vector>
#include "eigen3/Eigen/Core"
#include <Eigen/LU>

struct SamplingInfo
{
  std::vector<size_t> grid_num_vec;
  std::vector<double> max_val_vec;
  std::vector<double> min_val_vec;
};

class KernelInterface
{
public:
  KernelInterface() {}
  virtual Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd & x) const = 0;
  virtual size_t getDimFeature() const = 0;

protected:
  virtual Eigen::VectorXd calcSingleKernelMatrix(const Eigen::VectorXd & x) const = 0;
};

class RBFKernel : public KernelInterface
{
public:
  RBFKernel(const double gamma, const SamplingInfo & sampling_info)
    : KernelInterface(), gamma_(gamma)
  {
    // calculate dimension of feature (the number of RBF kernel)
    dim_feature_ = 1;
    for (const size_t grid_num : sampling_info.grid_num_vec) {
      dim_feature_ *= grid_num;
    }

    std::vector<std::vector<double>> base_x_vec;
    for (size_t x_idx = 0; x_idx < sampling_info.grid_num_vec.size(); ++x_idx) {
      const int grid_num = sampling_info.grid_num_vec.at(x_idx);
      const double max_val = sampling_info.max_val_vec.at(x_idx);
      const double min_val = sampling_info.min_val_vec.at(x_idx);

      std::vector<double> single_x_vec;
      for (size_t g_idx = 0; g_idx < grid_num; ++g_idx) {
        const double current_val = min_val + (max_val - min_val) * g_idx / (grid_num - 1.0);
        single_x_vec.push_back(current_val);
      }

      base_x_vec.push_back(single_x_vec);
    }

    base_state_mat_.resize(sampling_info.grid_num_vec.size(), dim_feature_);
    size_t divide_num = 1;
    for (size_t b_idx = 0; b_idx < base_x_vec.size(); ++b_idx) {
      divide_num *= base_x_vec.at(b_idx).size();
      const size_t same_data_width = dim_feature_ / divide_num;
      for (size_t g_idx = 0; g_idx < base_x_vec.at(b_idx).size(); ++g_idx) {
        size_t m_idx = 0;
        for (size_t d_idx = 0; d_idx < dim_feature_; ++d_idx) {
          if (d_idx != 0 && d_idx % same_data_width == 0) {
            m_idx += 1;
          }
          base_state_mat_(b_idx, d_idx) = base_x_vec.at(b_idx).at(m_idx % base_x_vec.at(b_idx).size());
        }
      }
    }
  }

  Eigen::MatrixXd calcKernelMatrix(const Eigen::MatrixXd & x) const override
  {
    Eigen::MatrixXd kernel_mat(x.cols(), dim_feature_);

    for (size_t c_idx = 0; c_idx < x.cols(); ++c_idx) {
      const Eigen::VectorXd single_x = x.block(0, c_idx, x.rows(), 1);
      const Eigen::VectorXd single_phi = calcSingleKernelMatrix(single_x);

      kernel_mat.block(c_idx, 0, 1, dim_feature_) = single_phi.transpose();
    }

    return kernel_mat;
  }

  size_t getDimFeature() const { return dim_feature_; }

private:
  double gamma_;

  size_t dim_feature_;
  Eigen::MatrixXd base_state_mat_;

  Eigen::VectorXd calcSingleKernelMatrix(const Eigen::VectorXd & x) const override
  {
    Eigen::VectorXd feature_vec(base_state_mat_.cols());

    for (size_t c_idx = 0; c_idx < base_state_mat_.cols(); ++c_idx) {
      const Eigen::VectorXd diff_state = x - base_state_mat_.block(0, c_idx, base_state_mat_.rows(), 1);
      feature_vec(c_idx) = std::exp(-std::pow(diff_state.norm(), 2) / 2.0 / std::pow(gamma_, 2));
    }

    return feature_vec;
  }
};

enum class KernelType
{
 RBF = 0,
};

class BayesianLinearRegression
{
public:
  BayesianLinearRegression(KernelType kernel_type, const SamplingInfo & sampling_info, const double initial_weight_sigma, const double observe_sigma)
    : observe_sigma_(observe_sigma)
  {
    if (kernel_type == KernelType::RBF) {
      const double gamma = 1.0;
      kernel_ = std::make_unique<RBFKernel>(gamma, sampling_info);
    } else {
      throw std::logic_error("Kernel type is invalid.");
    }

    dim_feature_ = kernel_->getDimFeature();
    initializeWeight(initial_weight_sigma);
  }

  void initializeWeight(const double initial_weight_sigma)
  {
    weight_mu_ = Eigen::VectorXd::Zero(dim_feature_);
    weight_sigma_ = Eigen::MatrixXd::Identity(dim_feature_, dim_feature_) * initial_weight_sigma;
  }

  void reset(const double initial_weight_sigma)
  {
    initializeWeight(initial_weight_sigma);
  }

  void fit(const std::vector<std::vector<double>> & x_vec, const std::vector<std::vector<double>> & t_vec)
  {
    if (x_vec.at(0).size() != t_vec.at(0).size()) {
      std::stringstream ss;
      ss << "The sizes of x_vec and t_vec are different. x_vec.at(0).size() = " << x_vec.at(0).size() << ", t_vec.at(0).size() = " << t_vec.at(0).size();
      throw std::logic_error(ss.str());
    }

    Eigen::MatrixXd x_eigen_mat(x_vec.size(), x_vec.at(0).size());
    Eigen::MatrixXd t_eigen_mat(t_vec.size(), t_vec.at(0).size());

    // convert std vector to eigen vector
    for (size_t i = 0; i < x_vec.size(); ++i) {
      x_eigen_mat.block(i, 0, 1, x_vec.at(i).size()) = Eigen::Map<const Eigen::MatrixXd>(x_vec.at(i).data(), 1, x_vec.at(i).size());
    }
    for (size_t i = 0; i < t_vec.size(); ++i) {
      t_eigen_mat.block(i, 0, 1, t_vec.at(i).size()) = Eigen::Map<const Eigen::MatrixXd>(t_vec.at(i).data(), 1, t_vec.at(i).size());
    }

    // update kernel weight by regression
    updateWeight(x_eigen_mat, t_eigen_mat);
  }

  Eigen::MatrixXd predict(const std::vector<double> & max_val_vec, const std::vector<double> & min_val_vec) const
  {
    constexpr int grid_num = 100;
    const int sample_num = std::pow(grid_num, max_val_vec.size());

    std::cout << "po1" << std::endl;
    std::vector<std::vector<double>> sample_x_vec;
    for (size_t x_idx = 0; x_idx < max_val_vec.size(); ++x_idx) {
      const double max_val = max_val_vec.at(x_idx);
      const double min_val = min_val_vec.at(x_idx);

      std::vector<double> single_x_vec;
      for (size_t g_idx = 0; g_idx < grid_num; ++g_idx) {
        const double current_val = min_val + (max_val - min_val) * g_idx / (grid_num - 1.0);
        single_x_vec.push_back(current_val);
      }

      sample_x_vec.push_back(single_x_vec);
    }

    std::cout << "po2" << std::endl;
    Eigen::MatrixXd sample_state_mat(max_val_vec.size(), sample_num);
    size_t divide_num = 1;
    for (size_t b_idx = 0; b_idx < sample_x_vec.size(); ++b_idx) {
      divide_num *= sample_x_vec.at(b_idx).size();
      const size_t same_data_width = sample_num / divide_num;
      for (size_t g_idx = 0; g_idx < sample_x_vec.at(b_idx).size(); ++g_idx) {
        size_t m_idx = 0;
        for (size_t d_idx = 0; d_idx < sample_num; ++d_idx) {
          if (d_idx != 0 && d_idx % same_data_width == 0) {
            m_idx += 1;
          }
          sample_state_mat(b_idx, d_idx) = sample_x_vec.at(b_idx).at(m_idx % sample_x_vec.at(b_idx).size());
        }
      }
    }

    std::cout << "po3" << std::endl;
    // predict and return
    const Eigen::MatrixXd phi = kernel_->calcKernelMatrix(sample_state_mat);
    const Eigen::MatrixXd predicted_state = weight_mu_.transpose() * phi.transpose();
    return predicted_state;
  }

private:
  double observe_sigma_;

  std::unique_ptr<KernelInterface> kernel_;
  size_t dim_feature_;

  Eigen::VectorXd weight_mu_;
  Eigen::MatrixXd weight_sigma_;

  void updateWeight(const Eigen::MatrixXd & x, const Eigen::MatrixXd & t)
  {
    const Eigen::MatrixXd phi = kernel_->calcKernelMatrix(x);

    const Eigen::MatrixXd weight_sigma_inv = weight_sigma_.inverse();

    const Eigen::MatrixXd next_weight_sigma = weight_sigma_inv + std::pow(observe_sigma_, -2.0) * phi.transpose() * phi;
    const Eigen::VectorXd next_weight_mu = next_weight_sigma * (weight_sigma_inv * weight_mu_ + std::pow(observe_sigma_, -2.0) * phi.transpose() * t.transpose()); // TODO(murooka) t dimension

    weight_mu_ = next_weight_mu;
    weight_sigma_ = next_weight_sigma;
  }
};

#endif  // ACCEL_BRAKE_MAP_CALIBRATOR__BAYESIAN_LINEAR_REGRESSION_HPP_

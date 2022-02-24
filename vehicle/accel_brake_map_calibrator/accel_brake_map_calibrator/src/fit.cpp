#include <iostream>
#include <sstream>
#include <fstream>

#include "accel_brake_map_calibrator/bayesian_linear_regression/bayesian_linear_regression.hpp"

int main(int argc, char ** argv)
{
  std::ifstream ifs("./log.csv");

  std::vector<double> time_vec;
  std::vector<double> vel_vec;
  std::vector<double> acc_vec;
  std::vector<double> pedal_vec;

  std::string l_str;
  size_t l_idx = 0;
  while (getline(ifs, l_str)) {
    if (10 < l_idx) {
      break;
    }

    std::stringstream ss{l_str};
    try {
      double time;
      double vel;
      double acc;
      double throttle;
      double brake;
      std::string str;
      size_t e_idx = 0;
      while (getline(ss, str, ',')) {
        if (e_idx == 0) {
          time = std::stod(str);
        } else if (e_idx == 1) {
          vel = std::stod(str);
        } else if (e_idx == 2) {
          acc = std::stod(str);
        } else if (e_idx == 5) {
          throttle = std::stod(str);
        } else if (e_idx == 6) {
          brake = std::stod(str);
        }
        ++e_idx;
      }

      time_vec.push_back(time);
      vel_vec.push_back(vel);
      acc_vec.push_back(acc);
      const double pedal = std::abs(throttle) > std::abs(brake) ? throttle : brake;
      pedal_vec.push_back(pedal);
    } catch(...) {
    }
    ++l_idx;
  }

  // make sampling_info
  const double max_vel = *std::max_element(vel_vec.begin(), vel_vec.end());
  const double min_vel = *std::min_element(vel_vec.begin(), vel_vec.end());
  const double max_pedal = *std::max_element(pedal_vec.begin(), pedal_vec.end());
  const double min_pedal = *std::min_element(pedal_vec.begin(), pedal_vec.end());

  SamplingInfo sampling_info;
  sampling_info.grid_num_vec = {10, 10};
  sampling_info.max_val_vec = {max_vel, max_pedal};
  sampling_info.min_val_vec = {min_vel, min_pedal};

  BayesianLinearRegression regressor(KernelType::RBF, sampling_info, 100, 10);

  // fit
  std::cout << "[fit] start" << std::endl;
  regressor.fit({vel_vec, pedal_vec}, {acc_vec});
  std::cout << "[fit] end" << std::endl;

  // predict
  std::cout << "[predict] start" << std::endl;
  const Eigen::MatrixXd predicted_state = regressor.predict(sampling_info.max_val_vec, sampling_info.min_val_vec);
  std::cout << "[predict] end" << std::endl;

  for (size_t r_idx = 0; r_idx < predicted_state.rows(); ++r_idx) {
    for (size_t c_idx = 0; c_idx < predicted_state.cols(); ++c_idx) {
      std::cout << predicted_state(r_idx, c_idx) << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}

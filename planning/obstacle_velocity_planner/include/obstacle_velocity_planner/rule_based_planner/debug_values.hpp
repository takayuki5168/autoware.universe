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

#ifndef ADAPTIVE_CRUISE_CONTROLLER__DEBUG_MARKER_HPP_
#define ADAPTIVE_CRUISE_CONTROLLER__DEBUG_MARKER_HPP_

#include <array>

class DebugValues
{
public:
  enum class TYPE {
                   // current
    CURRENT_VELOCITY = 0,
    CURRENT_ACCELERATION,
    CURRENT_JERK, // ignored
    // stop
    STOP_CURRENT_OBJECT_DISTANCE,
    STOP_CURRENT_OBJECT_VELOCITY,
    STOP_TARGET_OBJECT_DISTANCE,
    STOP_TARGET_VELOCITY, // ignored
    STOP_TARGET_ACCELERATION,
    STOP_TARGET_JERK, // ignored
    STOP_ERROR_OBJECT_DISTANCE,
    // slow down
    SLOW_DOWN_CURRENT_OBJECT_DISTANCE,
    SLOW_DOWN_CURRENT_OBJECT_VELOCITY,
    SLOW_DOWN_TARGET_OBJECT_DISTANCE,
    SLOW_DOWN_TARGET_VELOCITY,
    SLOW_DOWN_TARGET_ACCELERATION,
    SLOW_DOWN_TARGET_JERK,
    SLOW_DOWN_ERROR_OBJECT_DISTANCE,

    SIZE
  };

  /**
   * @brief get the index corresponding to the given value TYPE
   * @param [in] type the TYPE enum for which to get the index
   * @return index of the type
   */
  int getValuesIdx(const TYPE type) const { return static_cast<int>(type); }
  /**
   * @brief get all the debug values as an std::array
   * @return array of all debug values
   */
  std::array<double, static_cast<int>(TYPE::SIZE)> getValues() const { return values_; }
  /**
   * @brief set the given type to the given value
   * @param [in] type TYPE of the value
   * @param [in] value value to set
   */
  void setValues(const TYPE type, const double val) { values_.at(static_cast<int>(type)) = val; }
  /**
   * @brief set the given type to the given value
   * @param [in] type index of the type
   * @param [in] value value to set
   */
  void setValues(const int type, const double val) { values_.at(type) = val; }

  void resetValues() { values_.fill(0.0); }

private:
  static constexpr int num_debug_values_ = static_cast<int>(TYPE::SIZE);
  std::array<double, static_cast<int>(TYPE::SIZE)> values_;
};

#endif  // OBSTACLE_STOP_PLANNER__DEBUG_MARKER_HPP_

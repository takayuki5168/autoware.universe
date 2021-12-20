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
/*
 * This Code is inspired by code from https://github.com/LiJiangnanBit/path_optimizer
 *
 * MIT License
 *
 * Copyright (c) 2020 Li Jiangnan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef OBSTACLE_AVOIDANCE_PLANNER__OBSTACLE_FILTER_HPP_
#define OBSTACLE_AVOIDANCE_PLANNER__OBSTACLE_FILTER_HPP_

#include <vector>

class ObjectFilter
{
public:
  bool getAvoidingObjects();

private:
  bool isAvoidingObject(
    const PolygonPoints & polygon_points,
    const autoware_auto_perception_msgs::msg::PredictedObject & object,
    const cv::Mat & clearance_map, const nav_msgs::msg::MapMetaData & map_info,
    const std::vector<autoware_auto_planning_msgs::msg::PathPoint> & path_points,
    const TrajectoryParam & traj_param);

  bool isAvoidingObjectType(
    const autoware_auto_perception_msgs::msg::PredictedObject & object,
    const TrajectoryParam & traj_param);

  std::vector<autoware_auto_perception_msgs::msg::PredictedObject> avoiding_objects_;
  std::vector<autoware_auto_perception_msgs::msg::PredictedObject> non_avoiding_objects_;
  std::vector<autoware_auto_perception_msgs::msg::PredictedObject> unknown_objects_;
};

#endif  // OBSTACLE_AVOIDANCE_PLANNER__OBSTACLE_FILTER_HPP_

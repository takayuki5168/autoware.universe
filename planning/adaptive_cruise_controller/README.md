# Adaptive Cruise Controller

## Overview

## Input topics

| Name                        | Type                                            | Description       |
| --------------------------- | ----------------------------------------------- | ----------------- |
| `~/input/trajectory`        | autoware_auto_planning_msgs::Trajectory         | trajectory        |
| `~/input/odometry`          | nav_msgs::Odometry                              | vehicle velocity  |
| `~/input/predicted_objects` | autoware_auto_perception_msgs::PredictedObjects | dynamic objects   |

## Output topics

| Name                   | Type                                    | Description                            |
| ---------------------- | --------------------------------------- | -------------------------------------- |
| `~output/trajectory`   | autoware_auto_planning_msgs::Trajectory | trajectory to be followed              |
| `~output/stop_reasons` | tier4_planning_msgs::StopReasonArray    | reasons that cause the vehicle to stop |


## Role

## Flowchart

## Known Limits

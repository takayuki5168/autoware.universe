namespace
{
double AdaptiveCruiseController::getMedianVel(const std::vector<nav_msgs::msg::Odometry> vel_que)
{
  if (vel_que.size() == 0) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "size of vel que is 0. Something has wrong.");
    return 0.0;
  }

  std::vector<double> raw_vel_que;
  for (const auto vel : vel_que) {
    raw_vel_que.emplace_back(vel.twist.twist.linear.x);
  }

  double med_vel;
  if (raw_vel_que.size() % 2 == 0) {
    size_t med1 = (raw_vel_que.size()) / 2 - 1;
    size_t med2 = (raw_vel_que.size()) / 2;
    std::nth_element(raw_vel_que.begin(), raw_vel_que.begin() + med1, raw_vel_que.end());
    const double vel1 = raw_vel_que[med1];
    std::nth_element(raw_vel_que.begin(), raw_vel_que.begin() + med2, raw_vel_que.end());
    const double vel2 = raw_vel_que[med2];
    med_vel = (vel1 + vel2) / 2;
  } else {
    size_t med = (raw_vel_que.size() - 1) / 2;
    std::nth_element(raw_vel_que.begin(), raw_vel_que.begin() + med, raw_vel_que.end());
    med_vel = raw_vel_que[med];
  }

  return med_vel;
}
}  // namespace

class LidarBasedVelocityUpdater
{
public:
  LidarBasedVelocityUpdater()
  {
    // Subscribers
    detected_object_sub_ = this->create_subscription<DetectedObjects>(
      "~/input/detected_objects", 1,
      std::bind(&ObstacleStopPlannerNode::onDetectedObjects, this, std::placeholders::_1),
      createSubscriptionOptions(this));
    obstacle_pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "~/input/pointcloud", rclcpp::SensorDataQoS(),
      std::bind(&ObstacleStopPlannerNode::onObstaclePointcloud, this, std::placeholders::_1),
      createSubscriptionOptions(this));
  }

  void updateObjectVelocity()
  {
    if (!objects_ptr_ || !objects_ros_pointcloud_ptr_) {
      return;
    }

    // Update objects velocity
    const auto vehicle_objects = extractVehicleObjects(objects);
    calcObjectsBottomLine(vehicle_objects);
    const auto objects_with_updated_velocity = updateObjectsVelocity();

    // Publish objects
  }

private:
  struct ObjectBottomLine
  {
    ObjectBottomLine(
      const double x_arg, const double y_arg, const double theta_arg,
      const rclcpp::Time & ros_time_arg)
    : x(x_arg), y(y_arg), theta(theta_arg), ros_time(ros_time_arg)
    {
    }

    double x;
    double y;
    double theta;
    rclcpp::Time ros_time;
  };

  std::shared_ptr<sensor_msgs::msg::PointCloud2> objects_ros_pointcloud_ptr_;
  std::shared_ptr<DetectedObjects> objects_ptr_;

  // Callback functions
  void onDetectedObjects(const DetectedObjects::ConstSharedPtr input_msg)
  {
    objects_ptr_ = input_msg;
  }

  void onObjectsPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr input_msg)
  {
    objects_ros_pointcloud_ptr_ = std::make_shared<sensor_msgs::msg::PointCloud2>();

    pcl::VoxelGrid<pcl::PointXYZ> filter;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr no_height_pointcloud_ptr(
      new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr no_height_filtered_pointcloud_ptr(
      new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*input_msg, *pointcloud_ptr);

    for (const auto & point : pointcloud_ptr->points) {
      no_height_pointcloud_ptr->push_back(pcl::PointXYZ(point.x, point.y, 0.0));
    }
    filter.setInputCloud(no_height_pointcloud_ptr);
    filter.setLeafSize(0.05f, 0.05f, 100000.0f);
    filter.filter(*no_height_filtered_pointcloud_ptr);
    pcl::toROSMsg(*no_height_filtered_pointcloud_ptr, *obstacle_ros_pointcloud_ptr_);
    objects_ros_pointcloud_ptr_->header = input_msg->header;
  }

  DetectecObjects extractVechicleObjects(const DetectedObjects & detected_objects)
  {
    DetectecObjects detected_vehicle_objects;
    for (const auto & detected_object : detected_objects.objects) {
      const bool is_vehicle = isVehicle(detected_object);
      if (is_vehicle) {
        detected_vehicle_objects.push_back(detected_object);
      }
    }
    return detected_vehicle_objects;
  }

  void updateObjectVelocityFromPointCloud(
    const DetectedObjects & detected_vehicle_objects,
    const pcl::PointCloud<pcl::PointXYZ> & pointcloud)
  {
    const auto & pointcloud_stamp = pointcloud.header.stamp;

    for (size_t i = 0; i < pointcloud.size(); ++i) {
      const auto & x = pointcloud.at(i).x;
      const auto & y = pointcloud.at(i).y;

      const bool is_inside = isInsideDetectionArea(x, y);
      if (!is_inside) {
        continue;
      }

      for (size_t o_idx = 0; o_idx < detected_objects.objects.size(); ++o_idx) {
        const auto & object = detectec_objects.objects.at(o_idx);
        const auto object_polygon = createPolygon(object);
        const bool is_inside_object = isInsidePolygon(x, y, object_polygon);
        if (is_inside_object) {
          ObjectBottomLine();
        }
      }
    }

    geometry_msgs::msg::Point nearest_collision_p_ros;
    nearest_collision_p_ros.x = nearest_collision_point.x;
    nearest_collision_p_ros.y = nearest_collision_point.y;
    nearest_collision_p_ros.z = nearest_collision_point.z;

    /* estimate velocity */
    const double delta_time =
      nearest_collision_point_time.seconds() - prev_collision_point_time_.seconds();

    // if get same pointcloud with previous step,
    // skip estimate process
    if (std::fabs(delta_time) > std::numeric_limits<double>::epsilon()) {
      // valid time check
      if (delta_time < 0 || param_.valid_est_vel_diff_time < delta_time) {
        prev_collision_point_time_ = nearest_collision_point_time;
        prev_collision_point_ = nearest_collision_point;
        prev_collision_point_valid_ = true;
        return false;
      }
      const double p_dx = nearest_collision_point.x - prev_collision_point_.x;
      const double p_dy = nearest_collision_point.y - prev_collision_point_.y;
      const double p_dist = std::hypot(p_dx, p_dy);
      const double p_yaw = std::atan2(p_dy, p_dx);
      const double p_vel = p_dist / delta_time;
      const double est_velocity = p_vel * std::cos(p_yaw - traj_yaw);
      // valid velocity check
      if (est_velocity <= param_.valid_est_vel_min || param_.valid_est_vel_max <= est_velocity) {
        prev_collision_point_time_ = nearest_collision_point_time;
        prev_collision_point_ = nearest_collision_point;
        prev_collision_point_valid_ = true;
        est_vel_que_.clear();
        return false;
      }

      // append new velocity and remove old velocity from que
      registerQueToVelocity(est_velocity, nearest_collision_point_time);
    }

    // calc average(median) velocity from que
    *velocity = getMedianVel(est_vel_que_);
    debug_values_.data.at(DBGVAL::ESTIMATED_VEL_PCL) = *velocity;

    prev_collision_point_time_ = nearest_collision_point_time;
    prev_collision_point_ = nearest_collision_point;
    prev_target_velocity_ = *velocity;
    prev_collision_point_valid_ = true;
    return true;
  }
};

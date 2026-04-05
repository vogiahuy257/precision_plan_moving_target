#pragma once
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/quaternion.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <geometry_msgs/msg/vector3.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <std_msgs/msg/string.hpp>
class ArucoTrackerNode : public rclcpp::Node
{
public:
	ArucoTrackerNode();

private:
	void loadParameters();

	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void annotate_image(cv_bridge::CvImagePtr image, const cv::Vec3d& target);
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _image_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _kalman_reset_pub;
	// rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr _gimbal_sub;
	std::unique_ptr<cv::aruco::ArucoDetector> _detector;
	cv::Mat _camera_matrix;
	cv::Mat _dist_coeffs;

	int _param_aruco_id {};
	int _param_dictionary {};
	double _param_marker_size {};
	
	std::vector<cv::Point3f> _object_points;
    void updateMarkerGeometry();

	// Tracking state --- 
	rclcpp::Time _last_seen_time;
	bool _has_valid_pose = false;

	Eigen::Quaterniond _q_gimbal = Eigen::Quaterniond::Identity();
	bool _gimbal_valid = false;

    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr _odom_sub;
    bool _has_odom = false;
    Eigen::Vector3d _drone_pos_world;
    Eigen::Quaterniond _drone_q_world;
    Eigen::Vector3d _T_offset_body = Eigen::Vector3d(
		0.00,   // X forward (m)
		0.00,   // Y right
		0.01    // Z down
	);

};

#include "ArucoTracker.hpp"
#include <iomanip>
#include <px4_msgs/msg/vehicle_odometry.hpp>

constexpr double MARKER_TIMEOUT_SEC = 5.0;

ArucoTrackerNode::ArucoTrackerNode()
    : Node("aruco_tracker_node")
{
    // Setup QoS for real camera compatibility (Best Effort/Volatile)
    auto qos = rclcpp::SensorDataQoS();

    loadParameters();
    updateMarkerGeometry();

    cv::aruco::DetectorParameters detectorParams;
    detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    auto dictionary = cv::aruco::getPredefinedDictionary(_param_dictionary);
    _detector = std::make_unique<cv::aruco::ArucoDetector>(dictionary, detectorParams);

    // Subscribers
    std::string image_topic, camera_info_topic;

    get_parameter_or(
        "image_topic",
        image_topic,
        std::string("/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image"));

    get_parameter_or(
        "camera_info_topic",
        camera_info_topic,
        std::string("/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info"));

    RCLCPP_INFO(get_logger(), "Sub: %s | %s", image_topic.c_str(), camera_info_topic.c_str());

    _image_sub = create_subscription<sensor_msgs::msg::Image>(
        image_topic,
        qos,
        std::bind(&ArucoTrackerNode::image_callback, this, std::placeholders::_1));

    _camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic,
        qos,
        std::bind(&ArucoTrackerNode::camera_info_callback, this, std::placeholders::_1));

    // Publishers
    _image_pub = create_publisher<sensor_msgs::msg::Image>("/Aruco/image_proc", qos);
    _target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>("/Aruco/target_pose_FRD", qos);
    _kalman_reset_pub = create_publisher<std_msgs::msg::String>("/Aruco/target_state", qos);
}

void ArucoTrackerNode::loadParameters()
{
    declare_parameter<int>("aruco_id", 0);
    declare_parameter<int>("dictionary", 2);
    declare_parameter<double>("marker_size", 0.5);

    _param_aruco_id = get_parameter("aruco_id").as_int();
    _param_dictionary = get_parameter("dictionary").as_int();
    _param_marker_size = get_parameter("marker_size").as_double();
}

void ArucoTrackerNode::updateMarkerGeometry()
{
    float half_size = _param_marker_size / 2.0f;
    _object_points = {
        cv::Point3f(-half_size,  half_size, 0.0f),
        cv::Point3f( half_size,  half_size, 0.0f),
        cv::Point3f( half_size, -half_size, 0.0f),
        cv::Point3f(-half_size, -half_size, 0.0f)
    };
}

void ArucoTrackerNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
    _camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();
    _dist_coeffs = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double *>(msg->d.data())).clone();

    if (_camera_matrix.at<double>(0, 0) != 0.0)
    {
        RCLCPP_INFO(get_logger(), "Camera intrinsics received. Unsubscribing.");
        _camera_info_sub.reset();
    }
}

void ArucoTrackerNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (_camera_matrix.empty() || _dist_coeffs.empty())
    {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Waiting for camera calibration...");
        return;
    }

    try
    {
        // Dùng cùng clock của node để đồng bộ thời gian với KalmanFilter
        const rclcpp::Time nodeNow = now();

        cv_bridge::CvImagePtr cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        _detector->detectMarkers(cv_ptr->image, corners, ids);

        if (!ids.empty())
        {
            cv::aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);
        }

        bool found = false;

        for (size_t i = 0; i < ids.size(); ++i)
        {
            if (ids[i] != _param_aruco_id)
            {
                continue;
            }

            found = true;

            cv::Vec3d rvec, tvec;

            cv::solvePnP(_object_points, corners[i], _camera_matrix, _dist_coeffs, rvec, tvec);
            cv::drawFrameAxes(cv_ptr->image, _camera_matrix, _dist_coeffs, rvec, tvec, _param_marker_size);

            geometry_msgs::msg::PoseStamped pose_msg;

            // Không copy nguyên stamp của image nữa
            // Dùng time của node để KalmanFilter và ArucoTracker cùng hệ clock
            pose_msg.header.stamp = nodeNow;
            pose_msg.header.frame_id = msg->header.frame_id;

            Eigen::Vector3d p_cam(tvec[0], tvec[1], tvec[2]);
            Eigen::Vector3d p_cam_corrected = p_cam;

            pose_msg.pose.position.x = p_cam_corrected.x();
            pose_msg.pose.position.y = p_cam_corrected.y();
            pose_msg.pose.position.z = p_cam_corrected.z();

            cv::Mat rot_mat;
            cv::Rodrigues(rvec, rot_mat);
            cv::Quatd q_marker_cv = cv::Quatd::createFromRotMat(rot_mat).normalize();

            Eigen::Quaterniond q_marker(
                q_marker_cv.w,
                q_marker_cv.x,
                q_marker_cv.y,
                q_marker_cv.z);

            Eigen::Quaterniond q_marker_corrected = q_marker;
            q_marker_corrected.normalize();

            pose_msg.pose.orientation.x = q_marker_corrected.x();
            pose_msg.pose.orientation.y = q_marker_corrected.y();
            pose_msg.pose.orientation.z = q_marker_corrected.z();
            pose_msg.pose.orientation.w = q_marker_corrected.w();

            _target_pose_pub->publish(pose_msg);

            annotate_image(cv_ptr, tvec);

            _has_valid_pose = true;
            _last_seen_time = nodeNow;
            break;
        }

        // Timeout reset cũng dùng cùng clock của node
        if (_has_valid_pose && !found)
        {
            const rclcpp::Time currentTime = nodeNow;
            const double dt = (currentTime - _last_seen_time).seconds();

            if (dt > MARKER_TIMEOUT_SEC)
            {
                std_msgs::msg::String state_msg;
                state_msg.data = "RESET";
                _kalman_reset_pub->publish(state_msg);
                _has_valid_pose = false;
            }
        }
        else if (found)
        {
            std_msgs::msg::String state_msg;
            state_msg.data = "ACTIVE";
            _kalman_reset_pub->publish(state_msg);
        }

        _image_pub->publish(*cv_ptr->toImageMsg());
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(get_logger(), "CV Bridge error: %s", e.what());
    }
}

void ArucoTrackerNode::annotate_image(cv_bridge::CvImagePtr image, const cv::Vec3d &target)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << "XYZ: " << target;

    cv::putText(
        image->image,
        stream.str(),
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 255),
        2);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArucoTrackerNode>());
    rclcpp::shutdown();
    return 0;
}
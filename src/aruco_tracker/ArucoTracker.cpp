#include "ArucoTracker.hpp"
#include <iomanip>
#include <sstream>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <std_msgs/msg/string.hpp>

constexpr double MARKER_TIMEOUT_SEC = 5.0;

namespace
{
void publishArucoTiming(
    const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr &pub,
    double imageStampSec,
    double rxNowSec,
    double procStartSec,
    double procEndSec,
    double pubNowSec,
    bool found)
{
    if (!pub)
    {
        return;
    }

    std_msgs::msg::String msg;
    std::ostringstream ss;

    const double rxWaitDt = rxNowSec - imageStampSec;
    const double queueBeforeProcDt = procStartSec - rxNowSec;
    const double processingDt = procEndSec - procStartSec;
    const double sendDt = pubNowSec - procEndSec;
    const double totalNodeDt = pubNowSec - imageStampSec;

    ss << std::fixed << std::setprecision(6)
       << "{"
       << "\"node\":\"aruco\","
       << "\"image_stamp\":" << imageStampSec << ","
       << "\"rx_now\":" << rxNowSec << ","
       << "\"proc_start\":" << procStartSec << ","
       << "\"proc_end\":" << procEndSec << ","
       << "\"pub_now\":" << pubNowSec << ","
       << "\"rx_wait_dt\":" << rxWaitDt << ","
       << "\"queue_before_proc_dt\":" << queueBeforeProcDt << ","
       << "\"processing_dt\":" << processingDt << ","
       << "\"send_dt\":" << sendDt << ","
       << "\"total_node_dt\":" << totalNodeDt << ","
       << "\"found\":" << (found ? 1 : 0)
       << "}";

    msg.data = ss.str();
    pub->publish(msg);
}
}

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

    // Debug timing publisher
    _debug_dt_pub = create_publisher<std_msgs::msg::String>(
        "/debug_dt/aruco",
        rclcpp::QoS(10).best_effort());
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
        const rclcpp::Time rxNow = now();
        const rclcpp::Time procStart = now();

        cv_bridge::CvImagePtr cv_ptr =
            cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        _detector->detectMarkers(cv_ptr->image, corners, ids);

        if (!ids.empty())
        {
            cv::aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);
        }

        const rclcpp::Time imageStamp = msg->header.stamp;
        const double imageStampSec = imageStamp.seconds();
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
            pose_msg.header.stamp = msg->header.stamp;
            pose_msg.header.frame_id = msg->header.frame_id;

            Eigen::Vector3d p_cam(tvec[0], tvec[1], tvec[2]);
            pose_msg.pose.position.x = p_cam.x();
            pose_msg.pose.position.y = p_cam.y();
            pose_msg.pose.position.z = p_cam.z();

            cv::Mat rot_mat;
            cv::Rodrigues(rvec, rot_mat);
            cv::Quatd q_marker_cv = cv::Quatd::createFromRotMat(rot_mat).normalize();

            Eigen::Quaterniond q_marker(
                q_marker_cv.w,
                q_marker_cv.x,
                q_marker_cv.y,
                q_marker_cv.z);

            q_marker.normalize();

            pose_msg.pose.orientation.x = q_marker.x();
            pose_msg.pose.orientation.y = q_marker.y();
            pose_msg.pose.orientation.z = q_marker.z();
            pose_msg.pose.orientation.w = q_marker.w();

            annotate_image(cv_ptr, tvec);

            const rclcpp::Time procEnd = now();
            const rclcpp::Time pubNow = now();

            publishArucoTiming(
                _debug_dt_pub,
                imageStampSec,
                rxNow.seconds(),
                procStart.seconds(),
                procEnd.seconds(),
                pubNow.seconds(),
                true);

            _target_pose_pub->publish(pose_msg);

            _has_valid_pose = true;
            _last_seen_time = pubNow;
            break;
        }

        if (!found)
        {
            const rclcpp::Time procEnd = now();
            const rclcpp::Time pubNow = procEnd;

            publishArucoTiming(
                _debug_dt_pub,
                imageStampSec,
                rxNow.seconds(),
                procStart.seconds(),
                procEnd.seconds(),
                pubNow.seconds(),
                false);
        }

        if (_has_valid_pose && !found)
        {
            const rclcpp::Time currentTime = now();
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
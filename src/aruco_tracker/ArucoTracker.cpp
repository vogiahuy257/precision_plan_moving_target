
#include "ArucoTracker.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <std_msgs/msg/string.hpp>


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
    auto qos = rclcpp::SensorDataQoS();

    loadParameters();
    updateMarkerGeometry();

    cv::aruco::DetectorParameters detectorParams;
    detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    auto dictionary = cv::aruco::getPredefinedDictionary(_param_dictionary);
    _detector = std::make_unique<cv::aruco::ArucoDetector>(dictionary, detectorParams);

    std::string image_topic, camera_info_topic;

    get_parameter_or(
        "image_topic",
        image_topic,
        std::string("/camera/image"));

    get_parameter_or(
        "camera_info_topic",
        camera_info_topic,
        std::string("/camera/camera_info"));

    _image_sub = create_subscription<sensor_msgs::msg::Image>(
        image_topic,
        qos,
        std::bind(&ArucoTrackerNode::image_callback, this, std::placeholders::_1));

    _camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
        camera_info_topic,
        qos,
        std::bind(&ArucoTrackerNode::camera_info_callback, this, std::placeholders::_1));

    _image_pub = create_publisher<sensor_msgs::msg::Image>("/Aruco/image_proc", qos);
    _target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>("/Aruco/target_pose_optical", qos);

    const auto state_qos = rclcpp::QoS(10).reliable();
    _kalman_reset_pub = create_publisher<std_msgs::msg::String>("/Aruco/target_state", state_qos);

    _debug_dt_pub = create_publisher<std_msgs::msg::String>(
        "/debug_dt/aruco",
        rclcpp::QoS(10).best_effort());
}

void ArucoTrackerNode::loadParameters()
{
    // declare_parameter<int>("aruco_id", 7);
    // declare_parameter<int>("dictionary", 4);
    // declare_parameter<double>("marker_size", 0.28);
    declare_parameter<int>("aruco_id", 10);
    declare_parameter<int>("dictionary", 5);
    declare_parameter<double>("marker_size", 0.08);
    declare_parameter<double>("lost_reset_sec", 5.0);

    _param_aruco_id = get_parameter("aruco_id").as_int();
    _param_dictionary = get_parameter("dictionary").as_int();
    _param_marker_size = get_parameter("marker_size").as_double();
    _param_lost_reset_sec = get_parameter("lost_reset_sec").as_double();
}

void ArucoTrackerNode::updateMarkerGeometry()
{
    const float half_size = static_cast<float>(_param_marker_size / 2.0);

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
        _camera_info_sub.reset();
    }
}

void ArucoTrackerNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    if (_camera_matrix.empty() || _dist_coeffs.empty())
    {
        return;
    }

    const rclcpp::Time rxNow = now();
    const rclcpp::Time procStart = now();

    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (const cv_bridge::Exception &)
    {
        return;
    }

    const cv::Mat &gray = cv_ptr->image;

    if (gray.empty())
    {
        return;
    }

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;

    _detector->detectMarkers(gray, corners, ids);

    bool found = false;
    int target_index = -1;

    for (size_t i = 0; i < ids.size(); ++i)
    {
        if (ids[i] == _param_aruco_id)
        {
            found = true;
            target_index = static_cast<int>(i);
            break;
        }
    }

    rclcpp::Time imageStamp = msg->header.stamp;
    if (imageStamp.nanoseconds() == 0)
    {
        imageStamp = now();
    }

    const double imageStampSec = imageStamp.seconds();

    cv::Vec3d target_tvec(0.0, 0.0, 0.0);

    if (found && target_index >= 0)
    {
        cv::Vec3d rvec;

        cv::solvePnP(
            _object_points,
            corners[target_index],
            _camera_matrix,
            _dist_coeffs,
            rvec,
            target_tvec);

        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp = msg->header.stamp;
        pose_msg.header.frame_id = msg->header.frame_id;

        Eigen::Vector3d p_optical(
            target_tvec[0],
            target_tvec[1],
            target_tvec[2]);

        pose_msg.pose.position.x = p_optical.x();
        pose_msg.pose.position.y = p_optical.y();
        pose_msg.pose.position.z = p_optical.z();

        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);

        cv::Quatd q_marker_cv =
            cv::Quatd::createFromRotMat(rot_mat).normalize();

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

        _target_pose_pub->publish(pose_msg);

        _has_valid_pose = true;
        _target_lost = false;
        _reset_sent = false;

        std_msgs::msg::String state_msg;
        state_msg.data = "ACTIVE";
        _kalman_reset_pub->publish(state_msg);
    }

    if (!found)
    {
        // First missed frame after a valid target: announce LOST once.
        if (_has_valid_pose)
        {
            std_msgs::msg::String state_msg;
            state_msg.data = "LOST";
            _kalman_reset_pub->publish(state_msg);

            _has_valid_pose = false;
            _target_lost = true;
            _reset_sent = false;
            _lost_since = imageStamp;
        }

        // Keep LOST for the requested grace period, then send RESET once.
        if (_target_lost && !_reset_sent)
        {
            const double lost_time_sec =
                (imageStamp - _lost_since).seconds();

            if (std::isfinite(lost_time_sec) &&
                lost_time_sec >= _param_lost_reset_sec)
            {
                std_msgs::msg::String state_msg;
                state_msg.data = "RESET";
                _kalman_reset_pub->publish(state_msg);

                _reset_sent = true;
            }
        }
    }

    const rclcpp::Time procEnd = now();

    if (_image_pub && _image_pub->get_subscription_count() > 0)
    {
        cv::Mat debug_image = gray.clone();

        if (found && target_index >= 0)
        {
            const auto &target_corners = corners[target_index];

            std::vector<cv::Point> contour;
            contour.reserve(4);

            cv::Point2f center(0.0f, 0.0f);

            for (const auto &p : target_corners)
            {
                contour.emplace_back(
                    static_cast<int>(std::round(p.x)),
                    static_cast<int>(std::round(p.y)));

                center += p;
            }

            center *= 0.25f;

            const int cx = static_cast<int>(std::round(center.x));
            const int cy = static_cast<int>(std::round(center.y));

            cv::polylines(
                debug_image,
                contour,
                true,
                cv::Scalar(255),
                2,
                cv::LINE_AA);

            cv::circle(
                debug_image,
                cv::Point(cx, cy),
                4,
                cv::Scalar(255),
                -1,
                cv::LINE_AA);

            std::ostringstream label;
            label << std::fixed << std::setprecision(3)
                  << "ID:" << ids[target_index]
                  << " X:" << target_tvec[0]
                  << " Y:" << target_tvec[1]
                  << " Z:" << target_tvec[2] << "m";

            const std::string text = label.str();

            int baseline = 0;
            const double font_scale = 0.55;
            const int thickness = 1;

            const cv::Size text_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
                &baseline);

            int tx = cx + 8;
            int ty = cy - 8;

            if (tx + text_size.width + 4 >= debug_image.cols)
            {
                tx = cx - text_size.width - 8;
            }

            if (ty - text_size.height - 4 < 0)
            {
                ty = cy + text_size.height + 12;
            }

            cv::rectangle(
                debug_image,
                cv::Point(tx - 2, ty - text_size.height - 2),
                cv::Point(tx + text_size.width + 2, ty + baseline + 2),
                cv::Scalar(0),
                -1);

            cv::putText(
                debug_image,
                text,
                cv::Point(tx, ty),
                cv::FONT_HERSHEY_SIMPLEX,
                font_scale,
                cv::Scalar(255),
                thickness,
                cv::LINE_AA);
        }

        cv_bridge::CvImage debug_msg;
        debug_msg.header = msg->header;
        debug_msg.encoding = sensor_msgs::image_encodings::MONO8;
        debug_msg.image = debug_image;

        _image_pub->publish(*debug_msg.toImageMsg());
    }

    const rclcpp::Time pubNow = now();

    publishArucoTiming(
        _debug_dt_pub,
        imageStampSec,
        rxNow.seconds(),
        procStart.seconds(),
        procEnd.seconds(),
        pubNow.seconds(),
        found);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArucoTrackerNode>());
    rclcpp::shutdown();

    return 0;
}

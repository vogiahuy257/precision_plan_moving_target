#pragma once

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_command_ack.hpp>

#include "ControlTypes.hpp"

namespace precision_land
{
class DisarmController
{
public:
    void configure(
        const DisarmControllerParams &params,
        rclcpp::Node *node,
        const rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr &vehicleCommandPub);

    void reset();

    DisarmControllerOutput update(const DisarmControllerInput &input);
    DisarmDecisionStatus handleAck(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg);

    DisarmDecisionStatus status() const;

private:
    float selectAltitude(const DisarmControllerInput &input, bool &isValid) const;
    bool shouldDisarm(const DisarmControllerInput &input, float &selectedAltitude, bool &selectedAltitudeValid) const;
    bool sendDisarmCommand();
    void publishVehicleCommand(uint16_t command, float param1, float param2);

private:
    DisarmControllerParams params_{};

    rclcpp::Node *node_{nullptr};
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicleCommandPub_{nullptr};

    bool disarmSent_{false};
    bool waitingAck_{false};
    rclcpp::Time disarmRequestTime_{0, 0, RCL_ROS_TIME};
    DisarmDecisionStatus status_{DisarmDecisionStatus::Idle};
};
}
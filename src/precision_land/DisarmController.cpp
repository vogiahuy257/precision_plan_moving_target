#include "DisarmController.hpp"

#include <cmath>

namespace precision_land
{
namespace
{
constexpr double kRetryCooldownSec = 0.1;
}

void DisarmController::configure(
    const DisarmControllerParams &params,
    rclcpp::Node *node,
    const rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr &vehicleCommandPub)
{
    params_ = params;
    node_ = node;
    vehicleCommandPub_ = vehicleCommandPub;
    reset();
}

void DisarmController::reset()
{
    disarmSent_ = false;
    waitingAck_ = false;
    status_ = DisarmDecisionStatus::Idle;

    if (node_ != nullptr)
    {
        disarmRequestTime_ = rclcpp::Time(0, 0, node_->get_clock()->get_clock_type());
    }
}

float DisarmController::selectAltitude(const DisarmControllerInput &input, bool &isValid) const
{
    switch (params_.altitudeSource)
    {
    case DisarmAltitudeSource::DistBottom:
    {
        isValid = input.distBottomValid;
        return input.distBottom;
    }

    case DisarmAltitudeSource::LocalPositionZ:
    {
        isValid = input.localPositionZValid;
        return std::abs(input.localPositionZ);
    }

    default:
    {
        isValid = false;
        return 0.0f;
    }
    }
}

bool DisarmController::shouldDisarm(
    const DisarmControllerInput &input,
    float &selectedAltitude,
    bool &selectedAltitudeValid) const
{
    if (params_.mode == DisarmMode::Disabled)
    {
        selectedAltitude = 0.0f;
        selectedAltitudeValid = false;
        return false;
    }

    if (input.landed && params_.allowLandedImmediateDisarm)
    {
        selectedAltitude = 0.0f;
        selectedAltitudeValid = true;
        return true;
    }

    selectedAltitude = selectAltitude(input, selectedAltitudeValid);

    if (!selectedAltitudeValid)
    {
        return false;
    }

    const bool heightOk = selectedAltitude <= params_.disarmHeight;
    // const bool lateralOk = input.lateralError <= params_.lateralErrorThreshold;
    // const bool vzOk = input.verticalSpeedAbs <= params_.verticalSpeedThreshold;

    return heightOk;//&& lateralOk && vzOk;
}

DisarmControllerOutput DisarmController::update(const DisarmControllerInput &input)
{
    DisarmControllerOutput output{};
    output.status = status_;
    output.shouldSendDisarm = false;

    float selectedAltitude = 0.0f;
    bool selectedAltitudeValid = false;

    const bool allowDisarmNow = shouldDisarm(input, selectedAltitude, selectedAltitudeValid);

    output.selectedAltitude = selectedAltitude;
    output.selectedAltitudeValid = selectedAltitudeValid;

    if (params_.mode == DisarmMode::Disabled)
    {
        status_ = DisarmDecisionStatus::Disabled;
        output.status = status_;
        return output;
    }

    if (status_ == DisarmDecisionStatus::Accepted)
    {
        output.status = status_;
        return output;
    }

    if (!allowDisarmNow)
    {
        waitingAck_ = false;
        status_ = DisarmDecisionStatus::Blocked;
        output.status = status_;
        return output;
    }

    if (node_ == nullptr)
    {
        status_ = DisarmDecisionStatus::Rejected;
        output.status = status_;
        return output;
    }

    const double dtFromLastRequest = (node_->now() - disarmRequestTime_).seconds();

    // Gui lan dau neu chua gui lan nao
    if (!disarmSent_)
    {
        const bool sendOk = sendDisarmCommand();
        output.shouldSendDisarm = sendOk;
        output.status = status_;
        return output;
    }

    // Neu dang cho ACK hoac da bi reject, van retry lien tuc theo chu ky ngan
    if (dtFromLastRequest >= kRetryCooldownSec)
    {
        const bool sendOk = sendDisarmCommand();
        output.shouldSendDisarm = sendOk;
        output.status = status_;
        return output;
    }

    status_ = waitingAck_ ? DisarmDecisionStatus::WaitingAck : status_;
    output.status = status_;
    return output;
}

bool DisarmController::sendDisarmCommand()
{
    if (node_ == nullptr || vehicleCommandPub_ == nullptr)
    {
        status_ = DisarmDecisionStatus::Rejected;
        return false;
    }

    try
    {
        publishVehicleCommand(
            px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM,
            static_cast<float>(px4_msgs::msg::VehicleCommand::ARMING_ACTION_DISARM),
            0.0f);

        disarmSent_ = true;
        waitingAck_ = true;
        disarmRequestTime_ = node_->now();
        status_ = DisarmDecisionStatus::WaitingAck;

        RCLCPP_WARN(node_->get_logger(), "[DisarmController] Da gui lenh DISARM, tiep tuc retry den khi ACCEPTED");
        return true;
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(node_->get_logger(), "[DisarmController] Exception khi gui DISARM: %s", e.what());
        status_ = DisarmDecisionStatus::Rejected;
        return false;
    }
    catch (...)
    {
        RCLCPP_ERROR(node_->get_logger(), "[DisarmController] Unknown exception khi gui DISARM");
        status_ = DisarmDecisionStatus::Rejected;
        return false;
    }
}

void DisarmController::publishVehicleCommand(uint16_t command, float param1, float param2)
{
    px4_msgs::msg::VehicleCommand msg{};
    msg.timestamp = node_->now().nanoseconds() / 1000;
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.confirmation = 0;
    msg.from_external = true;

    vehicleCommandPub_->publish(msg);

    RCLCPP_INFO(
        node_->get_logger(),
        "[DisarmController] Publish VehicleCommand: command=%u param1=%.3f param2=%.3f",
        static_cast<unsigned>(command),
        static_cast<double>(param1),
        static_cast<double>(param2));
}

DisarmDecisionStatus DisarmController::handleAck(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return status_;
    }

    if (msg->command != px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM)
    {
        return status_;
    }

    if (node_ != nullptr)
    {
        RCLCPP_WARN(
            node_->get_logger(),
            "[DisarmController] Nhan ACK DISARM: command=%u result=%u from_external=%d",
            static_cast<unsigned>(msg->command),
            static_cast<unsigned>(msg->result),
            static_cast<int>(msg->from_external));
    }

    if (msg->result == px4_msgs::msg::VehicleCommandAck::VEHICLE_CMD_RESULT_ACCEPTED)
    {
        waitingAck_ = false;
        status_ = DisarmDecisionStatus::Accepted;
    }
    else
    {
        // Khong Accepted thi de update() tiep tuc retry
        waitingAck_ = false;
        status_ = DisarmDecisionStatus::Rejected;
        disarmRequestTime_ = node_ != nullptr ? node_->now() : disarmRequestTime_;
    }

    return status_;
}

DisarmDecisionStatus DisarmController::status() const
{
    return status_;
}
} // namespace precision_land
#!/usr/bin/env python3

import subprocess
import signal
import sys
import time
import os

TOPIC = "/land_pad/cmd_vel"
RATE = 10
DURATION = 100
SPEED = 3.0

current_process = None
stop_requested = False


def stop_robot() -> None:
    print("Dung robot")

    msg = (
        "{linear: {x: 0.0, y: 0.0, z: 0.0}, "
        "angular: {x: 0.0, y: 0.0, z: 0.0}}"
    )

    cmd = [
        "ros2",
        "topic",
        "pub",
        TOPIC,
        "geometry_msgs/msg/Twist",
        msg,
        "--wait-matching-subscriptions",
        "0",
        "--once",
    ]

    try:
        subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Loi khi dung robot: {e}")


def kill_current_process() -> None:
    global current_process

    if current_process is None:
        return

    try:
        if current_process.poll() is None:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
            time.sleep(0.1)

            if current_process.poll() is None:
                os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
    except Exception as e:
        print(f"Loi khi kill process: {e}")
    finally:
        current_process = None


def signal_handler(signum, frame) -> None:
    global stop_requested
    stop_requested = True

    print("\nNhan Ctrl+C -> thoat ngay")
    kill_current_process()

    # Thoat ngay, khong cho chay tiep finally
    os._exit(0)


def publish_segment(vx: float, vy: float, vz: float, wz: float) -> None:
    global current_process, stop_requested

    if stop_requested:
        return

    print(f"Dang chay: vx={vx}, vy={vy}, vz={vz}, wz={wz} trong {DURATION}s")

    msg = (
        f"{{linear: {{x: {vx}, y: {vy}, z: {vz}}}, "
        f"angular: {{x: 0.0, y: 0.0, z: {wz}}}}}"
    )

    cmd = [
        "ros2",
        "topic",
        "pub",
        TOPIC,
        "geometry_msgs/msg/Twist",
        msg,
        "--wait-matching-subscriptions",
        "0",
        "-r",
        str(RATE),
    ]

    try:
        current_process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.monotonic()

        while True:
            if stop_requested:
                kill_current_process()
                return

            if current_process.poll() is not None:
                break

            elapsed = time.monotonic() - start_time
            if elapsed >= DURATION:
                kill_current_process()
                break

            time.sleep(0.05)

    except Exception as e:
        print(f"Loi khi publish segment: {e}")
        kill_current_process()


def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not stop_requested:

            publish_segment(0.0, SPEED, 0.0, 0.0)
            if stop_requested:
                break
            time.sleep(0.5)

            publish_segment(SPEED, 0.0, 0.0, 0.0)
            if stop_requested:
                break
            time.sleep(0.5)

            publish_segment(0.0, -SPEED, 0.0, 0.0)
            if stop_requested:
                break
            time.sleep(0.5)

            publish_segment(-SPEED, 0.0, 0.0, 0.0)
            if stop_requested:
                break
            time.sleep(0.5)

            print("Hoan thanh 1 vong hinh vuong")
    finally:
        kill_current_process()


if __name__ == "__main__":
    main()
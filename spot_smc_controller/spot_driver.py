#!/usr/bin/env python3
"""
Spot extern controller - chß║íy b├¬n trong Webots, giao tiß║┐p vß╗¢i ROS2
"""
import sys
import os
sys.path.insert(0, '/usr/local/webots/lib/controller/python')

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from controller import Robot

MOTOR_NAMES = [
    "front left shoulder abduction motor",
    "front left shoulder rotation motor",
    "front left elbow motor",
    "front right shoulder abduction motor",
    "front right shoulder rotation motor",
    "front right elbow motor",
    "rear left shoulder abduction motor",
    "rear left shoulder rotation motor",
    "rear left elbow motor",
    "rear right shoulder abduction motor",
    "rear right shoulder rotation motor",
    "rear right elbow motor",
]

ROS_NAMES = [
    "front_left_shoulder_abduction", "front_left_shoulder_rotation", "front_left_elbow",
    "front_right_shoulder_abduction", "front_right_shoulder_rotation", "front_right_elbow",
    "rear_left_shoulder_abduction", "rear_left_shoulder_rotation", "rear_left_elbow",
    "rear_right_shoulder_abduction", "rear_right_shoulder_rotation", "rear_right_elbow",
]


class SpotROS2Controller(Node):
    def __init__(self, robot):
        super().__init__("spot_controller")
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())

        # Khß╗ƒi tß║ío motors
        self.motors = []
        for name in MOTOR_NAMES:
            m = robot.getDevice(name)
            # Chß║┐ ─æß╗Ö torque control: setPosition(inf) + setVelocity(0)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motors.append(m)

        # Khß╗ƒi tß║ío position sensors (enable feedback)
        self.sensors = []
        for name in MOTOR_NAMES:
            sensor_name = name.replace("motor", "sensor")
            s = robot.getDevice(sensor_name)
            if s is not None:
                s.enable(self.timestep)
                self.get_logger().info(f"Sensor enabled: {sensor_name}")
            else:
                self.get_logger().warn(f"Sensor NOT found: {sensor_name}")
            self.sensors.append(s)

        # Chß║┐ ─æß╗Ö mß║╖c ─æß╗ïnh: position control
        self.control_mode = "position"
        self.target_positions = [0.0] * 12
        # Set t╞░ thß║┐ ─æß╗⌐ng ban ─æß║ºu
        stand_pose = [
            -0.1,  0.0,  0.0,   # FL
            -0.1,  0.0,  0.0,   # FR
            -0.1,  0.0,  0.0,   # RL
            -0.1,  0.0,  0.0,   # RR
        ]
        for i, m in enumerate(self.motors):
            m.setPosition(stand_pose[i])
        self.target_positions = stand_pose

        # Publisher
        self.js_pub = self.create_publisher(JointState, "/Spot/joint_states", 10)

        # Subscribers
        self.create_subscription(
            Float64MultiArray, "/Spot/joint_position_commands",
            self.position_cmd_cb, 10)
        self.create_subscription(
            Float64MultiArray, "/Spot/joint_torque_commands",
            self.torque_cmd_cb, 10)

        self.get_logger().info("SpotROS2Controller ready!")

    def position_cmd_cb(self, msg):
        if len(msg.data) == 12:
            for i, m in enumerate(self.motors):
                m.setPosition(msg.data[i])
            self.target_positions = list(msg.data)

    def torque_cmd_cb(self, msg):
        """Torque control cho SMC"""
        if len(msg.data) == 12:
            for i, m in enumerate(self.motors):
                m.setTorque(msg.data[i])

    def publish_joint_states(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ROS_NAMES
        positions = []
        velocities = []
        for i, s in enumerate(self.sensors):
            if s is not None:
                positions.append(s.getValue())
                velocities.append(self.motors[i].getVelocity())
            else:
                positions.append(self.target_positions[i])
                velocities.append(0.0)
        js.position = positions
        js.velocity = velocities
        js.effort = [0.0] * 12
        self.js_pub.publish(js)


def main():
    rclpy.init()
    robot = Robot()
    node = SpotROS2Controller(robot)
    timestep = int(robot.getBasicTimeStep())

    while robot.step(timestep) != -1:
        rclpy.spin_once(node, timeout_sec=0)
        node.publish_joint_states()

    rclpy.shutdown()


if __name__ == "__main__":
    main()


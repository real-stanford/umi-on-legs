import logging
from os import system
import time
from api.utils.velocity_estimator import MovingWindowFilter, VelocityEstimator
import rclpy
from rclpy.node import Node
import rclpy.publisher
from modules.pose_estimator import MotionEstimator
from unitree_go.msg import (
    LowState,
    WirelessController,
)
from transforms3d import quaternions, affines, axangles
import numpy as np
from modules.common import (
    LEG_DOF,
)
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from rclpy.clock import ClockType
from robot_state.msg import PoseMultiStamped


def publish_pose(
    publisher: rclpy.publisher.Publisher,
    tick: int,
    pos: np.ndarray,
    rot_quat: np.ndarray,
):
    # msg.tick is in milliseconds
    t = Time(nanoseconds=tick * 1e6, clock_type=ClockType.STEADY_TIME)
    msg = PoseStamped()
    msg.header.stamp = t.to_msg()
    # convert to rcl time object
    msg.pose.position.x = pos[0]
    msg.pose.position.y = pos[1]
    msg.pose.position.z = pos[2]
    msg.pose.orientation.w = rot_quat[0]
    msg.pose.orientation.x = rot_quat[1]
    msg.pose.orientation.y = rot_quat[2]
    msg.pose.orientation.z = rot_quat[3]
    publisher.publish(msg)


def publish_multistamped_pose(
    publisher: rclpy.publisher.Publisher,
    tick: int,
    system_time: float,
    ros_time: float,
    source_time: float,
    pos: np.ndarray,  # x, y, z
    rot_quat: np.ndarray,  # qx, qy, qz, qw
):
    msg = PoseMultiStamped()
    msg.tick = tick
    msg.system_time = system_time
    msg.ros_time = ros_time
    msg.source_time = source_time
    msg.pose[:3] = pos
    msg.pose[3:] = rot_quat
    publisher.publish(msg)


class PoseEstimationNode(Node):
    def __init__(self, pose_smoothing_history: int = 1, warning_freq: float = 100.0):
        super().__init__("pose_estimator_node")  # type: ignore
        self.warning_freq = warning_freq
        self.robot_pose_pub = self.create_publisher(
            PoseStamped, "motion_estimator/robot_pose", 1
        )
        self.iphone_transformed_pose_pub = self.create_publisher(
            PoseMultiStamped, "motion_estimator/iphone_transformed_pose", 1
        )
        self.imu_integrated_pose_pub = self.create_publisher(
            PoseMultiStamped, "motion_estimator/imu_integrated_pose", 1
        )
        self.pose_smoothing_history = pose_smoothing_history
        self.robot_position_filter = MovingWindowFilter(
            window_size=pose_smoothing_history, data_dim=3
        )
        self.robot_orientation_axangle_filter = MovingWindowFilter(
            window_size=pose_smoothing_history, data_dim=3
        )  # TODO: smaller window size for lower latency?
        # Motion Estimation
        self.motion_estimator = MotionEstimator(
            linear_velocity_estimator=VelocityEstimator(
                hip_length=0.0955,
                thigh_length=0.213,
                calf_length=0.213,
                default_control_dt=0.005,
            ),
            angular_velocity_filter=MovingWindowFilter(
                window_size=pose_smoothing_history, data_dim=3
            ),
            accelerometer_filter=MovingWindowFilter(
                window_size=pose_smoothing_history, data_dim=3
            ),
            robot2imu=affines.compose(
                T=np.array([-0.02557, 0.0, 0.04232]), R=np.identity(3), Z=np.ones(3)
            ),
            low_level_state_dt=0.005,
            # pose_latency=0.140,
            pose_latency=0.0,
            buffer_multiplier=4.0,
            base2iphone=affines.compose(
                # T=np.array([0.040048, -0.087285, 0.157327]),
                # 90 degree mount
                # T=np.array([0.040048, -0.157327, 0.087285]),
                # R=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                # 60 degree mount
                T=np.array([0.036827,  -0.08479177,  0.17114034]),
                R=np.array(
                    [[ 6.93889399e-18 , 1.00000000e+00  ,1.20185167e-17],
                    [ 5.00000007e-01, -2.08166820e-17,  8.66025400e-01],
                    [ 8.66025400e-01, -3.60555502e-17, -5.00000007e-01]]
                ),
                Z=np.ones(3),
            ),
        )
        self.motion_bias_time = -1.0

        self.lowlevel_state_sub = self.create_subscription(
            LowState, "lowstate", self.lowlevel_state_cb, 1
        )  # "/lowcmd" or  "lf/lowstate" (low frequencies)
        self.lowlevel_state_sub  # prevent unused variable warning
        self.foot_contact_thres = 20.0

        self.joy_stick_sub = self.create_subscription(
            WirelessController,
            "wirelesscontroller",
            self.joy_stick_cb,
            1,
        )
        self.joy_stick_sub  # prevent unused variable warning
        self.prev_state_time = time.monotonic()
        self.key_is_pressed = False
        self.prev_iphone_pub_timestamp = -1.0

    def joy_stick_cb(self, msg):
        if msg.keys == int(2**14):  # Down: reset accelerometer
            if not self.key_is_pressed:
                self.get_logger().info("Resetting motion bias")
                self.motion_estimator.set_motion_bias()
                self.motion_bias_time = time.monotonic()
            self.key_is_pressed = True
        if msg.keys == int(2**13):  # Right
            if not self.key_is_pressed:
                self.motion_estimator.pose_latency = min(
                    0.200, self.motion_estimator.pose_latency + 0.010
                )
                self.get_logger().info(
                    f"incremented pose latency to {self.motion_estimator.pose_latency:.03f}s"
                )
            self.key_is_pressed = True
        if msg.keys == int(2**15):  # Left
            # pass
            if not self.key_is_pressed:
                self.motion_estimator.pose_latency = max(
                    0.0, self.motion_estimator.pose_latency - 0.010
                )
                self.get_logger().info(
                    f"decremented pose latency to {self.motion_estimator.pose_latency:.03f}s"
                )
            self.key_is_pressed = True
        if self.key_is_pressed:
            # Prevent multiple key presses
            if msg.keys == 0:
                self.key_is_pressed = False

    # @profile
    def lowlevel_state_cb(self, msg: LowState):
        # imu data
        imu_data = msg.imu_state

        self.state_q = np.array(
            [motor_data.q for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.state_dq = np.array(
            [motor_data.dq for motor_data in msg.motor_state[:LEG_DOF]]
        )
        self.state_tau = np.array(
            [motor_data.tau_est for motor_data in msg.motor_state[:LEG_DOF]]
        )
        acceleration = np.array(imu_data.accelerometer, dtype=np.float64)
        quaternion = np.array(imu_data.quaternion, dtype=np.float64)
        foot_force = np.array(
            [msg.foot_force[foot_id] for foot_id in range(4)], dtype=np.float64
        )
        foot_contact = np.array(foot_force > self.foot_contact_thres, dtype=np.float64)
        self.motion_estimator.update_velocity(
            timestamp_s=float(msg.tick) / 1000,
            acceleration=acceleration,
            gyroscope=np.array(imu_data.gyroscope, dtype=np.float64),
            foot_contact=foot_contact,
            quaternion=quaternion,
            joint_velocity=self.state_dq,
            joint_position=self.state_q,
        )
        self.motion_estimator.update_pose()

        raw_robot_pose = self.motion_estimator.pose.copy()
        raw_robot_pos = raw_robot_pose[:3, 3]
        raw_robot_rot_mat = raw_robot_pose[:3, :3]
        raw_robot_quat = quaternions.mat2quat(raw_robot_rot_mat)

        if self.pose_smoothing_history > 1:
            smoothed_robot_pos = self.robot_position_filter.calculate_average(
                raw_robot_pos
            )
            axis, angle = axangles.mat2axangle(raw_robot_rot_mat)
            smoothed_robot_rot = (
                self.robot_orientation_axangle_filter.calculate_average(angle * axis)
            )
            angle = np.linalg.norm(smoothed_robot_rot)
            axis = (
                smoothed_robot_rot / angle
                if angle > 1e-6
                else np.array([0.0, 0.0, 1.0])
            )
            smoothed_robot_quat = quaternions.mat2quat(
                axangles.axangle2mat(axis, angle)
            )
        else:
            smoothed_robot_pos = raw_robot_pos
            smoothed_robot_quat = quaternions.mat2quat(raw_robot_rot_mat)

        robot_pose_msg = PoseStamped()
        robot_pose_msg.header.frame_id = "/arsession"

        if self.motion_estimator.published_pose_timestamp != -1.0:
            publish_pose(
                self.robot_pose_pub, msg.tick, smoothed_robot_pos, smoothed_robot_quat
            )
            publish_multistamped_pose(
                publisher=self.imu_integrated_pose_pub,
                tick=msg.tick,
                system_time=time.monotonic(),
                ros_time=float(self.get_clock().now().nanoseconds) / 1e9,
                source_time=self.motion_estimator.latest_update_timestamp,  # Should be the same as float(msg.tick)/1000
                pos=raw_robot_pos,
                rot_quat=raw_robot_quat,
            )

        new_iphone_pub_timestamp = self.motion_estimator.published_pose_timestamp
        if (
            new_iphone_pub_timestamp != -1.0
            and new_iphone_pub_timestamp != self.prev_iphone_pub_timestamp
        ):
            # Publish transformed iphone poses on
            iphone_transformed_pose = (
                self.motion_estimator.raw_iphone_pose
                @ self.motion_estimator.base2iphone
            )
            publish_multistamped_pose(
                publisher=self.iphone_transformed_pose_pub,
                tick=msg.tick,
                system_time=time.monotonic(),
                ros_time=float(self.get_clock().now().nanoseconds) / 1e9,
                source_time=new_iphone_pub_timestamp,
                pos=iphone_transformed_pose[:3, 3],
                rot_quat=quaternions.mat2quat(iphone_transformed_pose[:3, :3]),
            )
            self.prev_iphone_pub_timestamp = new_iphone_pub_timestamp
        freq = 1 / (time.monotonic() - self.prev_state_time)
        # if freq < self.warning_freq:
        #     self.get_logger().warning(f"robot_pose pub freq={freq:.2f} Hz")
        self.prev_state_time = time.monotonic()


# Run the server
if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rclpy.init(args=None)
    pose_estimation_node = PoseEstimationNode()
    rclpy.spin(pose_estimation_node)
    rclpy.shutdown()

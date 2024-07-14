import base64
from collections import deque
import logging
import os
import time
from typing import Callable, Dict, List, Optional
from modules.velocity_estimator import MovingWindowFilter, VelocityEstimator
import eventlet
import socketio
import base64
import struct
import numpy as np
from transforms3d import quaternions, affines, axangles
from threading import Thread, Lock
import numba as nb


def quat_rotate_inv(q: np.ndarray, v: np.ndarray):
    return quaternions.rotate_vector(
        v=v,
        q=quaternions.qinverse(q),
    )


GRAVITY = np.array([0, 0, -9.81])


class DataPacket:
    # https://developer.apple.com/documentation/arkit/arframe/2867973-timestamp
    # https://stackoverflow.com/questions/45320278/arkit-what-does-the-arframe-timestamp-represent
    def __init__(self, transform_matrix: np.ndarray, timestamp: float):
        self.transform_matrix = transform_matrix.copy()
        # this only represents the uptime of the iphone
        self.timestamp = timestamp

    def __str__(self):
        return f"Translation: {self.transform_matrix[:3, 3]}, Timestamp: {self.timestamp:.3f}"


def decode_data(encoded_str):
    # Decode the base64 string to bytes
    data_bytes = base64.b64decode(encoded_str)

    transform_matrix = np.zeros((4, 4))
    # Unpack transform matrix (16 floats)
    for i in range(4):
        for j in range(4):
            transform_matrix[i, j] = struct.unpack(
                "f", data_bytes[4 * (4 * i + j) : 4 * (4 * i + j + 1)]
            )[0]
    # The transform matrix is stored in column-major order in swift, so we need to transpose it in python
    transform_matrix = transform_matrix.T

    # Unpack timestamp (1 double)
    timestamp = struct.unpack("d", data_bytes[64:72])[0]

    return DataPacket(transform_matrix, timestamp)


def run_pose_receiver(
    callback: Callable[[DataPacket], None], port: int = 5555, silent: bool = True
):
    # Create a Socket.IO server
    sio = socketio.Server()
    # Create a WSGI app
    app = socketio.WSGIApp(sio)

    # Event handler for messages on 'update' channel
    @sio.event
    def update(sid, data):
        callback(decode_data(data))

    eventlet.wsgi.server(
        eventlet.listen(("", port)), app, log=open(os.devnull, "w") if silent else None
    )


class IMU2BodyTwist:
    def __init__(
        self,
        prev_linear_velocity: np.ndarray,  # (3,) initial linear velocity
        robot2imu: np.ndarray,  # (4, 4) transformation matrix from robot to imu frame
        imu_dt: float,
    ):
        self.prev_linear_velocity = prev_linear_velocity
        self.prev_timestamp: Optional[float] = None
        self.imu_dt = imu_dt  # used to initialize previous timestamp if is first update
        self.robot2imu = robot2imu

    def update(
        self,
        acceleration: np.ndarray,  # (3,) in sensor frame
        gyroscope: np.ndarray,  # (3,) in sensor frame
        timestamp_s: float,
        prev_linear_velocity: Optional[np.ndarray] = None,
    ):
        """
        Fuses acceleration and gyroscope information to estimate the velocity
        of the sensor
        """
        if self.prev_timestamp is None:
            self.prev_timestamp = timestamp_s - self.imu_dt
        dt = timestamp_s - self.prev_timestamp
        # naive integration to get new linear velocity
        current_linear_velocity = (
            prev_linear_velocity
            if prev_linear_velocity is not None
            else self.prev_linear_velocity
        ) + dt * acceleration

        # compute twist
        twist = np.zeros(6)
        twist[:3] = gyroscope
        twist[3:] = current_linear_velocity

        # compute robot velocity from imu velocity, using the adjoint representation
        # of the robot2imu transform
        transform_adjoint = np.zeros((6, 6))
        transform_adjoint[:3, :3] = self.robot2imu[:3, :3]
        transform_adjoint[3:, 3:] = self.robot2imu[:3, :3]
        transform_adjoint[:3, 3:] = np.cross(
            self.robot2imu[:3, :3], self.robot2imu[:3, 3]
        )
        new_twist = transform_adjoint @ twist

        self.prev_timestamp = timestamp_s
        self.prev_linear_velocity = new_twist[:3]

        return new_twist


# Useful reference: Modern Robotics book by Kevin Lynch and Frank Park (Section 3.2)


# To get the robot's pose in the iphone's odometry frame using the iphone's outdated
# pose information, we need to integrate the robot's rigid body velocities forward
# from the iphone's outdated pose to the current time. We will treat the iphone's
# odometry frame as the fixed space frame.
def integrate_frame_from_pose(
    start_frame: np.ndarray,  # (4, 4) initial pose
    start_time: float,  # initial timestamp (i.e., since it's outdated, we assume this is in the past)
    rigid_body_vels: List[
        np.ndarray
    ],  # (T, 4, 4) the rigid body's velocity in the body frame. These are skew-symmetric representation of the twist
    rigid_body_vel_timestamps: List[float],  # (T,) # sorted in ascending order
):
    if not len(rigid_body_vels) == len(rigid_body_vel_timestamps):
        logging.warning("Length mismatch")
        return start_frame
    if not np.diff(rigid_body_vel_timestamps).all() >= 0:
        logging.warning("Timestamps not sorted, skipping integration")
        return start_frame

    dts = np.diff([start_time] + rigid_body_vel_timestamps)
    if not (dts >= 0.0).all():
        logging.warning(f"Timestamps not increasing: {dts}, skipping integration")
        return start_frame
    if not (dts <= 0.5).all():
        logging.warning(f"Timestamps too large: {dts}, skipping integration")
        return start_frame

    pose = start_frame.copy()
    for dt, vel in zip(dts, rigid_body_vels):
        assert pose.shape == (4, 4), "Pose should be a 4x4 matrix"
        assert vel.shape == (4, 4), "Velocity should be a 4x4 matrix"
        # NOTE: If the rigid body velocities are in the fixed-space frame, we need to
        # left multiply the pose by the velocities instead. Since they are in the body frame, we right multiply.
        pose = pose + pose @ vel * dt
        # NOTE this time integration might cause rotation matrices to no longer be unitary,
        # so enforce that they are
        q = quaternions.mat2quat(pose[:3, :3])
        # normalize q
        q /= np.linalg.norm(q)
        pose[:3, :3] = quaternions.quat2mat(q)
    return pose


# Where `rigid_body_vels` are computed from the linear and angular velocities of the robot in the body frame
def skew_symmetric_matrix_from_body_vel(lin_vel: np.ndarray, ang_vel: np.ndarray):
    assert lin_vel.shape == (3,), "lin vel should be a 3-dimensional vector"
    assert ang_vel.shape == (3,), "ang vel should be a 3-dimensional vector"
    wedge = np.array(
        [
            [0, -ang_vel[2], ang_vel[1], lin_vel[0]],
            [ang_vel[2], 0, -ang_vel[0], lin_vel[1]],
            [-ang_vel[1], ang_vel[0], 0, lin_vel[2]],
            [0, 0, 0, 0],
        ]
    )
    assert (-wedge[:3, :3].T == wedge[:3, :3]).all(), "Wedge not skew-symmetric"
    return wedge


# The only remaining question is how to get the body-frame rigid body velocities.
# On a quadruped system, we typically have IMU, joint positions/velocities, and foot contact information.
# We could
# - 1. use odometry information from the quadruped's odometry ros topic. However, we do not know where this information comes from and how it is computed.
# - 2. use the IMU information to estimate the body velocities.
# - 3. Combine UMI and joint information to estimate the body velocities using a Kalman filter. Note that this could be most robust if we weigh down the foot contact information when it could be unreliable (below a certain threshold, run a median filter on the foot contact information to remove outliers).


class MotionEstimator:
    def __init__(
        self,
        base2iphone: np.ndarray,
        linear_velocity_estimator: VelocityEstimator,
        robot2imu: np.ndarray,
        angular_velocity_filter: MovingWindowFilter,
        accelerometer_filter: MovingWindowFilter,
        low_level_state_dt: float,  # 500 Hz
        pose_latency: float,  # Measured approximately
        buffer_multiplier: float = 1.5,
        foot_contact_filter_size: int = 5,  # Number of previous foot contact values to consider
    ):
        num_steps = int(buffer_multiplier * pose_latency / low_level_state_dt)
        self.velocities = deque(maxlen=num_steps)
        self.velocity_timestamps = deque(maxlen=num_steps)
        self.linear_velocity_estimator = linear_velocity_estimator
        self.angular_velocity_filter = angular_velocity_filter
        self.prev_linear_velocity = np.zeros(3)
        self.pose = np.identity(4)
        self.raw_iphone_pose = np.identity(4)
        self.latest_update_timestamp = (
            -1.0
        )  # From the latest velocity estimator update call
        self.published_pose_timestamp = -1.0
        self.received_pose_timestamp = -1.0
        self.pub_rec_time_offsets = deque(maxlen=15)
        self.pub_rec_time_offset: Optional[float] = None
        self.pose_latency = pose_latency
        self.base2iphone = base2iphone

        self.imu2body_twist = IMU2BodyTwist(
            prev_linear_velocity=np.zeros(
                3
            ),  # TODO have a way to calibrate this from pose data
            robot2imu=robot2imu,
            imu_dt=low_level_state_dt,
        )
        self.accelerometer_filter = accelerometer_filter
        self.accelerometer_history = deque(maxlen=500)
        self.accelerometer_offset = np.zeros(3)
        self.linear_velocity_body_filter = deque(maxlen=500)
        self.linear_velocity_body_offset = np.zeros(3)
        self.pose_receiver_thread = Thread(
            target=run_pose_receiver, args=(self.update_pose,)
        )
        self.pose_update_lock = Lock()
        self.pose_receiver_thread.start()
        self.foot_contacts = deque(maxlen=foot_contact_filter_size)

    def set_motion_bias(self):
        self.accelerometer_offset = np.mean(self.accelerometer_history, axis=0)
        self.accelerometer_offset += GRAVITY  # gravity
        self.linear_velocity_body_offset = np.mean(
            self.linear_velocity_body_filter, axis=0
        )

    # @profile
    def update_velocity(
        self,
        timestamp_s: float,
        acceleration: np.ndarray,
        gyroscope: np.ndarray,
        foot_contact: np.ndarray,
        quaternion: np.ndarray,
        joint_velocity: np.ndarray,
        joint_position: np.ndarray,
    ):
        """
        Switch to IMU only estimation from previous linear velocity estimation when foot contact is unreliable.
        """
        self.latest_update_timestamp = timestamp_s
        self.accelerometer_history.append(acceleration)
        self.foot_contacts.append(foot_contact)
        filtered_foot_contact = np.array(self.foot_contacts).all(axis=0)
        assert filtered_foot_contact.shape == (4,)
        acceleration = self.accelerometer_filter.calculate_average(
            np.array(acceleration, dtype=np.float64)
        )
        # TODO check that this acceleration and gyroscope is always
        # in the odometry frame
        self.linear_velocity_estimator.update(
            new_timestamp_s=timestamp_s,
            acceleration=acceleration - self.accelerometer_offset,
            foot_contact=filtered_foot_contact,
            quaternion=quaternion[[1, 2, 3, 0]],
            joint_velocity=joint_velocity,
            joint_position=joint_position,
        )
        if np.any(filtered_foot_contact):
            lin_vel_body = self.linear_velocity_estimator.estimated_velocity.copy()
        else:
            twist = self.imu2body_twist.update(
                acceleration=acceleration - self.accelerometer_offset + GRAVITY,
                gyroscope=gyroscope,
                timestamp_s=timestamp_s,
                # use the previous linear velocity to initialize the integration
                # which could come from the foot contacts and therefore be more reliable
                prev_linear_velocity=self.prev_linear_velocity,
            )
            lin_vel_body = twist[3:]
        self.linear_velocity_body_filter.append(lin_vel_body.copy())

        lin_vel_body -= self.linear_velocity_body_offset

        self.prev_linear_velocity = lin_vel_body.copy()

        ang_vel = self.angular_velocity_filter.calculate_average(
            np.array(gyroscope, dtype=np.float64)
        )
        # convert to body frame
        ang_vel_body = quat_rotate_inv(quaternion, ang_vel)

        self.velocities.append(
            skew_symmetric_matrix_from_body_vel(
                lin_vel_body,
                ang_vel_body,
            )
        )
        self.velocity_timestamps.append(timestamp_s)

    # @profile
    def update_pose(self, pose: Optional[DataPacket] = None):
        with self.pose_update_lock:
            # we assume this `pose` drifts less than quadruped's body frame
            if pose is not None and (pose.transform_matrix[:3, 3] != 0.0).all():
                self.raw_iphone_pose = pose.transform_matrix
                self.pub_rec_time_offsets.append(
                    float(time.monotonic()) - pose.timestamp
                )
                self.pub_rec_time_offset = float(np.median(self.pub_rec_time_offsets))
                rec_freq = 1 / (time.monotonic() - self.received_pose_timestamp)
                # if rec_freq < 40.0:  # it should be close to 60Hz
                #     logging.warning(f"Pose receiver frequency: {rec_freq:.2f} Hz")
                self.received_pose_timestamp = time.monotonic()
                self.published_pose_timestamp = pose.timestamp
            self.pose = self.raw_iphone_pose @ self.base2iphone
            if self.pub_rec_time_offset is None:
                return self.pose

            base_latency = (
                time.monotonic() - self.published_pose_timestamp
            ) - self.pub_rec_time_offset

            estimated_total_pose_latency = base_latency + self.pose_latency
            # get only the last `pose_latency` seconds of data
            if len(self.velocity_timestamps) == 0:
                return
            if (
                self.velocity_timestamps[-1] - self.velocity_timestamps[0]
                < estimated_total_pose_latency
            ):
                # Not enough samples yet
                return
            closest_time = self.velocity_timestamps[-1] - estimated_total_pose_latency
            start_idx = np.searchsorted(
                np.array(self.velocity_timestamps), closest_time
            )
            # NOTE: I've tried jit compiling the code below, but to no eval.
            self.pose = integrate_frame_from_pose(
                start_frame=self.pose,
                start_time=closest_time,
                rigid_body_vels=list(self.velocities)[start_idx:],
                rigid_body_vel_timestamps=list(self.velocity_timestamps)[start_idx:],
            )
            # NOTE: this will only intergrate from the estimated time of the pose to
            # the latest low level state msg tick. This means if low level state is outdated
            # then this estimate will also be outdated.


"""
Questions:
 - 1. 4x4 wedge matrices used for integration are formed from twists?
"""

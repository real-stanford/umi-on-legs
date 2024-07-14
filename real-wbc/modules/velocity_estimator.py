from operator import le
import time
from typing import List

import numpy as np
import numpy.typing as npt
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
import numba as nb
from numba.experimental import jitclass
import collections


"""Moving window filter to smooth out sensor readings. From https://github.com/erwincoumans/motion_imitation"""

spec = [
    ("_window_size", nb.int64),
    ("_data_dim", nb.int64),
    ("_value_deque", nb.types.Array(nb.float64, 2, "C")),
    ("_sum", nb.float64[:]),
    ("_correction", nb.float64[:]),
]


@jitclass(spec=spec)  # type: ignore
class MovingWindowFilter(object):
    """A stable O(1) moving filter for incoming data streams.
    We implement the Neumaier's algorithm to calculate the moving window average,
    which is numerically stable.
    """

    def __init__(self, window_size: int, data_dim: int):
        """Initializes the class.

        Args:
          window_size: The moving window size.
        """
        assert window_size > 0
        self._window_size: int = window_size
        self._data_dim = data_dim
        # self._value_deque = collections.deque(maxlen=window_size)
        # Use numpy array to simulate deque so that it can be compiled by numba
        self._value_deque = np.zeros((self._data_dim, window_size), dtype=np.float64)
        # The moving window sum.
        self._sum = np.zeros((self._data_dim,), dtype=np.float64)
        # The correction term to compensate numerical precision loss during
        # calculation.
        self._correction = np.zeros((self._data_dim,), dtype=np.float64)

    def _neumaier_sum(self, value: npt.NDArray[np.float64]):
        """Update the moving window sum using Neumaier's algorithm.

        For more details please refer to:
        https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements

        Args:
          value: The new value to be added to the window.
        """
        assert value.shape == (self._data_dim,)
        new_sum = self._sum + value
        for k in range(self._data_dim):
            if abs(self._sum[k]) >= abs(value[k]):
                # If self._sum is bigger, low-order digits of value are lost.
                self._correction[k] += (self._sum[k] - new_sum[k]) + value[k]
            else:
                # low-order digits of sum are lost
                self._correction[k] += (value[k] - new_sum[k]) + self._sum[k]

        self._sum = new_sum

    def calculate_average(
        self, new_value: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes the moving window average in O(1) time.

        Args:
          new_value: The new value to enter the moving window.

        Returns:
          The average of the values in the window.

        """
        assert new_value.shape == (self._data_dim,)

        self._neumaier_sum(-self._value_deque[:, 0])
        self._neumaier_sum(new_value)

        # self._value_deque.append(new_value)
        for i in range(self._data_dim):
            self._value_deque[i, :] = np.roll(self._value_deque[i, :], -1)
        self._value_deque[:, -1] = new_value

        return (self._sum + self._correction) / self._window_size


@nb.jit(nopython=True, cache=True, parallel=True)
def analytical_leg_jacobian(
    leg_angles: npt.NDArray[np.float64],
    leg_id: int,
    hip_length: float,
    thigh_length: float,
    calf_length: float,
):
    """
    Computes the analytical Jacobian.
    Args:
        leg_angles: a list of 3 numbers for current abduction, hip and knee angle.
        l_hip_sign: whether it's a left (1) or right(-1) leg.
    """
    assert len(leg_angles) == 3
    assert leg_id in [0, 1, 2, 3]

    hip_angle, thigh_angle, calf_angle = leg_angles[0], leg_angles[1], leg_angles[2]

    # Compute the effective length of the leg
    leg_length_eff = np.sqrt(
        thigh_length**2
        + calf_length**2
        + 2 * thigh_length * calf_length * np.cos(calf_angle)
    )
    leg_angle_eff = thigh_angle + calf_angle / 2

    J = np.zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = -leg_length_eff * np.cos(leg_angle_eff)
    J[0, 2] = (
        calf_length
        * thigh_length
        * np.sin(calf_angle)
        * np.sin(leg_angle_eff)
        / leg_length_eff
        - leg_length_eff * np.cos(leg_angle_eff) / 2
    )
    J[1, 0] = -hip_length * np.sin(hip_angle) + leg_length_eff * np.cos(
        hip_angle
    ) * np.cos(leg_angle_eff)
    J[1, 1] = -leg_length_eff * np.sin(hip_angle) * np.sin(leg_angle_eff)
    J[1, 2] = (
        -calf_length
        * thigh_length
        * np.sin(hip_angle)
        * np.sin(calf_angle)
        * np.cos(leg_angle_eff)
        / leg_length_eff
        - leg_length_eff * np.sin(hip_angle) * np.sin(leg_angle_eff) / 2
    )
    J[2, 0] = hip_length * np.cos(hip_angle) + leg_length_eff * np.sin(
        hip_angle
    ) * np.cos(leg_angle_eff)
    J[2, 1] = leg_length_eff * np.sin(leg_angle_eff) * np.cos(hip_angle)
    J[2, 2] = (
        calf_length
        * thigh_length
        * np.sin(calf_angle)
        * np.cos(hip_angle)
        * np.cos(leg_angle_eff)
        / leg_length_eff
        + leg_length_eff * np.sin(leg_angle_eff) * np.cos(hip_angle) / 2
    )
    return J


@nb.jit(nopython=True, cache=True, parallel=True)
def inv_with_jit(M: npt.NDArray[np.float64]):
    return np.linalg.inv(M)


class VelocityEstimator:
    """Estimates base velocity of A1 robot.

    The velocity estimator consists of 2 parts:
    1) A state estimator for CoM velocity.

    Two sources of information are used:
    The integrated reading of accelerometer and the velocity estimation from
    contact legs. The readings are fused together using a Kalman Filter.

    2) A moving average filter to smooth out velocity readings
    """

    def __init__(
        self,
        hip_length: float,
        thigh_length: float,
        calf_length: float,
        accelerometer_variance=0.1,
        sensor_variance=0.1,
        initial_variance=0.1,
        moving_window_filter_size=120,
        default_control_dt=0.002,
    ):
        """Initiates the velocity estimator.

        See filterpy documentation in the link below for more details.
        https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html

        Args:
          accelerometer_variance: noise estimation for accelerometer reading.
          sensor_variance: noise estimation for motor velocity reading.
          initial_covariance: covariance estimation of initial state.
        """

        self.filter = KalmanFilter(dim_x=3, dim_z=3, dim_u=3)
        self.filter.x = np.zeros(3)
        self._initial_variance = initial_variance
        self.filter.P = np.eye(3) * self._initial_variance  # State covariance
        self.filter.Q = np.eye(3) * accelerometer_variance
        self.filter.R = np.eye(3) * sensor_variance

        self.filter.H = np.eye(3)  # measurement function (y=H*x)
        self.filter.F = np.eye(3)  # state transition matrix
        self.filter.B = np.eye(3)  # type: ignore
        self.filter.inv = inv_with_jit  # type: ignore     To accelerate inverse calculation (~3x faster)

        self._window_size = moving_window_filter_size
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )
        self._estimated_velocity = np.zeros(3)
        self._last_timestamp_s = 0.0
        self._default_control_dt = default_control_dt
        self.hip_length = hip_length
        self.thigh_length = thigh_length
        self.calf_length = calf_length

        # Precompile all jit functions:
        start_precompile_time = time.monotonic()
        analytical_leg_jacobian(
            leg_angles=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            leg_id=0,
            hip_length=self.hip_length,
            thigh_length=self.thigh_length,
            calf_length=self.calf_length,
        )
        self.filter.inv(np.eye(3))
        print(f"Precompile time spent {time.monotonic() - start_precompile_time:.5f}")

    def reset(self):
        self.filter.x = np.zeros(3)
        self.filter.P = np.eye(3) * self._initial_variance
        self.moving_window_filter = MovingWindowFilter(
            window_size=self._window_size, data_dim=3
        )

        self._last_timestamp_s = 0.0

    def _compute_delta_time(self, new_timestamp_s: float):
        if self._last_timestamp_s == 0.0:
            # First timestamp received, return an estimated delta_time.
            delta_time_s = self._default_control_dt
        else:
            delta_time_s = new_timestamp_s - self._last_timestamp_s
        self._last_timestamp_s = new_timestamp_s
        return delta_time_s

    def update(
        self,
        new_timestamp_s: float,
        acceleration: npt.NDArray[np.float64],
        foot_contact: npt.NDArray[np.float64],
        quaternion: npt.NDArray[np.float64],
        joint_velocity: npt.NDArray[np.float64],
        joint_position: npt.NDArray[np.float64],
    ):
        """Propagate current state estimate with new accelerometer reading."""
        assert acceleration.shape == (3,)
        assert foot_contact.shape == (4,)
        # foot_contact should be either 0.0 or 1.0 (however checking this value takes a long time)
        # assert np.all(np.logical_or(foot_contact == 0.0, foot_contact == 1.0))
        assert quaternion.shape == (4,)
        assert joint_velocity.shape == (12,)

        delta_time_s = self._compute_delta_time(new_timestamp_s)
        # Get rotation matrix from quaternion
        rot_mat = R.from_quat(quaternion).as_matrix()
        rot_mat = np.array(rot_mat).reshape((3, 3))
        calibrated_acc = rot_mat.dot(acceleration) + np.array([0.0, 0.0, -9.81])
        self.filter.predict(u=calibrated_acc * delta_time_s)

        # Correct estimation using contact legs
        observed_velocities = []
        for leg_id in range(4):
            if foot_contact[leg_id]:
                jacobian = analytical_leg_jacobian(
                    leg_angles=joint_position[leg_id * 3 : (leg_id + 1) * 3],
                    leg_id=leg_id,
                    hip_length=self.hip_length,
                    thigh_length=self.thigh_length,
                    calf_length=self.calf_length,
                )  # TODO: Find out range mapping
                # Only pick the jacobian related to joint motors
                joint_velocities = joint_velocity[leg_id * 3 : (leg_id + 1) * 3]
                leg_velocity_in_base_frame = jacobian.dot(joint_velocities)
                base_velocity_in_base_frame = -leg_velocity_in_base_frame[:3]
                observed_velocities.append(rot_mat.dot(base_velocity_in_base_frame))

        if observed_velocities:
            observed_velocities = np.mean(observed_velocities, axis=0)
            self.filter.update(observed_velocities)

        self._estimated_velocity = self.moving_window_filter.calculate_average(
            self.filter.x
        )

    @property
    def estimated_velocity(self):
        return self._estimated_velocity.copy()

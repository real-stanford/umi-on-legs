import numpy as np
import numpy.typing as npt
import logging
from typing import List

from scipy.spatial.transform import Rotation, Slerp


def slerp_wxyz(quat1, quat2, alpha):
    w, x, y, z = quat1
    start_rot = Rotation.from_quat([x, y, z, w])
    w, x, y, z = quat2
    end_rot = Rotation.from_quat([x, y, z, w])
    orientation_slerp = Slerp(
        times=[0, 1], rotations=Rotation.concatenate([start_rot, end_rot])
    )
    x, y, z, w = orientation_slerp([alpha])[0].as_quat()
    return np.array([w, x, y, z])


class RealtimeTraj:

    def __init__(self):
        self.translations = np.zeros((0, 3), dtype=np.float64)
        self.quaternions_wxyz = np.zeros((0, 4), dtype=np.float64)
        self.gripper_pos = np.zeros((0,), dtype=np.float64)
        self.timestamps = np.zeros((0,), dtype=np.float64)  # in seconds

    def update(
        self,
        translations: npt.NDArray[np.float64],
        quaternions_wxyz: npt.NDArray[np.float64],
        gripper_pos: npt.NDArray[np.float64],
        timestamps: npt.NDArray[np.float64],
        current_timestamp: float,
        adaptive_latency_matching: bool = False,
        smoothen_time: float = 0.0,
    ):
        assert (
            translations.shape[1] == 3
        ), f"Invalid shape {translations.shape[1]} for translations!"
        assert (
            quaternions_wxyz.shape[1] == 4
        ), f"Invalid shape {quaternions_wxyz.shape[1]} for quaternions_wxyz!"
        assert (
            len(gripper_pos.shape) == 1
        ), f"Invalid shape {gripper_pos.shape} for gripper_pos!"
        assert (
            len(timestamps.shape) == 1
        ), f"Invalid shape {timestamps.shape} for timestamps!"
        assert (
            translations.shape[0]
            == quaternions_wxyz.shape[0]
            == gripper_pos.shape[0]
            == timestamps.shape[0]
        ), f"Input number inconsistent!"
        if len(timestamps) > 1 and np.any(timestamps[1:] - timestamps[:-1] <= 0):
            logging.warning(f"Input timestamps are not monotonically increasing!")

        if self.translations.shape[0] == 0:
            self.translations = np.array(translations)
            self.quaternions_wxyz = np.array(quaternions_wxyz)
            self.gripper_pos = np.array(gripper_pos)
            self.timestamps = np.array(timestamps)
        else:
            input_traj = RealtimeTraj()
            input_traj.update(
                translations=translations,
                quaternions_wxyz=quaternions_wxyz,
                gripper_pos=gripper_pos,
                timestamps=timestamps,
                current_timestamp=timestamps[0],
            )
            if adaptive_latency_matching:
                latency_precision = 0.02
                max_latency = 1.5
                min_latency = -0.0
                matching_dt = 0.05

                pose_samples = np.zeros((3, 3))
                for i in range(3):
                    t = self.interpolate_translation(
                        current_timestamp + (i - 1) * matching_dt
                    )
                    pose_samples[i, :3] = t
                errors = []
                error_weights = np.array(
                    [1, 1, 1]
                )  # x, y, z, qw, qx, qy, qz

                for latency in np.arange(min_latency, max_latency, latency_precision):
                    input_pose_samples = np.zeros((3, 3))
                    for i in range(3):
                        t = input_traj.interpolate_translation(
                            current_timestamp + latency + (i - 1) * matching_dt
                        )
                        input_pose_samples[i, :3] = t

                    error = np.sum(
                        np.abs(input_pose_samples - pose_samples) * error_weights
                    )
                    errors.append(error)
                errors = np.array(errors)
                best_latency = np.arange(min_latency, max_latency, latency_precision)[
                    np.argmin(errors)
                ]
                print(f"{best_latency=}")
                # input_traj.timestamps -= latency

            if smoothen_time > 0.0:
                for i in range(len(input_traj.timestamps)):
                    if input_traj.timestamps[i] <= current_timestamp:
                        t, q, g = self.interpolate(input_traj.timestamps[i])
                        input_traj.translations[i] = t
                        input_traj.quaternions_wxyz[i] = q
                        input_traj.gripper_pos[i] = g
                    elif input_traj.timestamps[i] <= current_timestamp + smoothen_time:
                        alpha = (
                            input_traj.timestamps[i] - current_timestamp
                        ) / smoothen_time
                        t, q, g = self.interpolate(input_traj.timestamps[i])
                        input_traj.translations[i] = (
                            alpha * input_traj.translations[i] + (1 - alpha) * t
                        )
                        input_traj.quaternions_wxyz[i] = (
                            alpha * input_traj.quaternions_wxyz[i] + (1 - alpha) * q
                        )
                        input_traj.gripper_pos[i] = (
                            alpha * input_traj.gripper_pos[i] + (1 - alpha) * g
                        )
                    else:
                        break

            # Find the last timestamp prior to the first timestamp of the input data
            idx = np.searchsorted(self.timestamps, input_traj.timestamps[0])

            # Remove all data after this timestamp
            self.translations = self.translations[:idx]
            self.quaternions_wxyz = self.quaternions_wxyz[:idx]
            self.gripper_pos = self.gripper_pos[:idx]
            self.timestamps = self.timestamps[:idx]

            self.translations = np.concatenate(
                [self.translations, input_traj.translations]
            )
            self.quaternions_wxyz = np.concatenate(
                [self.quaternions_wxyz, input_traj.quaternions_wxyz]
            )
            self.gripper_pos = np.concatenate(
                [self.gripper_pos, input_traj.gripper_pos]
            )
            self.timestamps = np.concatenate([self.timestamps, input_traj.timestamps])

            assert np.all(
                self.timestamps[1:] - self.timestamps[:-1] > 0
            ), f"Timestamps are not monotonically increasing!"

        current_idx = np.searchsorted(self.timestamps, current_timestamp)
        # Only keep one data point before the current timestamp (for interpolation)
        if current_idx >= 2:
            self.translations = self.translations[current_idx - 1 :]
            self.quaternions_wxyz = self.quaternions_wxyz[current_idx - 1 :]
            self.gripper_pos = self.gripper_pos[current_idx - 1 :]
            self.timestamps = self.timestamps[current_idx - 1 :]
    
    def interpolate_translation(self, timestamp: float):

        if len(self.timestamps) == 0:
            raise ValueError("Trajectory not initialized")
        if timestamp <= self.timestamps[0]:
            return self.translations[0].copy()
        if timestamp >= self.timestamps[-1]:
            return self.translations[-1].copy()

        # There should be at least two timestamps
        idx = np.searchsorted(self.timestamps, timestamp)
        alpha = (timestamp - self.timestamps[idx - 1]) / (
            self.timestamps[idx] - self.timestamps[idx - 1]
        )
        translation = (1 - alpha) * self.translations[
            idx - 1
        ] + alpha * self.translations[idx]
        
        return translation

    def interpolate(self, timestamp: float):
        if len(self.timestamps) == 0:
            raise ValueError("Trajectory not initialized")
        if timestamp <= self.timestamps[0]:
            return (
                self.translations[0].copy(),
                self.quaternions_wxyz[0].copy(),
                self.gripper_pos[0].copy(),
            )
        if timestamp >= self.timestamps[-1]:
            return (
                self.translations[-1].copy(),
                self.quaternions_wxyz[-1].copy(),
                self.gripper_pos[-1].copy(),
            )

        # There should be at least two timestamps
        idx = np.searchsorted(self.timestamps, timestamp)
        alpha = (timestamp - self.timestamps[idx - 1]) / (
            self.timestamps[idx] - self.timestamps[idx - 1]
        )
        translation = (1 - alpha) * self.translations[
            idx - 1
        ] + alpha * self.translations[idx]
        
        quaternion_wxyz = slerp_wxyz(self.quaternions_wxyz[idx-1], self.quaternions_wxyz[idx], alpha)

        
        gripper_pos = (1 - alpha) * self.gripper_pos[
            idx - 1
        ] + alpha * self.gripper_pos[idx]

        return (
            translation,
            quaternion_wxyz,
            gripper_pos,
        )

    def interpolate_traj(self, timestamps: List[float]):
        
        assert len(timestamps) >= 1, "Not enough timestamps"
        
        translations = []
        quaternions_wxyz = []
        gripper_positions = []

        for timestamp in timestamps:
            t, q, g = self.interpolate(timestamp)
            translations.append(t)
            quaternions_wxyz.append(q)
            gripper_positions.append(g)
        
        return (
            np.stack(translations), # (N, 3)
            np.stack(quaternions_wxyz), # (N, 4)
            np.array(gripper_positions), # (N, )
        )
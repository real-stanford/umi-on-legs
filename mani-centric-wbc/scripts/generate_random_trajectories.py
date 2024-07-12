import pickle
from io import BytesIO
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch3d.transforms as p3d
import ray
import torch
from mpl_toolkits.mplot3d import Axes3D
from rich.progress import track
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from transforms3d import euler, quaternions, axangles

if __name__ == "__main__":
    time_scaling = 1.0
    sampling_rate = 200
    parsed_plan = []
    ee_pos_bounds = np.array(
        [
            [-0.5, -0.5, 0.02],
            [0.5, 0.5, 0.6],
        ]
    )
    ee_euler_bounds = np.array(
        [[-np.pi / 2, -np.pi, -np.pi], [np.pi / 2, np.pi, np.pi]]
    )
    ee_lin_vel_bounds = np.array([0.05, 2.0])
    ee_ang_vel_bounds = np.array([0.1, 3.0])
    episode_len = 3000
    base_ee_rotation = euler.euler2mat(-1.57079632679, 0.0, -1.57079632679, axes="sxyz")
    for i in range(400):
        ee_pos_waypoints = [np.array([0.0, 0.0, 0.4])]
        while (
            len(ee_pos_waypoints) < episode_len
        ):  # similar length to scaling up bin transport trajectories
            new_ee_pos = np.random.uniform(ee_pos_bounds[0], ee_pos_bounds[1])
            dist_from_last = np.linalg.norm(new_ee_pos - ee_pos_waypoints[-1])
            time = dist_from_last / np.random.uniform(
                ee_lin_vel_bounds[0], ee_lin_vel_bounds[1]
            )
            ee_pos_waypoints.extend(
                np.array(
                    [
                        np.interp(
                            np.linspace(0, 1.0, int(time * sampling_rate) + 1),
                            np.array([0.0, 1.0]),
                            np.array([ee_pos_waypoints[-1][i], new_ee_pos[i]]),
                        )
                        for i in range(3)
                    ]
                ).T
            )
        ee_axis_angle_waypoints = [np.array([0.0, 0.0, 0.0])]
        while len(ee_axis_angle_waypoints) < episode_len:
            new_euler = np.random.uniform(ee_euler_bounds[0], ee_euler_bounds[1])
            new_rot_mat = euler.euler2mat(*new_euler, axes="sxyz")
            axis, angle = axangles.mat2axangle(new_rot_mat)
            new_ee_axis_angle = axis * angle
            dist_from_last = np.linalg.norm(
                new_ee_axis_angle - ee_axis_angle_waypoints[-1]
            )
            dist_from_last = dist_from_last % 2 * np.pi
            dist_from_last = min(dist_from_last, 2 * np.pi - dist_from_last)
            time = dist_from_last / np.random.uniform(
                ee_ang_vel_bounds[0], ee_ang_vel_bounds[1]
            )
            ee_axis_angle_waypoints.extend(
                np.array(
                    [
                        np.interp(
                            np.linspace(0, 1.0, int(time * sampling_rate) + 1),
                            np.array([0.0, 1.0]),
                            np.array(
                                [ee_axis_angle_waypoints[-1][i], new_ee_axis_angle[i]]
                            ),
                        )
                        for i in range(3)
                    ]
                ).T
            )
        ee_pos = np.array(ee_pos_waypoints)[:episode_len]
        ee_rot_mat = p3d.axis_angle_to_matrix(
            torch.tensor(ee_axis_angle_waypoints[:episode_len])
        )
        ee_rot_mat = torch.matmul(
            torch.from_numpy(base_ee_rotation).repeat(episode_len, 1, 1), ee_rot_mat
        )
        ee_axis_angle = p3d.matrix_to_axis_angle(ee_rot_mat).numpy()
        gripper_width = np.zeros_like(ee_pos)
        t = np.linspace(0, len(ee_pos) / sampling_rate, len(ee_pos) + 1)
        parsed_plan.append(
            {
                "t": t,
                "ee_pos": ee_pos,
                "ee_axis_angle": ee_axis_angle,
                "gripper_width": gripper_width,
            }
        )
    pickle.dump(
        parsed_plan,
        open("random_trajectories.pkl", "wb"),
    )

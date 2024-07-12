import pickle
from io import BytesIO
from pathlib import Path
from typing import Optional
import seaborn as sns
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
from transforms3d import euler, quaternions
from scipy.signal import butter, lfilter, filtfilt, medfilt
import numpy as np
from scipy.spatial.transform import Rotation
import seaborn as sns
import matplotlib.pyplot as plt


def skew_symmetric(matrix):
    return np.array(
        [
            [0, -matrix[2], matrix[1]],
            [matrix[2], 0, -matrix[0]],
            [-matrix[1], matrix[0], 0],
        ]
    )


def compute_angular_velocity(rot_matrices, timestamps):
    """
    https://physics.stackexchange.com/questions/293037/how-to-compute-the-angular-velocity-from-the-angles-of-a-rotation-matrix

    https://math.stackexchange.com/questions/668866/how-do-you-find-angular-velocity-given-a-pair-of-3x3-rotation-matrices
    """
    angular_velocities = []
    for i in range(1, len(rot_matrices)):
        dt = timestamps[i] - timestamps[i - 1]
        rotation_diff = rot_matrices[i] @ np.linalg.inv(rot_matrices[i - 1])

        theta = np.arccos((np.trace(rotation_diff) - 1) / 2)
        if theta == 0:
            angular_velocities.append(np.zeros(3))
            continue
        J_v = (rotation_diff - rotation_diff.T) / (2 * np.sin(theta))
        W = J_v * theta / dt
        angular_velocity = np.array([W[2, 1], W[0, 2], W[1, 0]])
        angular_velocities.append(angular_velocity)
    return np.array(angular_velocities)


def smooth_traj(y: np.ndarray, b, a, pad: int = 100, shift_idx: int = 12):
    smoothed = medfilt(y, 5)
    smoothed = filtfilt(b, a, y, axis=0)
    return smoothed


def visualize(raw, smoothed, path: Optional[str] = None):
    fig, axes = plt.subplots(3, figsize=(15, 15))
    x = np.arange(len(raw))
    for i in range(3):
        sns.lineplot(
            x=x,
            y=raw[:, i],
            ax=axes[i],
            label=f"raw",
        )
        sns.lineplot(
            x=x,
            y=smoothed[:, i],
            ax=axes[i],
            label=f"smoothed",
        )
        axes[i].legend()
        axes[i].set_title(f"Dimension {i}")
    if path is not None:
        plt.tight_layout(pad=0)
        plt.savefig(path)
    else:
        plt.show()
    # close everything
    plt.cla()
    plt.clf()
    plt.close()
    # delete
    del fig
    del axes


SMOOTH = False
VISUALIZE = False
if __name__ == "__main__":
    path = "/path/to/umi/dataset_plan.pkl"

    time_scaling = 1 / 0.25
    time_scaling = 1.0
    sampling_rate = 200

    # pose_cutoff = 20.0
    pose_cutoff = 12.0
    pose_sampling_rate = 60.0  # Hz
    # pose_order = 10
    pose_order = 2
    pose_b, pose_a = butter(
        pose_order, pose_cutoff, fs=sampling_rate, btype="low", analog=False
    )

    vel_cutoff = 8.0
    vel_sampling_rate = 200.0  # Hz
    vel_order = 8
    vel_b, vel_a = butter(
        vel_order, vel_cutoff, fs=sampling_rate, btype="low", analog=False
    )

    sns.set_style("darkgrid")
    plan = pickle.load(open(path, "rb"))
    parsed_plan = []
    for i in range(len(plan)):
        if "toss_objects" in path:
            ee_pose = plan[i]["tcp_poses"]
            gripper_width = plan[i]["gripper_widths"]
            t = plan[i]["video_timestamps"]
        else:
            ee_pose = plan[i]["grippers"][0]["tcp_pose"]
            gripper_width = plan[i]["grippers"][0]["gripper_width"]
            t = plan[i]["episode_timestamps"]
        dt = np.median(np.diff(t))
        assert dt < 1 / sampling_rate - 1e-2 or dt > 1 / sampling_rate + 1e-2
        ee_pos = ee_pose[:, :3].astype(np.float32)
        ee_axis_angle = ee_pose[:, 3:].astype(np.float32)
        if SMOOTH:
            smoothed_ee_pos = np.array(
                [
                    smooth_traj(
                        ee_pos[:, i].copy(), b=pose_b, a=pose_a, pad=100, shift_idx=7
                    )
                    for i in range(3)
                ]
            ).T
            if VISUALIZE:
                visualize(
                    ee_pos,
                    smoothed_ee_pos,
                    path=path.replace(".pkl", f"_pos_{i:04d}.png"),
                )
            ee_pos = smoothed_ee_pos
        ee_pose = torch.from_numpy(np.identity(4)).repeat(len(ee_pos), 1, 1)
        ee_pose[:, :3, 3] = torch.from_numpy(ee_pos)
        ee_axis_angle = torch.from_numpy(ee_axis_angle)
        ee_pose[:, :3, :3] = p3d.axis_angle_to_matrix(ee_axis_angle)
        # rotate by 90 degrees about the z axis
        transform_mat = torch.from_numpy(np.identity(4)).repeat(len(ee_pos), 1, 1)
        transform_mat[:, :3, :3] = p3d.euler_angles_to_matrix(
            torch.tensor([0, 0, -np.pi / 2]),
            "XYZ",
        )
        ee_pose = torch.matmul(transform_mat, ee_pose)
        ee_pos = ee_pose[:, :3, 3].numpy()
        ee_quat_wxyz = p3d.matrix_to_quaternion(ee_pose[:, :3, :3]).numpy()

        t = t - t.min()
        t *= time_scaling

        # interpolate at 200Hz
        new_t = np.linspace(t.min(), t.max(), int((t.max() - t.min()) * sampling_rate))
        ee_pos = np.array([np.interp(new_t, t, ee_pos[:, i]) for i in range(3)]).T

        # compute dynamics using central difference
        linear_velocity = np.gradient(ee_pos, axis=0) / np.gradient(new_t)[:, None]
        # smooth out lin vel
        smoothed_linear_velocity = np.array(
            [
                smooth_traj(
                    linear_velocity[:, i].copy(),
                    b=vel_b,
                    a=vel_a,
                    pad=100,
                    shift_idx=20,
                )
                for i in range(3)
            ]
        ).T
        linear_velocity = smoothed_linear_velocity
        linear_acceleration = (
            np.gradient(linear_velocity, axis=0) / np.gradient(new_t)[:, None]
        )
        if VISUALIZE:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            for i in range(3):
                sns.lineplot(x=new_t, y=ee_pos[:, i], ax=axes[i, 0], label="pos")
                sns.lineplot(
                    x=new_t, y=linear_velocity[:, i], ax=axes[i, 1], label="vel"
                )
                sns.lineplot(
                    x=new_t, y=linear_acceleration[:, i], ax=axes[i, 2], label="acc"
                )
                axes[i, 0].set_title(f"Dimension {i}")
                axes[i, 1].set_title(f"Dimension {i}")
                axes[i, 2].set_title(f"Dimension {i}")
            plt.show()

        # interpolate orientation using scipy (scalar last)
        ee_quat_xyzw = ee_quat_wxyz[:, [1, 2, 3, 0]]
        r = R.from_quat(ee_quat_xyzw)
        s = Slerp(times=t, rotations=r)
        new_r = s(new_t)
        ee_axis_angle = new_r.as_rotvec()

        ang_vel = compute_angular_velocity(
            new_r.as_matrix(),
            new_t,
        )
        ang_vel = np.concatenate((ang_vel, [ang_vel[-1]]))

        # smooth out ang vel
        smoothed_ang_vel = np.array(
            [
                smooth_traj(
                    ang_vel[:, i].copy(), b=vel_b, a=vel_a, pad=100, shift_idx=20
                )
                for i in range(3)
            ]
        ).T
        if VISUALIZE:
            visualize(
                ang_vel,
                smoothed_ang_vel,
            )
        ang_vel = smoothed_ang_vel

        # plot angular velocity and rotation
        if VISUALIZE:
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            for i in range(3):
                sns.lineplot(x=new_t, y=ee_axis_angle[:, i], ax=axes[i, 0])
                sns.lineplot(x=new_t, y=ang_vel[:, i], ax=axes[i, 1])
            plt.show()

        gripper_width = np.interp(new_t, t, gripper_width)
        ee_pos[:, 1] -= ee_pos[0, 1]
        ee_pos[:, 0] -= ee_pos[0, 0]
        parsed_plan.append(
            {
                "t": np.linspace(0, len(new_t) / sampling_rate, len(new_t)),
                "ee_pos": ee_pos,
                "ee_axis_angle": ee_axis_angle,
                "gripper_width": gripper_width,
                "linear_velocity": linear_velocity,
                "linear_acceleration": linear_acceleration,
                "angular_velocity": ang_vel,
            }
        )
    max_len = max([len(p["t"]) for p in parsed_plan])
    # pad to max len
    for p in parsed_plan:
        for k in p.keys():
            if len(p[k]) == max_len:
                continue
            elif len(p[k]) > max_len:
                p[k] = p[k][:max_len]
            else:
                p[k] = np.concatenate(
                    (
                        p[k],
                        [p[k][-1]] * (max_len - len(p[k])),
                    )
                )
    prefix = path.split("/")[1]
    smoothed = "smoothed" if SMOOTH else "raw"
    output_path = f"{prefix}_{smoothed}.pkl"
    print(f"{len(plan)} trajectories processed.")
    input(f"dump to {output_path}?")
    pickle.dump(
        parsed_plan,
        open(output_path, "wb"),
    )

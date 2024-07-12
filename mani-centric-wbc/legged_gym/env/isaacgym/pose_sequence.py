"""
Implements pose sequence samplers, which are used to generate a sequence of
poses for the robot to follow.
"""

import pickle
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pytorch3d.transforms as pt3d
import torch

from legged_gym.env.isaacgym.utils import orn_diff_axis_angles, torch_rand_float


class SequenceSampler(ABC):
    """
    Abstract base class for pose sequence samplers.
    """

    def __init__(
        self,
        episode_length_s: float,
        dt: float,
        device: str,
    ):
        self.episode_length_s = episode_length_s  # in seconds
        self.dt = dt
        self.device = device

    @property
    def episode_length(self) -> int:
        return int(self.episode_length_s / self.dt)

    @abstractmethod
    def sample(self, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a sequence of poses.
        Returns:
            pos_seq: (T, 3) sequence of positions in world frame, first position starts
                     at (0,0,0)
            orn_seq: (T, 3, 3) sequence of orientations (rot mat) in world frame
        """
        pass


class PicklePoseSequenceLoader(SequenceSampler):
    def __init__(
        self,
        file_path: str,
        planar_center: bool,
        add_random_height_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        episodes = pickle.load(open(file_path, "rb"))
        self.ee_pos = (
            torch.from_numpy(
                np.stack([episode["ee_pos"] for episode in episodes], axis=0)
            )
            .to(self.device)
            .float()
        )
        ee_axis_angle = (
            torch.from_numpy(
                np.stack([episode["ee_axis_angle"] for episode in episodes], axis=0)
            )
            .to(self.device)
            .float()
        )
        self.ee_rot_mat = pt3d.axis_angle_to_matrix(ee_axis_angle)
        self.planar_center = planar_center
        self.added_random_height_range = add_random_height_range

    def sample(self, batch_size: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        episode_idx = torch.randint(
            0,
            self.ee_pos.shape[0],
            (batch_size,),
            device=self.device,
            generator=generator,
        )
        pos_seq = self.ee_pos[episode_idx, : self.episode_length]
        rot_mat_seq = self.ee_rot_mat[episode_idx, : self.episode_length]
        if pos_seq.shape[1] < self.episode_length:
            # pad
            pos_seq = torch.cat(
                [
                    pos_seq,
                    pos_seq[:, -1, :][:, None, :].repeat(
                        1, self.episode_length - pos_seq.shape[1], 1
                    ),
                ],
                dim=1,
            )
            rot_mat_seq = torch.cat(
                [
                    rot_mat_seq,
                    rot_mat_seq[:, -1, :, :][:, None, :, :].repeat(
                        1, self.episode_length - rot_mat_seq.shape[1], 1, 1
                    ),
                ],
                dim=1,
            )
        if self.planar_center:
            # center the trajectory in the xy plane, average over 1st to 3rd steps
            pos_seq[..., :2] -= pos_seq[:, [1, 2, 3], :2].mean(dim=1, keepdim=True)
        if self.added_random_height_range is not None:
            added_height = (
                torch.rand(
                    (batch_size, 1),
                    device=self.device,
                    generator=generator,
                )
                * (
                    self.added_random_height_range[1]
                    - self.added_random_height_range[0]
                )
                + self.added_random_height_range[0]
            )
            pos_seq[..., 2] += added_height

        return pos_seq, rot_mat_seq

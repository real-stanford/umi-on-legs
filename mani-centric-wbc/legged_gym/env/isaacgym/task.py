from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_rotate_inverse
from matplotlib import pyplot as plt

from legged_gym.env.isaacgym.control import Control
from legged_gym.env.isaacgym.pose_sequence import SequenceSampler
from legged_gym.env.isaacgym.state import EnvState
from legged_gym.env.isaacgym.utils import torch_rand_float


def check_should_reset(
    time_s: torch.Tensor,
    dt: float,
    reset_time_s: float,
):
    num_episode_steps = (time_s / dt).long()
    reset_every_n_steps = int(reset_time_s / dt)
    should_reset = (num_episode_steps % reset_every_n_steps) == 0
    return should_reset


class Task:
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
    ):
        self.gym = gym
        self.sim = sim
        self.num_envs = gym.get_env_count(sim)
        self.device = device
        self.generator = generator

    @abstractmethod
    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # per simulation step callback, returns a dictionary task metrics (accuracy, etc.)
        pass

    def reset_idx(self, env_ids: torch.Tensor):
        # per episode
        pass

    def visualize(self, state: EnvState, viewer: gymapi.Viewer, vis_env_ids: List[int]):
        pass

    @abstractmethod
    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def observe(self, state: EnvState) -> torch.Tensor:
        pass


class Link3DVelocity(Task):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        target_link_name: str,
        lin_vel_obs_scale: float,
        ang_vel_obs_scale: float,
        resampling_time: float,
        min_target_lin_vel: float,
        lin_vel_range: List[Tuple[float, float]],
        ang_vel_range: List[Tuple[float, float]],
        tracking_sigma: float,
        lin_vel_reward_scale: float,
        ang_vel_reward_scale: float,
        feet_air_time_reward_scale: float,
        lin_vel_reward_power: float,
        ang_vel_reward_power: float,
        feet_sensor_indices: List[int],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
        )
        env = gym.get_env(sim, 0)
        self.target_link_idx = gym.find_actor_rigid_body_handle(
            env,
            gym.get_actor_handle(env, 0),
            target_link_name,
        )
        assert self.target_link_idx != -1
        self.target_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.lin_vel_obs_scale = lin_vel_obs_scale
        self.ang_vel_obs_scale = ang_vel_obs_scale
        self.resampling_time = resampling_time
        self.min_target_lin_vel = min_target_lin_vel

        self.lin_vel_range = torch.tensor(lin_vel_range).to(self.device)
        self.ang_vel_range = torch.tensor(ang_vel_range).to(self.device)
        assert len(self.lin_vel_range) == 3
        assert len(self.ang_vel_range) == 3
        self.tracking_sigma = tracking_sigma

        self.lin_vel_reward_scale = lin_vel_reward_scale
        self.ang_vel_reward_scale = ang_vel_reward_scale
        self.lin_vel_reward_power = lin_vel_reward_power
        self.ang_vel_reward_power = ang_vel_reward_power
        self.feet_air_time_reward_scale = feet_air_time_reward_scale

        self.feet_sensor_indices = torch.tensor(
            feet_sensor_indices,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )

        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_sensor_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_sensor_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

    def resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        self.target_lin_vel[env_ids, :] = torch_rand_float(
            self.lin_vel_range[:, 0],
            self.lin_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)
        self.target_ang_vel[env_ids, :] = torch_rand_float(
            self.ang_vel_range[:, 0],
            self.ang_vel_range[:, 1],
            (len(env_ids), 3),
            device=self.device,
            generator=self.generator,
        ).squeeze(1)

        # set small commands to zero
        self.target_lin_vel[env_ids] *= (
            torch.norm(self.target_lin_vel[env_ids], dim=1) > self.min_target_lin_vel
        ).unsqueeze(1)

    def reset_idx(self, env_ids: torch.Tensor):
        self.resample_commands(env_ids)
        self.feet_air_time[env_ids] = 0.0

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        env_ids = (
            check_should_reset(
                time_s=state.episode_time,
                dt=state.sim_dt,
                reset_time_s=self.resampling_time,
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        self.resample_commands(env_ids)
        return {
            "angular_vel_err": self.get_ang_vel_err(state=state),
            "linear_vel_err": self.get_lin_vel_err(state=state),
        }

    def observe(self, state: EnvState) -> torch.Tensor:
        obs_terms = [
            self.target_lin_vel * self.lin_vel_obs_scale,
            self.target_ang_vel * self.ang_vel_obs_scale,
        ]
        obs = torch.cat(
            obs_terms,
            dim=-1,
        )
        return obs

    def get_link_local_lin_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.rigid_body_lin_vel[:, self.target_link_idx],
        )

    def get_link_local_ang_vel(self, state: EnvState):
        return quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.rigid_body_ang_vel[:, self.target_link_idx],
        )

    def get_lin_vel_err(self, state: EnvState):
        return (self.target_lin_vel - self.get_link_local_lin_vel(state=state)).norm(
            dim=-1
        )

    def get_ang_vel_err(self, state: EnvState):
        return (self.target_ang_vel - self.get_link_local_ang_vel(state=state)).norm(
            dim=-1
        )

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # Tracking of linear velocity commands (xy axes)
        lin_vel_reward = torch.exp(
            -self.get_lin_vel_err(state=state) ** self.lin_vel_reward_power
            / self.tracking_sigma
        )
        # Tracking of angular velocity commands (yaw)
        yaw_reward = torch.exp(
            -self.get_ang_vel_err(state=state) ** self.ang_vel_reward_power
            / self.tracking_sigma
        )
        assert state.force_sensor_tensor is not None
        contact = state.force_sensor_tensor[:, self.feet_sensor_indices, 2].abs() > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += state.sim_dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.target_lin_vel[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact_filt

        return {
            "lin_vel": lin_vel_reward * self.lin_vel_reward_scale,
            "yaw": yaw_reward * self.ang_vel_reward_scale,
            "feet_air_time": rew_airTime * self.feet_air_time_reward_scale,
        }


class Link2DVelocity(Link3DVelocity):
    def observe(self, state: EnvState) -> torch.Tensor:
        obs_terms = [
            self.target_lin_vel[:, :2] * self.lin_vel_obs_scale,
            self.target_ang_vel[:, [2]] * self.ang_vel_obs_scale,
        ]
        obs = torch.cat(
            obs_terms,
            dim=-1,
        )
        return obs

    def get_lin_vel_err(self, state: EnvState):
        return (
            self.target_lin_vel[:, :2] - self.get_link_local_lin_vel(state=state)[:, :2]
        ).norm(dim=-1)

    def get_ang_vel_err(self, state: EnvState):
        return (
            self.target_ang_vel[:, 2] - self.get_link_local_ang_vel(state=state)[:, 2]
        ).abs()


class Link6DVelocity(Link2DVelocity):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        z_height_range: Tuple[float, float],
        z_height_sigma: float,
        z_height_reward_scale: float,
        roll_range: Tuple[float, float],
        pitch_range: Tuple[float, float],
        gravity_sigma: float,
        gravity_reward_scale: float,
        **kwargs,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            **kwargs,
        )
        self.z_height_range = torch.tensor(z_height_range).to(self.device)
        self.roll_range = torch.tensor(roll_range).to(self.device)
        self.pitch_range = torch.tensor(pitch_range).to(self.device)
        self.z_height_sigma = z_height_sigma
        self.gravity_sigma = gravity_sigma
        self.z_height_reward_scale = z_height_reward_scale
        self.gravity_reward_scale = gravity_reward_scale
        self.target_z_height = torch.zeros((self.num_envs,), device=self.device)
        self.target_local_gravity = torch.zeros((self.num_envs, 3), device=self.device)

    def resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        super().resample_commands(env_ids)
        self.target_z_height[env_ids] = torch_rand_float(
            self.z_height_range[0],
            self.z_height_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        roll = torch_rand_float(
            self.roll_range[0],
            self.roll_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        pitch = torch_rand_float(
            self.pitch_range[0],
            self.pitch_range[1],
            (len(env_ids),),
            device=self.device,
            generator=self.generator,
        )
        # convert to gravity vector with trigonometry
        self.target_local_gravity[env_ids, 0] = torch.tan(roll)
        self.target_local_gravity[env_ids, 1] = torch.tan(pitch)
        self.target_local_gravity[env_ids, 2] = -1.0
        self.target_local_gravity /= self.target_local_gravity.norm(dim=1, keepdim=True)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        stats = super().step(state=state, control=control)
        stats["z_height_err"] = self.get_z_height_err(state=state)
        stats["gravity_err"] = self.get_gravity_err(state=state)
        return stats

    def observe(self, state: EnvState) -> torch.Tensor:
        return torch.cat(
            [
                super().observe(state=state),
                self.target_z_height[:, None],
                self.target_local_gravity,
            ],
            dim=-1,
        )

    def get_z_height_err(self, state: EnvState):
        # Penalize base height away from target
        link_height = torch.mean(
            state.rigid_body_pos[:, self.target_link_idx, [2]]
            - state.measured_terrain_heights,
            dim=1,
        )
        return torch.square(self.target_z_height - link_height)

    def get_gravity_err(self, state: EnvState):
        link_local_gravity = quat_rotate_inverse(
            state.rigid_body_xyzw_quat[:, self.target_link_idx],
            state.gravity / torch.linalg.norm(state.gravity, dim=1, keepdims=True),
        )
        return torch.square(self.target_local_gravity - link_local_gravity).sum(dim=1)

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        stats = super().reward(state=state, control=control)

        stats["z_height"] = (
            torch.exp(-self.get_z_height_err(state=state) / self.z_height_sigma)
            * self.z_height_reward_scale
        )
        stats["gravity"] = (
            torch.exp(-self.get_gravity_err(state=state) / self.gravity_sigma)
            * self.gravity_reward_scale
        )
        return stats


@torch.jit.script
def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    return pt3d.quaternion_to_matrix(quat)


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    return pt3d.axis_angle_to_quaternion(axis_angle)


@torch.jit.script
def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


@torch.jit.script
def quaternion_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    return pt3d.quaternion_to_axis_angle(quat)


@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    return pt3d.matrix_to_quaternion(matrix)


@torch.jit.script
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


class ReachingLinkTask(Task):
    """
    World Frame Target Pose Tracking Task.

    Supports relative pose observations with pose latency, and curriculum learning for error thresholds.
    """

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        link_name: str,
        pos_obs_scale: float,
        orn_obs_scale: float,
        pos_err_sigma: float,
        orn_err_sigma: float,
        pos_reward_scale: float,
        orn_reward_scale: float,
        pose_reward_scale: float,
        target_obs_times: List[float],
        sequence_sampler: SequenceSampler,
        pose_latency: float,
        position_obs_encoding: str = "linear",
        pose_latency_variability: Optional[Tuple[float, float]] = None,
        pose_latency_warmup_steps: int = 0,
        pose_latency_warmup_start: int = 0,
        position_noise: float = 0.0,
        euler_noise: float = 0.0,
        target_relative_to_base: bool = False,
        pos_sigma_curriculum: Optional[
            Dict[float, float]
        ] = None,  # maps from error to sigma
        orn_sigma_curriculum: Optional[
            Dict[float, float]
        ] = None,  # maps from error to sigma
        init_pos_curriculum_level: int = 0,
        init_orn_curriculum_level: int = 0,
        smoothing_dt_multiplier: float = 4.0,
        storage_device: str = "cpu",
        pos_obs_clip: Optional[float] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=torch.Generator(device=storage_device),
        )
        self.storage_device = storage_device
        self.link_name = link_name
        env = gym.get_env(sim, 0)
        actor = gym.get_actor_handle(env, 0)
        self.link_index = gym.find_actor_rigid_body_handle(env, actor, link_name)
        logging.info(f"Link index: {self.link_index} ({link_name})")
        if self.link_index == -1:
            raise ValueError(
                f"Could not find {self.link_name!r} in actor {gym.get_actor_name(env, 0)!r}"
            )
        self.curr_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.curr_target_rot_mat = torch.zeros(
            (self.num_envs, 3, 3), device=self.device
        )

        self.target_pos_seq = torch.zeros(
            (self.num_envs, sequence_sampler.episode_length, 3),
            device=self.storage_device,
        )
        self.target_rot_mat_seq = torch.zeros(
            (self.num_envs, sequence_sampler.episode_length, 3, 3),
            device=self.storage_device,
        )
        self.position_obs_encoding = position_obs_encoding
        assert self.position_obs_encoding in {"linear", "log-direction"}
        self.pos_obs_scale = pos_obs_scale
        self.orn_obs_scale = orn_obs_scale
        # NOTE since we're using rotation matrix representation
        # we don't need to clip orientation observations
        self.pos_obs_clip = pos_obs_clip
        self.pos_err_sigma = pos_err_sigma
        self.orn_err_sigma = orn_err_sigma
        self.pos_reward_scale = pos_reward_scale
        self.orn_reward_scale = orn_reward_scale
        self.pose_reward_scale = pose_reward_scale
        self.sequence_sampler = sequence_sampler
        self.target_obs_times = target_obs_times
        self.target_relative_to_base = target_relative_to_base

        self.past_pos_err = torch.ones((self.num_envs,), device=self.device)
        self.past_orn_err = torch.ones((self.num_envs,), device=self.device)
        self.smoothing_dt_multiplier = smoothing_dt_multiplier
        self.pos_sigma_curriculum = pos_sigma_curriculum
        self.orn_sigma_curriculum = orn_sigma_curriculum
        self.pos_sigma_curriculum_level = (
            0 if init_pos_curriculum_level is None else init_pos_curriculum_level
        )
        self.orn_sigma_curriculum_level = (
            0 if init_orn_curriculum_level is None else init_orn_curriculum_level
        )
        if self.pos_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.pos_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.pos_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.pos_err_sigma = list(self.pos_sigma_curriculum.values())[
                self.pos_sigma_curriculum_level
            ]
            self.past_pos_err *= list(self.pos_sigma_curriculum.keys())[
                self.pos_sigma_curriculum_level
            ]
        if self.orn_sigma_curriculum is not None:
            # make sure the curriculum is sorted
            self.orn_sigma_curriculum = dict(
                map(
                    lambda x: (float(x[0]), float(x[1])),
                    sorted(
                        self.orn_sigma_curriculum.items(),
                        key=lambda x: x[0],
                        reverse=True,
                    ),
                )
            )
            self.orn_err_sigma = list(self.orn_sigma_curriculum.values())[
                self.orn_sigma_curriculum_level
            ]
            self.past_orn_err *= list(self.orn_sigma_curriculum.keys())[
                self.orn_sigma_curriculum_level
            ]
        self.pose_latency = pose_latency
        self.pose_latency_frames = (
            int(np.rint(pose_latency / self.sequence_sampler.dt)) + 1
        )
        self.pose_latency_frame_variability = (
            pose_latency_variability
            if pose_latency_variability is None
            else (
                int(np.rint(pose_latency_variability[0] / self.sequence_sampler.dt)),
                int(np.rint(pose_latency_variability[1] / self.sequence_sampler.dt)),
            )
        )

        self.link_pose_history = torch.zeros(
            (self.num_envs, self.pose_latency_frames, 4, 4),
            device=self.device,
        )
        self.link_pose_history[..., :, :] = torch.eye(4, device=self.device)
        self.root_pose_history = torch.zeros(
            (self.num_envs, self.pose_latency_frames, 4, 4),
            device=self.device,
        )
        self.root_pose_history[..., :, :] = torch.eye(4, device=self.device)
        self.pose_latency_warmup_steps = pose_latency_warmup_steps
        self.pose_latency_warmup_start = pose_latency_warmup_start
        self.steps = 0
        self.position_noise = position_noise
        self.euler_noise = euler_noise

    def reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        env_origins = torch.stack(
            [
                torch.tensor([env_origin.x, env_origin.y, env_origin.z])
                for env_origin in map(
                    lambda x: self.gym.get_env_origin(self.gym.get_env(self.sim, x)),
                    env_ids,
                )
            ],
            dim=0,
        ).to(
            self.storage_device
        )  # (num_envs, 3)
        # get root positions
        pos_seq, rot_mat_seq = self.sequence_sampler.sample(
            seed=int(
                torch.randint(
                    low=0,
                    high=2**32 - 1,
                    size=(1,),
                    generator=self.generator,
                    device=self.storage_device,
                )
                .cpu()
                .item()
            ),
            batch_size=len(env_ids),
        )
        assert rot_mat_seq.shape == (
            len(env_ids),
            self.sequence_sampler.episode_length,
            3,
            3,
        )

        assert pos_seq.shape == (len(env_ids), self.sequence_sampler.episode_length, 3)
        pos_seq = pos_seq.to(self.storage_device) + env_origins.unsqueeze(1)
        rot_mat_seq = rot_mat_seq.to(self.storage_device)
        device_env_ids = env_ids.to(self.storage_device)
        self.target_pos_seq[device_env_ids] = pos_seq
        self.target_rot_mat_seq[device_env_ids] = rot_mat_seq

        self.curr_target_pos[env_ids, :] = pos_seq[:, 0, :].to(self.device)
        self.curr_target_rot_mat[env_ids, :] = rot_mat_seq[:, 0, :].to(self.device)
        # update curriculum
        if self.pos_sigma_curriculum is not None:
            avg_pos_err = self.past_pos_err.mean().item()
            # find the first threshold that is greater than the average error
            for level, (threshold, sigma) in enumerate(
                self.pos_sigma_curriculum.items()
            ):
                if avg_pos_err < threshold:
                    self.pos_err_sigma = sigma
                    self.pos_sigma_curriculum_level = level
        if self.orn_sigma_curriculum is not None:
            avg_orn_err = self.past_orn_err.mean().item()
            # find the first threshold that is greater than the average error
            for level, (threshold, sigma) in enumerate(
                self.orn_sigma_curriculum.items()
            ):
                if avg_orn_err < threshold:
                    self.orn_err_sigma = sigma
                    self.orn_sigma_curriculum_level = level

        # update pose history
        self.link_pose_history[device_env_ids, :, :, :] = torch.eye(
            4, device=self.device
        )
        self.root_pose_history[device_env_ids, :, :, :] = torch.eye(
            4, device=self.device
        )

    def get_targets_at_times(
        self,
        times: torch.Tensor,
        sim_dt: float,
    ):
        episode_step = torch.clamp(
            # (torch.zeros_like(times) / sim_dt).long(),
            (times / sim_dt).long(),
            min=0,
            max=self.target_pos_seq.shape[1] - 1,
        )
        episode_step = torch.clamp(
            episode_step, min=0, max=self.target_pos_seq.shape[1] - 1
        ).to(self.storage_device)
        env_idx = torch.arange(0, self.num_envs)
        return (
            self.target_pos_seq[env_idx, episode_step].to(self.device),
            self.target_rot_mat_seq[env_idx, episode_step].to(self.device),
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        (
            self.curr_target_pos[:, :],
            self.curr_target_rot_mat[:, :],
        ) = self.get_targets_at_times(
            times=state.episode_time,
            sim_dt=state.sim_dt,
        )
        pos_err = self.get_pos_err(state=state)
        orn_err = self.get_orn_err(state=state)
        # moving average of the error
        smoothing = state.sim_dt * self.smoothing_dt_multiplier
        self.past_pos_err = (1 - smoothing) * self.past_pos_err + smoothing * pos_err
        self.past_orn_err = (1 - smoothing) * self.past_orn_err + smoothing * orn_err

        self.link_pose_history = torch.cat(
            [
                self.link_pose_history[:, 1:],
                self.get_link_pose(state=state).unsqueeze(1),
            ],
            dim=1,
        )
        self.root_pose_history = torch.cat(
            [
                self.root_pose_history[:, 1:],
                state.root_pose.clone().unsqueeze(1),
            ],
            dim=1,
        )
        self.steps += 1
        return {
            "pos_err": pos_err,
            "orn_err": orn_err,
            "smoothed_pos_err": self.past_pos_err.clone(),
            "smoothed_orn_err": self.past_orn_err.clone(),
            "pos_sigma_level": torch.ones_like(pos_err, device=self.device)
            * self.pos_sigma_curriculum_level,
            "orn_sigma_level": torch.ones_like(orn_err, device=self.device)
            * self.orn_sigma_curriculum_level,
            "pose_latency": torch.ones_like(pos_err, device=self.device)
            * self.get_latency_scheduler()
            * self.pose_latency,
        }

    def get_pos_err(self, state: EnvState) -> torch.Tensor:
        return torch.sum(
            torch.square(self.curr_target_pos - self.get_link_pos(state=state)),
            dim=1,
        ).sqrt()

    def get_orn_err(self, state: EnvState) -> torch.Tensor:
        link_rot_mat = self.get_link_rot_mat(state=state)
        # rotation from link to target
        rot_err_mat = self.curr_target_rot_mat @ link_rot_mat.transpose(1, 2)

        trace = torch.diagonal(rot_err_mat, dim1=-2, dim2=-1).sum(dim=-1)
        # to prevent numerical instability, clip the trace to [-1, 3]
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * np.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * np.pi - rotation_magnitude,
        )
        return rotation_magnitude

    def get_target_pose(self, times: torch.Tensor, sim_dt: float):
        # returns the current target pose in the local frame of the robot
        target_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )

        pos, rot_mat = self.get_targets_at_times(times=times, sim_dt=sim_dt)
        target_pose[..., :3, 3] = pos
        target_pose[..., :3, :3] = rot_mat
        return target_pose

    def get_link_rot_mat(self, state: EnvState):
        return quaternion_to_matrix(
            state.rigid_body_xyzw_quat[:, self.link_index][:, [3, 0, 1, 2]]
        )

    def get_link_pos(self, state: EnvState):
        return state.rigid_body_pos[:, self.link_index]

    def get_link_pose(self, state: EnvState):
        # returns the current link pose in the local frame of the robot
        link_pose = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        )
        link_pose[..., :3, 3] = self.get_link_pos(state=state)
        # pt3d quaternion convention is wxyz
        link_pose[..., :3, :3] = self.get_link_rot_mat(state=state)
        return link_pose

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        # compute reward using the current pose
        pos_reward = torch.exp(
            -(self.get_pos_err(state=state) ** 2) / self.pos_err_sigma
        )
        orn_reward = torch.exp(-self.get_orn_err(state=state) / self.orn_err_sigma)
        return {
            "pos": pos_reward * self.pos_reward_scale,
            "orn": orn_reward * self.orn_reward_scale,
            "pose": (pos_reward * orn_reward) * self.pose_reward_scale,
        }

    def get_latency_scheduler(self) -> float:
        return (
            min(
                max(
                    (self.steps - self.pose_latency_warmup_start)
                    / self.pose_latency_warmup_steps,
                    0.0,
                ),
                1.0,
            )
            if self.pose_latency_warmup_steps > 0
            else 1.0
        )

    def observe(self, state: EnvState) -> torch.Tensor:
        global_target_pose = torch.stack(
            [
                self.get_target_pose(
                    times=state.episode_time + t_offset,
                    sim_dt=state.sim_dt,
                )
                for t_offset in self.target_obs_times
            ],
            dim=1,
        )  # (num_envs, num_obs, 4, 4)
        # get the most outdated pose, to account for latency
        latency_idx = int(
            np.rint(self.get_latency_scheduler() * self.pose_latency_frames)
        )  # number of frames to wait
        if self.pose_latency_frame_variability is not None:
            latency_idx += int(
                torch.randint(
                    low=self.pose_latency_frame_variability[0],
                    high=self.pose_latency_frame_variability[1] + 1,
                    size=(1,),
                    device=self.storage_device,
                    generator=self.generator,
                )
                .cpu()
                .item()
            )
        # min is 1, since this index is negated to access last `latency_idx`th frame
        latency_idx = max(1, min(latency_idx, self.pose_latency_frames - 1))
        observation_link_pose = (
            self.root_pose_history[:, -latency_idx]
            if self.target_relative_to_base
            else self.link_pose_history[:, -latency_idx]
        ).clone()  # (num_envs, 4, 4), clone otherwise sim state will be modified
        if self.position_noise > 0 or self.euler_noise > 0:
            noise_transform = torch.zeros((self.num_envs, 4, 4), device=self.device)
            noise_transform[..., [0, 1, 2, 3], [0, 1, 2, 3]] = 1.0
            if self.position_noise > 0:
                noise_transform[..., :3, 3] = (
                    torch.randn((self.num_envs, 3), device=self.device)
                    * self.position_noise
                )
            if self.euler_noise > 0:
                euler_noise = (
                    torch.randn((self.num_envs, 3), device=self.device)
                    * self.euler_noise
                )
                noise_transform[..., :3, :3] = pt3d.euler_angles_to_matrix(
                    euler_noise, convention="XYZ"
                )
            observation_link_pose = noise_transform @ observation_link_pose
        local_target_pose = (
            torch.linalg.inv(observation_link_pose[:, None, :, :]) @ global_target_pose
        )
        if self.position_obs_encoding == "linear":
            pos_obs = (local_target_pose[..., :3, 3] * self.pos_obs_scale).view(
                self.num_envs, -1
            )
        elif self.position_obs_encoding == "log-direction":
            distance = (
                torch.linalg.norm(local_target_pose[..., :3, 3], dim=-1, keepdim=True)
                + 1e-8
            )
            direction = local_target_pose[..., :3, 3] / distance
            pos_obs = torch.cat(
                (
                    torch.log(distance * self.pos_obs_scale).reshape(self.num_envs, -1),
                    direction.reshape(
                        self.num_envs, -1
                    ),  # direction is already in normalized range
                ),
                dim=-1,
            )
        else:
            raise ValueError(
                f"Unknown position observation encoding: {self.position_obs_encoding!r}"
            )

        if self.pos_obs_clip is not None:
            pos_obs = torch.clamp(pos_obs, -self.pos_obs_clip, self.pos_obs_clip)
        orn_obs = (
            pt3d.matrix_to_rotation_6d(local_target_pose[..., :3, :3])
            * self.orn_obs_scale
        ).view(self.num_envs, -1)

        relative_pose_obs = torch.cat((pos_obs, orn_obs), dim=1)
        # NOTE after episode resetting, the first pose will be outdated
        # (this is a quirk of isaacgym, where state resets don't apply until the
        # next physics step), we will have to wait for `pose_latency` seconds
        # to get the first pose so just return special values for such cases
        waiting_for_pose_mask = (
            (observation_link_pose == torch.eye(4, device=self.device))
            .all(dim=-1)
            .all(dim=-1)
        )
        relative_pose_obs[waiting_for_pose_mask] = -1.0

        return relative_pose_obs

    def visualize(self, state: EnvState, viewer: gymapi.Viewer, vis_env_ids: List[int]):
        pos_err = self.get_pos_err(state=state)
        cm = plt.get_cmap("inferno")
        target_quats = (
            matrix_to_quaternion(self.curr_target_rot_mat)
            .cpu()
            .numpy()[:, [1, 2, 3, 0]]
        )
        link_pose = self.get_link_pose(state=state)
        curr_quats = (
            matrix_to_quaternion(link_pose[:, :3, :3])[:, [1, 2, 3, 0]].cpu().numpy()
        )
        link_pose = link_pose.cpu().numpy()
        for i in vis_env_ids:
            env = self.gym.get_env(self.sim, i)
            rgb = list(cm(max(1 - pos_err[i].item(), 0)))
            target_pose = gymapi.Transform(
                p=gymapi.Vec3(*self.curr_target_pos[i].cpu().numpy().tolist()),
                r=gymapi.Quat(*target_quats[i].tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                target_pose,
            )
            curr_pose = gymapi.Transform(
                p=gymapi.Vec3(*link_pose[i, :3, 3].copy().tolist()),
                r=gymapi.Quat(*curr_quats[i].copy().tolist()),
            )
            gymutil.draw_lines(
                gymutil.AxesGeometry(scale=0.2),
                self.gym,
                viewer,
                env,
                curr_pose,
            )
            vertices = np.array(
                [
                    *self.curr_target_pos[i].cpu().numpy().tolist(),
                    *link_pose[i, :3, 3].tolist(),
                ]
            ).astype(np.float32)
            colors = np.array(rgb).astype(np.float32)
            self.gym.add_lines(
                viewer,
                env,
                1,  # num lines
                vertices,  # vertices
                colors,  # color
            )

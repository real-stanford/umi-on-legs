import dataclasses
import os
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from legged_gym.env.isaacgym.state import EnvState
from legged_gym.env.obs import ObservationAttribute


@dataclasses.dataclass
class Control:
    torque: torch.Tensor  # (num_envs, control_dim) the current torque
    buffer: (
        torch.Tensor
    )  # (num_envs, buffer_len, control_dim) the buffer of past targets

    def push(self, action: torch.Tensor):
        self.buffer = torch.cat((action[:, None, :], self.buffer[:, :-1]), dim=1)

    @property
    def prev_action(self):
        return self.buffer[:, 1]

    @property
    def action(self):
        return self.buffer[:, 0]

    @property
    def ctrl_dim(self) -> int:
        return self.buffer.shape[-1]


class PDController:
    def __init__(
        self,
        control_dim: int,
        device: str,
        torque_limit: torch.Tensor,
        kp: torch.Tensor,
        kd: torch.Tensor,
        num_envs: int,
        seed: int = 0,
        decimation_count: Union[int, Tuple[int, int]] = (3, 5),
        scale: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ):
        self.scale = (
            torch.ones((1, control_dim), device=device)
            if scale is None
            else scale.to(device)
        )
        self.offset = (
            torch.zeros((1, control_dim), device=device)
            if offset is None
            else offset.to(device)
        )
        self.np_random = np.random.RandomState(seed)
        self.torque_limit = torque_limit.to(device)
        self.device = device
        self.decimation_count_range = (
            decimation_count
            if not isinstance(decimation_count, int)
            else (decimation_count, decimation_count)
        )
        self.control_dim = control_dim
        self.kp = kp.to(self.device)[None, :].float()
        self.kd = kd.to(self.device)[None, :].float()
        self.num_envs = num_envs
        self.prev_normalized_target = torch.zeros((1, control_dim), device=self.device)

    @property
    def decimation_count(self) -> int:
        return int(
            self.np_random.randint(
                self.decimation_count_range[0],
                self.decimation_count_range[1] + 1,
            )
        )

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        return self.compute_torque(
            normalized_action=action * self.scale
            + self.offset.repeat(action.shape[0], 1),
            state=state,
        )

    def compute_torque(self, normalized_action: torch.Tensor, state: EnvState):
        """
        normalized_action: (control_dim, ) after __call__
        """
        self.prev_normalized_target = normalized_action
        return torch.clip(
            normalized_action, min=-self.torque_limit, max=self.torque_limit
        )


class VelocityController(PDController):
    def __init__(self, **kwargs):
        kwargs["require_prev_state"] = True
        super().__init__(**kwargs)

    def compute_torque(
        self,
        normalized_action: torch.Tensor,
        state: EnvState,
    ):
        dt = state.sim_dt

        curr_vel = state.dof_vel.clone()
        prev_vel = state.dof_vel.clone()
        torques = (
            self.kp * (normalized_action - curr_vel)
            - self.kd * (curr_vel - prev_vel) / dt
        )
        return super().compute_torque(torques, state)


class PositionController(PDController):
    def compute_torque(
        self,
        normalized_action: torch.Tensor,
        state: EnvState,
    ):
        curr_pos = state.dof_pos.clone()
        curr_vel = state.dof_vel.clone()
        assert normalized_action.shape == curr_pos.shape
        assert curr_vel.shape == curr_pos.shape
        if normalized_action.shape[0] != self.kp.shape[0]:
            self.kp = self.kp.repeat(normalized_action.shape[0], 1)
            self.kd = self.kd.repeat(normalized_action.shape[0], 1)
        torques = self.kp * (normalized_action - curr_pos) - self.kd * curr_vel
        return super().compute_torque(torques, state)


class PositionControllerWithExtraFixedAction(PositionController):
    def __init__(self, extra_action: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.extra_action = extra_action.to(self.device)

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        action = torch.cat(
            (
                action,
                self.extra_action[None, :].repeat(action.shape[0], 1),
            ),
            dim=1,
        )
        return super().__call__(action, state)


class PositionControllerWithExtraOscillatingAction(PositionController):
    def __init__(
        self, target_a: torch.Tensor, target_b: torch.Tensor, frequency: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.target_a = target_a.to(self.device)[None, :]
        self.target_b = target_b.to(self.device)[None, :]
        self.frequency = frequency

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        t = state.episode_time
        target = (
            self.target_a
            + (self.target_b - self.target_a) * torch.sin(t * self.frequency)[:, None]
        )
        action = torch.cat(
            (action, target),
            dim=1,
        )
        return super().__call__(action, state)


class PositionControllerWithExtraActionList(PositionController):
    def __init__(self, targets: List[torch.Tensor], time_per_target: float, **kwargs):
        super().__init__(**kwargs)
        self.targets = torch.stack(targets).to(self.device)
        self.time_per_target = time_per_target

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        t = state.episode_time
        idx_float = (t / self.time_per_target) % self.targets.shape[0]
        idx_prev = idx_float.floor().long()
        idx_next = idx_float.ceil().long()
        idx_next[idx_next == idx_prev] += 1
        idx_next = idx_next % self.targets.shape[0]
        alpha = (idx_float - idx_prev.float())[:, None]
        target = self.targets[idx_prev] * (1 - alpha) + self.targets[idx_next] * alpha
        action = torch.cat(
            (action, target),
            dim=1,
        )
        return super().__call__(action, state)


class PositionControllerWithExtraActionSampler(PositionController):
    def __init__(
        self,
        target_bounds: torch.Tensor,
        time_per_target: float,
        seed: int,
        sim_dt: float,
        clip_curriculum: Optional[Dict[int, torch.Tensor]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_bounds = target_bounds.to(self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.steps_per_target = int(np.rint(time_per_target / sim_dt))
        self.step_counter = 0
        self.curriculum_steps = None
        if clip_curriculum is not None:
            self.curriculum_steps = sorted(clip_curriculum.keys())
            self.curriculum_clip_values = torch.stack(
                [clip_curriculum[step] for step in self.curriculum_steps]
            ).to(self.device)
            assert self.curriculum_clip_values.shape == (
                len(self.curriculum_steps),
                2,
                self.extra_ctrl_dim,
            )
        self.prev_targets = self.sample_joints()
        self.next_targets = self.sample_joints()

    def sample_joints(self):
        joints = (
            torch.rand(
                (self.num_envs, self.extra_ctrl_dim),
                device=self.device,
                generator=self.generator,
            )
            * (self.target_bounds[None, 1] - self.target_bounds[None, 0])
            + self.target_bounds[None, 0]
        )
        if self.curriculum_steps is not None:
            curriculum_level = self.curriculum_steps[
                max(
                    0,
                    np.searchsorted(
                        self.curriculum_steps, self.step_counter, side="left"
                    )
                    - 1,
                )
            ]
            joints = torch.clip(
                joints,
                min=self.curriculum_clip_values[curriculum_level, 0],
                max=self.curriculum_clip_values[curriculum_level, 1],
            )
        return joints

    @property
    def extra_ctrl_dim(self) -> int:
        return self.target_bounds.shape[1]

    def get_current_target(self):
        alpha = (self.step_counter % self.steps_per_target) / self.steps_per_target
        return self.prev_targets * (1 - alpha) + self.next_targets * alpha

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        if self.step_counter % self.steps_per_target == 0:
            self.prev_targets = self.next_targets
            self.next_targets = self.sample_joints()
        target = self.get_current_target()
        action = torch.cat(
            (action, target),
            dim=1,
        )
        self.step_counter += 1
        return super().__call__(action, state)


class LearnedPositionController(PositionController):
    def __init__(
        self,
        obs_attrs: Dict[str, ObservationAttribute],
        policy_ctrl_period: float,
        policy: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ckpt_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if policy is None:
            assert ckpt_path is not None
            self.policy = self.setup_policy(ckpt_path, torch.device(self.device))
        else:
            self.policy = policy.to(self.device)
        self.obs_attrs = {
            k: v for k, v in sorted(obs_attrs.items(), key=lambda x: x[0])
        }
        self.last_action = torch.zeros((1, self.control_dim), device=self.device)
        self.last_action_time = torch.zeros((1,), device=self.device)
        self.policy_ctrl_period = policy_ctrl_period

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        if self.last_action_time.shape != state.episode_time.shape:
            self.last_action_time = state.episode_time.clone()
        if self.last_action.shape[0] != state.episode_time.shape[0]:
            self.last_action = self.last_action.repeat(state.episode_time.shape[0], 1)
        resetted_mask = self.last_action_time > state.episode_time
        self.last_action_time[resetted_mask] = (
            state.episode_time[resetted_mask] - self.policy_ctrl_period
        )
        self.last_action[resetted_mask] = 0
        if (
            (state.episode_time - self.last_action_time)
            >= self.policy_ctrl_period - 1e-3
        ).any():
            base_policy_env_obs = torch.cat(
                [obs_attr(struct=state) for obs_attr in self.obs_attrs.values()],
                dim=1,
            )
            if len(self.last_action) != len(action):
                self.last_action = self.last_action.repeat(len(action), 1)
            obs = torch.cat(
                (
                    base_policy_env_obs,
                    action,
                    self.last_action,
                ),
                dim=1,
            )
            self.last_action = self.policy(obs)
            self.last_action_time = state.episode_time.clone()
        reduced_state = dataclasses.replace(
            state,
            dof_pos=state.dof_pos[:, : self.control_dim],
            dof_vel=state.dof_vel[:, : self.control_dim],
        )
        return super().__call__(self.last_action, reduced_state)

    @staticmethod
    def setup_policy(ckpt_path: str, device: torch.device):
        config_str = open(
            os.path.join(os.path.dirname(ckpt_path), "config.yaml"), "r"
        ).read()
        config_str = re.sub(r"cuda:\d+", "cpu", config_str)
        with open("/tmp/config.yaml", "w") as f:
            f.write(config_str)
        config = OmegaConf.to_container(
            OmegaConf.load("/tmp/config.yaml"),
            resolve=True,
        )
        config["wandb"]["mode"] = "offline"  # type: ignore
        for k in config.keys():
            try:
                config[k] = config[k]["value"]
                del config[k]["value"]
                del config[k]["desc"]
            except:
                pass
        device = torch.device(device)
        actor_critic = hydra.utils.instantiate(config["runner"]["alg"]["actor_critic"])
        ckpt = torch.load(ckpt_path, map_location=device)
        actor_critic.load_state_dict(ckpt["model_state_dict"])
        actor_critic.eval()
        actor_critic = actor_critic.to(device)
        placeholder_obs = torch.rand(
            int(config["env"]["cfg"]["env"]["num_observations"]), device=device
        )
        # Tracing the model with example input
        traced_actor = torch.jit.trace(actor_critic.actor, placeholder_obs)
        # Invoking torch.jit.freeze
        return torch.jit.freeze(traced_actor)


class PositionControllerWithLearnedBasePolicy(PositionController):
    def __init__(
        self,
        base_policy: LearnedPositionController,
        base_command_dims: int,
        base_command_offset: torch.Tensor,
        base_command_scale: torch.Tensor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_policy = base_policy
        self.base_command_dims = base_command_dims
        self.base_command_offset = base_command_offset
        self.base_command_scale = base_command_scale
        assert self.base_policy.decimation_count == self.decimation_count

    def __call__(
        self,
        action: torch.Tensor,
        state: EnvState,
    ):
        """
        action: (num_envs, control_dim) from the network
        """
        base_torque = self.base_policy(
            action=action[:, -self.base_command_dims :] * self.base_command_scale
            + self.base_command_offset,
            state=state,
        )
        reduced_state = dataclasses.replace(
            state,
            dof_pos=state.dof_pos[:, -self.control_dim :],
            dof_vel=state.dof_vel[:, -self.control_dim :],
        )
        nonbase_torque = super().__call__(
            action[:, : -self.base_command_dims], reduced_state
        )
        return torch.cat(
            (
                base_torque,
                nonbase_torque,
            ),
            dim=1,
        )

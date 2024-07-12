from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.env.isaacgym.control import Control
from legged_gym.env.isaacgym.state import EnvState
from legged_gym.env.isaacgym.task import Task

import pytorch3d.transforms as pt3d


class Constraint(Task):
    """
    Constraints are a special type of task that can be used to enforce
    constraints on the environment.
    Violations are used to incentivize hard constraints, while penalties
    are used to incentivize soft constraints.
    Finally, depending on the type of constraint, the action space can be
    modified to enforce the constraint.
    """

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        skip_stats: bool = True,
    ):
        super().__init__(gym=gym, sim=sim, device=device, generator=generator)
        self.violation_weight = violation_weight
        self.penalty_weight = penalty_weight
        self.terminate_on_violation = terminate_on_violation
        self.skip_stats = skip_stats

    def reward(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        retval = {}
        if self.violation_weight != 0:
            retval["hard_violation"] = (
                self.check_violation(state=state, control=control)
                * self.violation_weight
            )
        if self.penalty_weight != 0:
            retval["soft_penalty"] = (
                self.compute_penalty(state=state, control=control) * self.penalty_weight
            )
        return retval

    @abstractmethod
    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        """
        returns bool tensors indicating if the constraint is violated
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError()

    def check_termination(self, state: EnvState, control: Control) -> torch.Tensor:
        if self.terminate_on_violation:
            return self.check_violation(state=state, control=control)
        return torch.zeros(state.dof_pos.shape[0], device=self.device, dtype=torch.bool)


class JointLimit(Constraint):
    """
    Enforce actions to stay within its limits.
    """

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        upper: torch.Tensor,
        lower: torch.Tensor,
        violation_scale: float,
        penalty_scale: float,
        joint_names: Optional[List[str]] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.upper = upper[None, :]
        self.lower = lower[None, :]
        self.mid = (upper + lower) / 2.0
        assert (upper > lower).all(), "upper limit must be greater than lower limit"
        self.range = upper - lower
        # violation scale is the upper bound
        self.violation_scale = violation_scale
        # penalty scale is the lower bound where penalization kicks in
        self.penalty_scale = penalty_scale
        all_joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.joint_names = joint_names if joint_names is not None else all_joint_names
        self.joint_indices = torch.tensor(
            [all_joint_names.index(joint_name) for joint_name in self.joint_names]
        ).to(self.device, torch.long)
        assert (
            self.joint_indices != -1
        ).all(), f"joint names can't be found: {joint_names}"

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.logical_or(
            state.dof_pos[:, self.joint_indices] > self.upper * self.violation_scale,
            state.dof_pos[:, self.joint_indices] < self.lower * self.violation_scale,
        ).any(
            dim=1,
        )

    def compute_usage_range(self, value):
        return (value - self.mid).abs() / self.range

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        # Penalize dof positions too close to the limit
        penalizing_lower = self.mid - 0.5 * self.range * self.penalty_scale
        penalizing_upper = self.mid + 0.5 * self.range * self.penalty_scale
        out_of_limits = -(state.dof_pos[:, self.joint_indices] - penalizing_lower).clip(
            max=0.0
        )  # lower limit
        out_of_limits += (state.dof_pos[:, self.joint_indices] - penalizing_upper).clip(
            min=0.0
        )
        return torch.sum(out_of_limits, dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        # also compute, given the current action, how much of the action space is being used
        range_usage = self.compute_usage_range(value=state.dof_pos)
        stats = {
            "violation": violation,
            "avg_range_usage": range_usage.mean(dim=1),
        }
        for joint_idx, joint_name in zip(self.joint_indices, self.joint_names):
            stats[f"usage/{joint_name}"] = range_usage[:, joint_idx]
        return stats


class ActionRateLimit(Constraint):
    """
    Enforce an upper bound on the rate of change of actions
    """

    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_action_rate: float,
        power: float,
        joint_names: Optional[List[str]] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.power = power
        self.violation_action_rate = violation_action_rate
        all_joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.joint_indices = torch.tensor(
            [all_joint_names.index(joint_name) for joint_name in joint_names]
            if joint_names is not None
            else range(len(all_joint_names))
        ).to(self.device, torch.long)
        self.joint_names = joint_names if joint_names is not None else all_joint_names
        assert (
            self.joint_indices != -1
        ).all(), f"joint names can't be found: {joint_names}"

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        prev_action = control.prev_action
        action_rate = (control.action - prev_action)[:, self.joint_indices]
        return (action_rate.abs() > self.violation_action_rate).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        delta_action = (control.action - control.prev_action)[:, self.joint_indices]
        return torch.sum(delta_action.abs() ** self.power, dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        action_rate = control.action - control.prev_action
        stats = {
            "violation": violation,
            "avg_action_rate": action_rate.abs().mean(dim=1),
        }
        for joint_idx, joint_name in zip(self.joint_indices, self.joint_names):
            if joint_idx >= action_rate.shape[1]:
                break
            stats[f"usage/{joint_name}"] = action_rate[:, joint_idx].abs()
        return stats


class TorqueLimit(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_torque: float,
        penalty_scale: float,
        power: float,
        joint_names: Optional[List[str]] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_torque = violation_torque
        self.penalization_torque = penalty_scale * violation_torque
        all_joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.joint_names = joint_names if joint_names is not None else all_joint_names
        self.joint_indices = torch.tensor(
            [all_joint_names.index(joint_name) for joint_name in self.joint_names]
        ).to(self.device, torch.long)
        assert (
            self.joint_indices != -1
        ).all(), f"joint names can't be found: {joint_names}"
        self.power = power

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        return (
            control.torque[..., self.joint_indices].abs() > self.violation_torque
        ).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.sum(
            control.torque[..., self.joint_indices].abs() ** self.power, dim=1
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        stats = {
            "violation": violation,
            "avg_torque": control.torque[..., self.joint_indices].abs().mean(dim=1),
        }
        for joint_idx, joint_name in zip(self.joint_indices, self.joint_names):
            stats[f"usage/{joint_name}"] = control.torque[:, joint_idx].abs()
        return stats


class JointVelocity(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_vel: float,
        penalty_scale: float,
        power: float,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_vel = violation_vel
        self.penalization_vel = penalty_scale * violation_vel
        self.joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.power = power

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        return (state.dof_vel.abs() > self.violation_vel).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.sum(state.dof_vel.abs() ** self.power, dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        stats = {
            "violation": violation,
            "avg_velocity": state.dof_vel.abs().mean(dim=1),
        }
        for joint_idx, joint_name in enumerate(self.joint_names):
            stats[f"usage/{joint_name}"] = state.dof_vel[:, joint_idx].abs()
        return stats


class JointAccelerationLimit(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_acc: float,
        penalty_scale: float,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_acc = violation_acc
        self.penalization_acc = penalty_scale * violation_acc
        self.joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )

    def get_acc(self, state: EnvState):
        return (state.prev_dof_vel - state.dof_vel) / state.sim_dt

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        return (self.get_acc(state=state).abs() > self.violation_acc).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.sum(torch.square(self.get_acc(state=state)), dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        stats = {
            "violation": violation,
            "avg_acceleration": self.get_acc(state=state).abs().mean(dim=1),
        }
        for joint_idx, joint_name in enumerate(self.joint_names):
            stats[f"usage/{joint_name}"] = self.get_acc(state=state)[:, joint_idx].abs()
        return stats


class Collision(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        violation_force_norm: float,
        penalty_scale: float,
        link_names: List[str],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_force_norm = violation_force_norm
        self.penalization_force_norm = penalty_scale * violation_force_norm
        self.link_names = link_names
        all_link_names = self.gym.get_actor_rigid_body_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        missing_links = set(link_names) - set(all_link_names)
        assert (
            len(missing_links) == 0
        ), f"collision link names can't be found: {missing_links}"
        self.link_indices = torch.tensor(
            [all_link_names.index(link_name) for link_name in link_names]
        ).to(self.device, torch.long)
        assert (
            self.link_indices != -1
        ).all(), f"penalty link names can't be found: {link_names}"

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        return (
            torch.norm(
                state.contact_forces[:, self.link_indices, :],
                dim=-1,
            )
            > self.violation_force_norm
        ).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.sum(
            (
                torch.norm(
                    state.contact_forces[:, self.link_indices, :],
                    dim=-1,
                )
                > self.penalization_force_norm
            ),
            dim=1,
        ).float()

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        violation = self.check_violation(state=state, control=control)
        stats = {
            "violation": violation,
            "avg_force": torch.norm(
                state.contact_forces[:, self.link_indices, :],
                dim=-1,
            ).mean(dim=1),
        }
        for link_idx, link_name in zip(self.link_indices, self.link_names):
            stats[f"usage/{link_name}"] = torch.norm(
                state.contact_forces[:, link_idx, :],
                dim=-1,
            )
        return stats


class FeetDragging(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        penalty_feet_drag_height: float,
        violation_feet_drag_height: float,
        violation_feet_drag_speed: float,
        feet_drag_sigma: float,
        feet_rigid_body_indices: List[int],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.violation_feet_drag_height = violation_feet_drag_height
        self.violation_feet_drag_speed = violation_feet_drag_speed
        self.penalty_feet_drag_height = penalty_feet_drag_height
        self.feet_drag_sigma = feet_drag_sigma
        self.feet_rigid_body_indices = feet_rigid_body_indices

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        feet_height = state.rigid_body_pos[:, self.feet_rigid_body_indices, 2]
        feet_planar_speed = torch.sum(
            torch.square(state.rigid_body_lin_vel[:, self.feet_rigid_body_indices, :2]),
            dim=2,
        )
        return torch.logical_and(
            feet_height < self.violation_feet_drag_height,
            feet_planar_speed > self.violation_feet_drag_speed,
        ).any(dim=-1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        feet_height_diff = torch.clip(
            state.rigid_body_pos[:, self.feet_rigid_body_indices, 2]
            - self.penalty_feet_drag_height,
            max=0,
        )  # (num_envs, num_feet),
        feet_planar_speed = torch.sum(
            torch.square(state.rigid_body_lin_vel[:, self.feet_rigid_body_indices, :2]),
            dim=2,
        )  # (num_envs, num_feet)
        penalty_height_scale = -(
            torch.exp(feet_height_diff / self.feet_drag_sigma) - 1.0
        )
        return (penalty_height_scale * feet_planar_speed).sum(dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        return {"violation": self.check_violation(state=state, control=control)}


class StayCloseToDefaultConfig(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        default_config: torch.Tensor,
        power: float,
        joint_names: Optional[List[str]] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        all_joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        if joint_names is None:
            self.joint_names = all_joint_names
        else:
            self.joint_names = joint_names

        self.joint_indices = torch.tensor(
            [all_joint_names.index(joint_name) for joint_name in self.joint_names]
        ).to(self.device, torch.long)
        assert (
            self.joint_indices != -1
        ).all(), f"joint names can't be found: {joint_names}"
        self.default_config = default_config
        self.power = power

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError(
            "StayCloseToDefaultConfig violation check not supported"
        )

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.sum(
            (state.dof_pos[:, self.joint_indices] - self.default_config[None, :]).abs()
            ** self.power,
            dim=1,
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        stats = {}
        for ref_idx, (joint_idx, joint_name) in enumerate(
            zip(self.joint_indices, self.joint_names)
        ):
            stats[f"usage/{joint_name}"] = (
                state.dof_pos[:, joint_idx] - self.default_config[..., ref_idx]
            ).abs()
        return stats


class RootHeight(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        target_height: float,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.target_height = target_height

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError("RootHeight violation check not supported")

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        return torch.square(state.root_pos[:, 2] - self.target_height)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        return {"root_height": state.root_pos[:, 2]}


class PlanarPose(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        pos_err_scale: float,
        theta_err_scale: float,
        forward_vec: Tuple[float, float, float],
        target_x: Optional[float] = None,
        target_y: Optional[float] = None,
        target_theta: Optional[float] = None,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.target_x = torch.tensor(target_x).to(self.device) if target_x else None
        self.target_y = torch.tensor(target_y).to(self.device) if target_y else None
        self.target_theta = (
            torch.tensor(target_theta).to(self.device) if target_theta else None
        )
        self.env_2d_origins = torch.stack(
            [
                torch.tensor([env_origin.x, env_origin.y])
                for env_origin in map(
                    lambda x: self.gym.get_env_origin(self.gym.get_env(self.sim, x)),
                    range(self.num_envs),
                )
            ],
            dim=0,
        ).to(
            self.device
        )  # (num_envs, 2)
        self.pos_err_scale = pos_err_scale
        self.theta_err_scale = theta_err_scale
        self.forward_vec = torch.tensor(forward_vec).to(self.device)[None, :]

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError("PlanarPose violation check not supported")

    def get_theta_error(self, state: EnvState):
        theta_error = torch.zeros_like(state.root_pos[:, 0])
        if self.target_theta is not None:
            root_forward = quat_rotate_inverse(
                state.root_xyzw_quat,
                self.forward_vec,
            )
            root_theta = torch.atan2(root_forward[:, 1], root_forward[:, 0])
            if self.target_theta == 0:
                theta_error = torch.square(root_theta)
            else:
                theta_error = torch.square(root_theta - self.target_theta)
        return theta_error

    def get_pos_error(self, state: EnvState):
        pos_error = torch.zeros_like(state.root_pos[:, 0])
        if self.target_x is not None and self.target_y is not None:
            root_pos_env_frame = state.root_pos[:, :2] - self.env_2d_origins
            pos_error = torch.square(
                root_pos_env_frame - torch.stack([self.target_x, self.target_y], dim=0)
            ).sum(dim=1)
        elif self.target_x is not None:
            pos_error = torch.square(state.root_pos[:, 0] - self.target_x)
        elif self.target_y is not None:
            pos_error = torch.square(state.root_pos[:, 1] - self.target_y)
        return pos_error

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        pos_error = self.get_pos_error(state=state)

        theta_error = self.get_theta_error(state=state)

        return pos_error * self.pos_err_scale + theta_error * self.theta_err_scale

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        return {
            "pos_error": self.get_pos_error(state=state),
            "theta_error": self.get_theta_error(state=state),
        }


class EnergyUsage(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        power: int,
        # used to compute power usage
        torque_constant: Optional[Union[float, torch.Tensor]] = None,
        voltage: Optional[Union[float, torch.Tensor]] = None,
        skip_stats: bool = True,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
            skip_stats=skip_stats,
        )
        self.joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.power = power
        self.torque_constant = torque_constant
        self.voltage = voltage

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError("EnergyUsage violation check not supported")

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        energy = control.torque * state.dof_vel
        return torch.sum(energy**self.power, dim=1)

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        # mechanical power
        joint_energy = control.torque * state.dof_vel
        stats = {
            "sum_joint_energy": joint_energy.sum(dim=1),
        }
        for joint_idx, joint_name in enumerate(self.joint_names):
            stats[f"usage/{joint_name}"] = joint_energy[:, joint_idx]
        # eletrical power
        if self.torque_constant is not None and self.voltage is not None:
            electrical_power = (
                control.torque.abs() * self.torque_constant * self.voltage
            )
            stats["sum_electrical_power"] = electrical_power.sum(dim=1)
            for joint_idx, joint_name in enumerate(self.joint_names):
                stats[f"usage_electrical/{joint_name}"] = electrical_power[:, joint_idx]
        return stats


class EvenLegUsage(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        leg_joint_sets: Dict[str, List[str]],
        power: int,
        smoothing_dt_multiplier: float,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.joint_names = self.gym.get_actor_dof_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.leg_joint_sets = leg_joint_sets
        self.leg_joint_indices = {
            k: torch.tensor(
                [self.joint_names.index(joint_name) for joint_name in joint_set]
            ).to(self.device, torch.long)
            for k, joint_set in leg_joint_sets.items()
        }
        self.smoothing_dt_multiplier = smoothing_dt_multiplier
        self.past_leg_set_energy_usage = torch.zeros(
            self.num_envs,
            len(self.leg_joint_sets),
            dtype=torch.float32,
            device=self.device,
        )
        self.power = power

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError("EnergyUsage violation check not supported")

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        leg_set_energy_usage = torch.zeros(
            self.num_envs,
            len(self.leg_joint_sets),
            dtype=torch.float32,
            device=self.device,
        )
        # compute per leg joint energy usage
        for leg_set_idx, joint_indices in enumerate(self.leg_joint_indices.values()):
            leg_set_energy_usage[:, leg_set_idx] = torch.square(
                control.torque[:, joint_indices] * state.dof_vel[:, joint_indices]
            ).mean(dim=1)
        if self.smoothing_dt_multiplier > 0.0:
            smoothing = state.sim_dt * self.smoothing_dt_multiplier
            self.past_leg_set_energy_usage = (
                1 - smoothing
            ) * self.past_leg_set_energy_usage + smoothing * leg_set_energy_usage
        else:
            self.past_leg_set_energy_usage = leg_set_energy_usage
        # compute average difference in usage between each leg set
        diff_energy_usage = (
            self.past_leg_set_energy_usage[:, :, None]
            - self.past_leg_set_energy_usage[:, None, :]
        ).abs() ** self.power
        # this is symmetric matrix, so we only need to compute half of it
        return diff_energy_usage.triu(diagonal=1).sum(dim=(1, 2))

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        return {}


class LinkPosePair(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        target_distance: float,
        target_angle: float,
        position_weight: float,
        orientation_weight: float,
        link_1: str,
        link_2: str,
        distance_sigma: float,
        angle_sigma: float,
        terminate_on_violation: bool,
        planar: bool,
        mode: str = "exact",  # or 'max',
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.target_distance = target_distance
        self.target_angles = target_angle
        self.position_weight = position_weight
        self.orientation_weight = orientation_weight
        self.link_1 = link_1
        self.link_2 = link_2
        self.planar = planar
        all_link_names = self.gym.get_actor_rigid_body_names(
            self.gym.get_env(self.sim, 0),
            0,
        )
        self.link_1_index = all_link_names.index(link_1)
        assert self.link_1_index != -1, f"link {link_1} not found"
        self.link_2_index = all_link_names.index(link_2)
        assert self.link_2_index != -1, f"link {link_2} not found"
        self.distance_sigma = distance_sigma
        self.angle_sigma = angle_sigma
        self.mode = mode

    def get_distance(self, state: EnvState) -> torch.Tensor:
        link_1_pos = state.rigid_body_pos[:, self.link_1_index, :]
        link_2_pos = state.rigid_body_pos[:, self.link_2_index, :]
        if self.planar:
            return torch.norm(link_1_pos[:, :2] - link_2_pos[:, :2], dim=1)
        return torch.norm(link_1_pos - link_2_pos, dim=1)

    def get_angle(self, state: EnvState) -> torch.Tensor:
        link_1_mat = pt3d.quaternion_to_matrix(
            state.rigid_body_xyzw_quat[:, self.link_1_index, :][:, [3, 0, 1, 2]]
        )
        link_2_mat = pt3d.quaternion_to_matrix(
            state.rigid_body_xyzw_quat[:, self.link_2_index, :][:, [3, 0, 1, 2]]
        )
        # compute relative transform
        mat = link_1_mat @ link_2_mat.transpose(1, 2)

        if self.planar:
            euler_angles = pt3d.matrix_to_euler_angles(mat, "XYZ")
            return euler_angles[:, 2].abs()

        trace = torch.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)
        # to prevent numerical instability, clip the trace to [-1, 3]
        trace = torch.clamp(trace, min=-1 + 1e-8, max=3 - 1e-8)
        rotation_magnitude = torch.arccos((trace - 1) / 2)
        # account for symmetry
        rotation_magnitude = rotation_magnitude % (2 * torch.pi)
        rotation_magnitude = torch.min(
            rotation_magnitude,
            2 * torch.pi - rotation_magnitude,
        )
        return rotation_magnitude

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        termination = torch.zeros(
            state.dof_pos.shape[0], device=self.device, dtype=torch.bool
        )
        if self.position_weight > 0.0:
            if self.mode == "max":
                termination |= self.get_distance(state) > self.target_distance
            elif self.mode == "exact":
                termination |= (
                    self.get_distance(state) - self.target_distance
                ).abs() / self.distance_sigma > 1.0
        if self.orientation_weight > 0.0:
            if self.mode == "max":
                termination |= self.get_angle(state) > self.target_angles
            elif self.mode == "exact":
                termination |= (
                    self.get_angle(state) - self.target_angles
                ).abs() / self.angle_sigma > 1.0
        return termination

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        penalty = torch.zeros(state.dof_pos.shape[0], device=self.device)
        if self.position_weight > 0.0:
            dist = self.get_distance(state)
            if self.mode == "exact":
                dist_err = (dist - self.target_distance).abs()

            elif self.mode == "max":
                dist_err = torch.clip(dist - self.target_distance, min=0.0)
            else:
                raise NotImplementedError(f"mode {self.mode} not supported")
            dist_penalty = 1.0 - torch.exp(-dist_err / self.distance_sigma)
            penalty += dist_penalty * self.position_weight

        if self.orientation_weight > 0.0:
            angle = self.get_angle(state)
            if self.mode == "exact":
                angle_err = (angle - self.target_angles).abs()
            elif self.mode == "max":
                angle_err = torch.clip(angle - self.target_angles, min=0.0)
            else:
                raise NotImplementedError(f"mode {self.mode} not supported")
            angle_penalty = 1.0 - torch.exp(-angle_err / self.angle_sigma)
            penalty += angle_penalty * self.orientation_weight
        return penalty

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        return {
            "distance": self.get_distance(state),
            "angle": self.get_angle(state),
        }


class FootGroundContact(Constraint):
    # Force sensors will sense +z force when touching the ground
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        penalty_foot_force: float,
        violation_foot_force: float,
        power: float,
        feet_sensor_indices: List[int],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.penalty_foot_force = penalty_foot_force
        self.violation_foot_force = violation_foot_force
        self.feet_sensor_indices = torch.tensor(
            feet_sensor_indices,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.power = power

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        assert state.force_sensor_tensor is not None
        ground_foot_force = state.force_sensor_tensor[
            :, self.feet_sensor_indices, 2
        ].clip(min=0.0)
        return (ground_foot_force > self.violation_foot_force).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        assert state.force_sensor_tensor is not None
        ground_foot_force = state.force_sensor_tensor[
            :, self.feet_sensor_indices, 2
        ].clip(min=0.0)
        violation = (ground_foot_force - self.penalty_foot_force).clip(
            min=0.0
        ) ** self.power
        return torch.sum(
            violation,
            dim=1,
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        assert state.force_sensor_tensor is not None
        return {
            "foot_force": state.force_sensor_tensor[:, self.feet_sensor_indices].norm(
                dim=-1
            )
        }


class EvenMassDistribution(Constraint):
    # Uses force sensors in feet to make sure the mass is evenly distributed
    # between the feet
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        feet_sensor_indices: List[int],
        violation_threshold: float,
        power: float,
        flying_penalty: float,
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.feet_sensor_indices = torch.tensor(
            feet_sensor_indices,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.violation_threshold = violation_threshold
        self.power = power
        self.flying_penalty = flying_penalty

    def get_normalized_mass_distribution(self, state: EnvState):
        assert state.force_sensor_tensor is not None
        foot_force = state.force_sensor_tensor[:, self.feet_sensor_indices, 2].clip(
            min=0.0
        )
        return foot_force / (foot_force.sum(dim=1, keepdim=True) + 1e-8)

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        mass_distribution = self.get_normalized_mass_distribution(state=state)
        return (mass_distribution.std(dim=1) > self.violation_threshold).any(dim=1)

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        mass_distribution = self.get_normalized_mass_distribution(state=state)
        penalty = torch.zeros(state.dof_pos.shape[0], device=self.device)
        valid_mask = mass_distribution.sum(dim=1) > 1e-8
        penalty[valid_mask] = mass_distribution[valid_mask].std(dim=1) ** self.power
        penalty[~valid_mask] = (
            self.flying_penalty
        )  # large penalty for when all four legs are up in the air
        return penalty

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        mass_distribution = self.get_normalized_mass_distribution(state=state)
        stats = {}
        for i in range(mass_distribution.shape[1]):
            stats[f"mass_distribution/foot_{i}"] = mass_distribution[:, i]
        stats["std"] = mass_distribution.std(dim=1)
        return stats


class PointBodyAtGripper(Constraint):
    def __init__(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        generator: torch.Generator,
        violation_weight: float,
        penalty_weight: float,
        terminate_on_violation: bool,
        # constraint specific parameters
        gripper_link_name: str,
        power: float,
        dimensions: List[int],
    ):
        super().__init__(
            gym=gym,
            sim=sim,
            device=device,
            generator=generator,
            violation_weight=violation_weight,
            penalty_weight=penalty_weight,
            terminate_on_violation=terminate_on_violation,
        )
        self.gripper_link_name = gripper_link_name
        self.power = power
        self.gripper_link_index = self.gym.get_actor_rigid_body_names(
            self.gym.get_env(self.sim, 0),
            0,
        ).index(gripper_link_name)
        self.dimensions = dimensions
        # if [1,2], then regularizes both height and planar direction
        # if [1] only, then regularizes only planar direction

        assert (
            self.gripper_link_index != -1
        ), f"gripper link {gripper_link_name} not found"

    def get_gripper_in_body_frame(self, state: EnvState):
        gripper_pos = state.rigid_body_pos[:, self.gripper_link_index, :]
        body_pose = state.root_pose
        return (
            torch.linalg.inv(body_pose)
            @ torch.cat([gripper_pos, torch.ones_like(gripper_pos[..., [0]])], dim=-1)[
                :, :, None
            ]
        )[:, :3, 0]

    def check_violation(self, state: EnvState, control: Control) -> torch.Tensor:
        raise NotImplementedError("PointBodyAtGripper violation check not supported")

    def compute_penalty(self, state: EnvState, control: Control) -> torch.Tensor:
        gripper_pos_body_frame = self.get_gripper_in_body_frame(state=state)
        # if the body is pointing at the gripper, then the z and y component
        # of the gripper position in the body frame should be small
        gripper_body_direction = gripper_pos_body_frame / torch.linalg.norm(
            gripper_pos_body_frame, dim=-1, keepdim=True
        )
        return torch.sum(
            gripper_body_direction[:, self.dimensions].abs() ** self.power, dim=1
        )

    def step(self, state: EnvState, control: Control) -> Dict[str, torch.Tensor]:
        if self.skip_stats:
            return {}
        gripper_pos_body_frame = self.get_gripper_in_body_frame(state=state)
        deviation = torch.linalg.norm(
            gripper_pos_body_frame[:, self.dimensions], dim=-1
        )
        return {"gripper_deviation": deviation}

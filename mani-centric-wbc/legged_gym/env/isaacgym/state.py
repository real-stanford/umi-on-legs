from __future__ import annotations

import dataclasses
from typing import Optional

import pytorch3d.transforms as pt3d
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

"""
The returned GPU tensor has a gpu device side pointer to the data resident on the GPU as well as information about the data type and tensor shape. Sharing this data with a deep learning framework requires a tensor adapter, like the one provided in the gymtorch module for PyTorch interop:

camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
add_triangle_mesh
http://localhost:1234/programming/tensors.html
"""


@dataclasses.dataclass
class EnvState:
    root_state: torch.Tensor
    root_pos: torch.Tensor
    root_xyzw_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor

    dof_pos: torch.Tensor
    dof_vel: torch.Tensor

    prev_dof_pos: torch.Tensor
    prev_dof_vel: torch.Tensor

    rigid_body_pos: torch.Tensor
    rigid_body_xyzw_quat: torch.Tensor
    rigid_body_lin_vel: torch.Tensor
    rigid_body_ang_vel: torch.Tensor

    measured_terrain_heights: torch.Tensor  # around the robot

    contact_forces: torch.Tensor
    force_sensor_tensor: Optional[torch.Tensor]

    episode_time: torch.Tensor  # in seconds
    gravity: torch.Tensor

    time: float
    prev_time: float

    device: str

    @property
    def sim_dt(self) -> float:
        return self.time - self.prev_time

    @property
    def local_root_ang_vel(self) -> torch.Tensor:
        """
        Project gravity vector onto the root's local frame
        """
        return quat_rotate_inverse(
            self.root_xyzw_quat,
            self.root_ang_vel,
        )

    @property
    def local_root_lin_vel(self) -> torch.Tensor:
        return quat_rotate_inverse(
            self.root_xyzw_quat,
            self.root_lin_vel,
        )

    @property
    def local_root_gravity(self) -> torch.Tensor:
        return quat_rotate_inverse(
            self.root_xyzw_quat,
            self.gravity / torch.linalg.norm(self.gravity, dim=1, keepdims=True),
        )

    def get_local_link_gravity(self, link_idx: int) -> torch.Tensor:
        return quat_rotate_inverse(
            self.rigid_body_xyzw_quat[:, link_idx],
            self.gravity / torch.linalg.norm(self.gravity, dim=1, keepdims=True),
        )

    @property
    def root_pose(self):
        # returns the current root pose in the local frame of the robot
        root_pose = (
            torch.eye(4, device=self.device)
            .unsqueeze(0)
            .repeat(len(self.root_pos), 1, 1)
        )
        root_pose[..., :3, 3] = self.root_pos[:]
        # pt3d quaternion convention is wxyz
        root_pose[..., :3, :3] = pt3d.quaternion_to_matrix(
            self.root_xyzw_quat[:, [3, 0, 1, 2]]
        )
        return root_pose

    def step(
        self,
        gym: gymapi.Gym,
        sim: gymapi.Sim,
    ):
        self.prev_dof_pos[:] = self.dof_pos[:]
        self.prev_dof_vel[:] = self.dof_vel[:]
        self.prev_time = self.time

        gym.simulate(sim)
        if self.device == "cpu":
            gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_force_sensor_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        self.time = gym.get_sim_time(sim)
        self.episode_time += self.sim_dt

    def isnan(self) -> bool:
        return bool(
            (
                torch.isnan(self.root_state).any()
                or torch.isnan(self.root_pos).any()
                or torch.isnan(self.root_xyzw_quat).any()
                or torch.isnan(self.root_lin_vel).any()
                or torch.isnan(self.root_ang_vel).any()
                or torch.isnan(self.dof_pos).any()
                or torch.isnan(self.dof_vel).any()
                or torch.isnan(self.prev_dof_pos).any()
                or torch.isnan(self.prev_dof_vel).any()
                or torch.isnan(self.rigid_body_pos).any()
                or torch.isnan(self.rigid_body_xyzw_quat).any()
                or torch.isnan(self.rigid_body_lin_vel).any()
                or torch.isnan(self.rigid_body_ang_vel).any()
                or torch.isnan(self.contact_forces).any()
                or (
                    torch.isnan(self.force_sensor_tensor).any()
                    if self.force_sensor_tensor is not None
                    else torch.tensor([False], device=self.device)
                )
            ).item()
        )

    @staticmethod
    def initialize(
        gym: gymapi.Gym,
        sim: gymapi.Sim,
        device: str,
        terrain_heights: Optional[torch.Tensor],
    ) -> EnvState:
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_force_sensor_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        env_count = gym.get_env_count(sim)
        num_actors_per_env = gym.get_actor_count(gym.get_env(sim, 0))
        num_rigid_bodies = gym.get_env_rigid_body_count(gym.get_env(sim, 0))
        assert num_actors_per_env == 1, "Only one actor per env is supported. "
        num_dof = gym.get_env_dof_count(gym.get_env(sim, 0))

        dof_states = (
            gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
            .view(env_count, num_dof, 2)
            .to(device)
        )
        root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)).to(
            device
        )
        assert root_states.shape == (env_count * num_actors_per_env, 13)
        rigid_body_state_tensor = (
            gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
            .to(device)
            .view(env_count, num_rigid_bodies, 13)
        )
        """
        The buffer has shape (num_rigid_bodies, 13). State for each rigid body 
        contains position([0:3]), rotation([3:7]), linear velocity([7:10]), 
        and angular velocity([10:13]).
        """
        assert rigid_body_state_tensor.shape == (
            env_count,
            num_rigid_bodies,
            13,
        ), f"rigid_body_state_tensor.shape = {rigid_body_state_tensor.shape}"
        force_sensor_tensor = None
        if gym.get_sim_force_sensor_count(sim) > 0:
            gym.refresh_force_sensor_tensor(sim)
            force_sensor_tensor = gymtorch.wrap_tensor(
                gym.acquire_force_sensor_tensor(sim)
            )
            if force_sensor_tensor is not None:
                force_sensor_tensor = force_sensor_tensor.view(env_count, -1, 6).to(
                    device
                )  # num envs, num sensors, force and torque

        sim_params = gym.get_sim_params(sim)
        sim_gravity = sim_params.gravity

        contact_forces = (
            gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
            .view(env_count, -1, 3)
            .to(device)
        )
        assert contact_forces.shape == (
            env_count,
            num_rigid_bodies,
            3,
        ), f"contact_forces.shape = {contact_forces.shape}"

        return EnvState(
            # roots
            root_state=root_states,
            root_pos=root_states[..., 0:3],
            root_xyzw_quat=root_states[..., 3:7],
            root_lin_vel=root_states[..., 7:10],
            root_ang_vel=root_states[..., 10:13],
            # dof
            dof_pos=dof_states[..., 0],
            dof_vel=dof_states[..., 1],
            prev_dof_pos=dof_states[..., 0].clone(),
            prev_dof_vel=dof_states[..., 1].clone(),
            # rigid bodies
            rigid_body_pos=rigid_body_state_tensor[..., 0:3],
            rigid_body_xyzw_quat=rigid_body_state_tensor[..., 3:7],
            rigid_body_lin_vel=rigid_body_state_tensor[..., 7:10],
            rigid_body_ang_vel=rigid_body_state_tensor[..., 10:13],
            # others
            contact_forces=contact_forces,
            force_sensor_tensor=force_sensor_tensor,
            time=gym.get_sim_time(sim),
            prev_time=gym.get_sim_time(sim) - sim_params.dt,
            gravity=torch.tensor(
                [sim_gravity.x, sim_gravity.y, sim_gravity.z], device=device
            )
            .unsqueeze(dim=0)
            .repeat(env_count, 1),
            episode_time=torch.zeros(env_count, dtype=torch.float32, device=device),
            device=device,
            measured_terrain_heights=(
                terrain_heights
                if terrain_heights is not None
                else torch.zeros(1, dtype=torch.float32, device=device)
            ),
        )


@dataclasses.dataclass
class EnvSetup:
    kp: torch.Tensor
    kd: torch.Tensor

    rigidbody_mass: torch.Tensor
    rigidbody_com_offset: torch.Tensor
    rigidbody_restitution_coef: torch.Tensor
    rigid_shape_friction: torch.Tensor

    dof_friction: torch.Tensor
    dof_damping: torch.Tensor
    dof_velocity: torch.Tensor

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch3d.transforms as p3d
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_mul, to_torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.env.isaacgym.constraints import Constraint
from legged_gym.env.isaacgym.control import Control, PDController
from legged_gym.env.isaacgym.obs import EnvObservationAttribute, EnvSetupAttribute
from legged_gym.env.isaacgym.state import EnvSetup, EnvState
from legged_gym.env.isaacgym.task import Task
from legged_gym.env.isaacgym.terrain import TerrainPerlin
from legged_gym.env.isaacgym.utils import quat_apply_yaw, torch_rand_float
from legged_gym.env.obs import ObservationAttribute
from legged_gym.rsl_rl.env import VecEnv

PartialTask = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Task]
PartialConstraint = Callable[[gymapi.Gym, gymapi.Sim, str, torch.Generator], Constraint]


class IsaacGymEnv(VecEnv):
    def __init__(
        self,
        cfg,
        sim_params,
        sim_device,
        headless,
        controller: PDController,
        state_obs: Dict[str, EnvObservationAttribute],
        setup_obs: Dict[str, EnvSetupAttribute],
        privileged_state_obs: Dict[str, EnvObservationAttribute],
        privileged_setup_obs: Dict[str, EnvSetupAttribute],
        tasks: Dict[str, PartialTask],
        constraints: Dict[str, PartialConstraint],
        seed: int,
        dof_pos_reset_range_scale: float,
        obs_history_len: int,
        vis_resolution: Tuple[int, int],
        env_spacing: float,
        ctrl_buf_len: int,
        max_action_value: float,
        ctrl_delay: Optional[torch.Tensor] = None,
        init_dof_pos: Optional[torch.Tensor] = None,
        graphics_device_id: Optional[int] = None,
        debug_viz: float = True,
        attach_camera: bool = True,
        dense_rewards: bool = True,
    ):
        self.dof_pos_reset_range_scale = dof_pos_reset_range_scale

        self.cfg = cfg
        self.sim_params = sim_params
        self.debug_viz = debug_viz
        self.controller = controller
        self.controller.kp = self.controller.kp.repeat(cfg.env.num_envs, 1)
        self.controller.kd = self.controller.kd.repeat(cfg.env.num_envs, 1)
        self.init_kp = self.controller.kp.clone()
        self.init_kd = self.controller.kd.clone()
        self.gym_dt = (
            np.mean(self.controller.decimation_count_range) * self.sim_params.dt
        )
        self.reward_scales = self.cfg.rewards.scales
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = int(self.max_episode_length_s / self.gym_dt)
        self.env_spacing = env_spacing
        self.dense_rewards = dense_rewards
        self.reward_dt_scale = self.sim_params.dt
        if not self.dense_rewards:
            self.reward_dt_scale *= np.mean(self.controller.decimation_count_range)

        self.cfg.domain_rand.push_interval = np.ceil(
            self.cfg.domain_rand.push_interval_s / self.gym_dt
        )
        self.cfg.domain_rand.transport_interval = np.ceil(
            self.cfg.domain_rand.transport_interval_s / self.gym_dt
        )

        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.sim_device = sim_device
        sim_device_type = "cuda" if "cuda" in self.sim_device else "cpu"
        if sim_device_type == "cuda":
            self.sim_device_id = int(self.sim_device.split(":")[1])
        else:
            self.sim_device_id = -1
        self.headless = headless

        if sim_device_type == "cuda":
            self.device: str = self.sim_device
            self.sim_params.use_gpu_pipeline = True
            self.sim_params.physx.use_gpu = True
        else:
            self.device: str = "cpu"
            self.sim_params.use_gpu_pipeline = False
            self.sim_params.physx.use_gpu = False
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        # graphics device for rendering, -1 for no rendering
        if not attach_camera and headless:
            self.graphics_device_id = -1
        else:
            if graphics_device_id is None:
                self.graphics_device_id = self.sim_device_id
            else:
                self.graphics_device_id = graphics_device_id

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        self.init_dof_pos = (
            init_dof_pos if init_dof_pos is not None else self.controller.offset
        )
        self.init_dof_pos = self.init_dof_pos[None, :].to(self.device)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.state = EnvState.initialize(
            gym=self.gym,
            sim=self.sim,
            device=self.device,
            terrain_heights=(
                None
                if self.cfg.terrain.mode in {"none", "plane"}
                else torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.float
                )
            ),
        )
        # print body indices
        rb_names = self.gym.get_actor_rigid_body_names(
            self.envs[0], self.actor_handles[0]
        )
        for i, rb_name in enumerate(rb_names):
            logging.info(f"[{i:02d}] {rb_name}")

        dof_names = self.gym.get_actor_dof_names(self.envs[0], self.actor_handles[0])
        for i, dof_name in enumerate(dof_names):
            logging.info(f"[{i:02d}] {dof_name}")

        """Initialize torch tensors which will contain simulation states and processed quantities"""
        # get gym GPU state tensors
        self.ctrl = Control(
            buffer=torch.zeros(
                (self.num_envs, ctrl_buf_len, self.num_actions),
                dtype=torch.float,
                device=self.device,
            ),
            torque=torch.zeros(
                (self.num_envs, self.num_actions),
                dtype=torch.float,
                device=self.device,
            ),
        )
        self.max_action_value = max_action_value
        if ctrl_delay is not None:
            assert torch.allclose(
                torch.round(ctrl_delay / self.sim_params.dt),
                ctrl_delay / self.sim_params.dt,
            ), "ctrl_delay must be a multiple of the simulation dt"
            assert (ctrl_delay >= 0).all(), "ctrl_delay can't be negative"
            self.ctrl_delay_steps = torch.round(ctrl_delay / self.sim_params.dt)
        else:
            self.ctrl_delay_steps = torch.zeros(self.num_actions, device=self.device)

        # initialize some data used later on
        self.global_step = 0
        self.extras = {}
        self.state_obs = {
            k: v
            for k, v in sorted(state_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.setup_obs = {
            k: v
            for k, v in sorted(setup_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.privileged_state_obs = {
            k: v
            for k, v in sorted(privileged_state_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.privileged_setup_obs = {
            k: v
            for k, v in sorted(privileged_setup_obs.items(), key=lambda x: x[0])
            if isinstance(v, ObservationAttribute)
        }
        self.tasks = {
            k: v(self.gym, self.sim, self.device, self.generator)
            for k, v in tasks.items()
            if type(v) is functools.partial
        }
        self.constraints = {
            k: v(self.gym, self.sim, self.device, self.generator)
            for k, v in constraints.items()
            if type(v) is functools.partial
        }
        self._prepare_reward_function()

        # attach camera to last environment

        self.vis_env = self.envs[0]
        self.vis_cam_handle = None
        if attach_camera:
            cam_props = gymapi.CameraProperties()
            cam_props.horizontal_fov = 70.0
            cam_props.far_plane = 10.0
            cam_props.near_plane = 1e-2
            cam_props.height = vis_resolution[0]
            cam_props.width = vis_resolution[1]
            cam_props.enable_tensors = self.device != "cpu"
            cam_props.use_collision_geometry = False

            self.vis_cam_handle = self.gym.create_camera_sensor(self.vis_env, cam_props)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(1.6, 1.4, 0.8)
            local_transform.r = gymapi.Quat.from_euler_zyx(3.141592653589793, 2.8, 0.8)
            self.gym.attach_camera_to_body(
                self.vis_cam_handle,
                self.vis_env,
                self.actor_handles[0],
                local_transform,
                gymapi.FOLLOW_POSITION,
            )
        assert not self.state.isnan()
        self.obs_history = torch.zeros(
            (self.num_envs, obs_history_len, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.obs_history_len = obs_history_len

    @property
    def episode_step(self) -> torch.Tensor:
        return (self.state.episode_time / self.gym_dt).long()

    def reset(self):
        """Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs, privileged_obs

    def render(self, sync_frame_time=False):
        # fetch results
        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        # step graphics
        self.gym.step_graphics(self.sim)
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # fetch results
            # step graphics
            if self.enable_viewer_sync:
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
        self.gym.render_all_camera_sensors(self.sim)
        if self.vis_cam_handle is None:
            raise RuntimeError("No camera attached")
        env = self.vis_env
        rgb = self.gym.get_camera_image(
            self.sim, env, self.vis_cam_handle, gymapi.IMAGE_COLOR
        )
        rgb = rgb.reshape(rgb.shape[0], -1, 4)
        return rgb[..., :3]

    def step(
        self,
        action: torch.Tensor,
        return_vis: bool = False,
        callback: Optional[Callable[[IsaacGymEnv]]] = None,
    ):
        """
        Apply actions, simulate, call

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        info = {}
        self.ctrl.push(
            torch.clip(action, -self.max_action_value, self.max_action_value).to(
                self.device
            )
        )
        reward = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float,
        )
        # step physics and render each frame
        rendering = self.viewer is not None or return_vis

        if rendering and self.debug_viz:
            self.visualize(vis_env_ids=[0])  # the rendering env

        if return_vis and self.vis_cam_handle is not None:
            vis = self.render(sync_frame_time=False)
            if vis is not None:
                info["vis"] = vis
        decimation_count = self.controller.decimation_count
        for decimation_step in range(decimation_count):
            callback(self) if callback is not None else None
            # handle delay by indexing into the buffer of past targets
            # since new actions are pushed to the front of the buffer,
            # the current target is further back in the buffer for larger
            # delays.
            curr_target_idx = torch.ceil(
                ((self.ctrl_delay_steps - decimation_step)) / decimation_count
            ).long()
            assert (curr_target_idx >= 0).all()
            self.ctrl.torque = self.controller(
                action=self.ctrl.buffer.permute(2, 1, 0)[
                    torch.arange(self.num_actions, device=self.device),
                    curr_target_idx,
                    :,
                ].permute(1, 0),
                state=self.state,
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.ctrl.torque)
            )
            self.state.step(gym=self.gym, sim=self.sim)
            if self.cfg.terrain.mode in {"perlin"}:
                self.state.measured_terrain_heights = self._get_heights()
            if self.dense_rewards or decimation_step == decimation_count - 1:
                for task_name, task in self.tasks.items():
                    for k, v in task.step(state=self.state, control=self.ctrl).items():
                        stat_key = f"task/{task_name}/{k}"
                        if stat_key not in info:
                            info[stat_key] = v
                        else:
                            # compute mean (of decimation steps) in place
                            info[stat_key] = (info[stat_key] * decimation_step + v) / (
                                decimation_step + 1
                            )
                for constraint_name, constraint in self.constraints.items():
                    for k, v in constraint.step(
                        state=self.state, control=self.ctrl
                    ).items():
                        stat_key = f"constraint/{constraint_name}/{k}"
                        if stat_key not in info:
                            info[stat_key] = v
                        else:
                            # compute mean (of decimation steps) in place
                            info[stat_key] = (info[stat_key] * decimation_step + v) / (
                                decimation_step + 1
                            )
                reward_terms = self.compute_reward(state=self.state, control=self.ctrl)
                reward += reward_terms["reward/total"]
                for k, v in reward_terms.items():
                    if k in info:
                        info[k] += v
                    else:
                        info[k] = v
        self.global_step += 1
        if self.cfg.domain_rand.push_robots and (
            self.global_step % self.cfg.domain_rand.push_interval == 0
        ):
            """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
            self.state.root_state[:, 7:13] = torch_rand_float(
                -self.cfg.domain_rand.max_push_vel,
                self.cfg.domain_rand.max_push_vel,
                (self.num_envs, 6),
                device=self.device,
                generator=self.generator,
            )  # lin vel x/y/z, ang vel x/y/z
            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.state.root_state)
            )
        if self.cfg.domain_rand.transport_robots and (
            self.global_step % self.cfg.domain_rand.transport_interval == 0
        ):
            """Randomly transports the robots to a new location"""
            self.state.root_state[:, 0:3] += (
                torch.randn(
                    self.num_envs,
                    3,
                    device=self.device,
                    generator=self.generator,
                )
                * self.cfg.domain_rand.transport_pos_noise_std
            )
            euler_noise = (
                torch.randn(
                    self.num_envs,
                    3,
                    device=self.device,
                    generator=self.generator,
                )
                * self.cfg.domain_rand.transport_euler_noise_std
            )
            quat_wxyz_transport = p3d.matrix_to_quaternion(
                p3d.euler_angles_to_matrix(euler_noise, "XYZ")
            )
            self.state.root_state[:, 3:7] = quat_mul(
                self.state.root_state[:, 3:7],
                quat_wxyz_transport[..., [1, 2, 3, 0]],  # reorder to xyzw
            )

            self.gym.set_actor_root_state_tensor(
                self.sim, gymtorch.unwrap_tensor(self.state.root_state)
            )
        self.check_termination(state=self.state, control=self.ctrl)
        for constraint_name, constraint in self.constraints.items():
            info[f"constraint/{constraint_name}/termination"] = (
                constraint.check_termination(state=self.state, control=self.ctrl)
            )
            self.reset_buf |= info[f"constraint/{constraint_name}/termination"]
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        obs = self.get_observations(
            state=self.state,
            setup=self.setup,
            state_obs=self.state_obs,
            setup_obs=self.setup_obs,
        )
        self.obs_history = torch.cat(
            (self.obs_history[:, 1:, :], obs.unsqueeze(1)), dim=1
        )
        privileged_obs = self.get_observations(
            state=self.state,
            setup=self.setup,
            state_obs=self.privileged_state_obs,
            setup_obs=self.privileged_setup_obs,
        )

        info.update(self.extras)
        return (
            self.obs_history.view(self.num_envs, -1),
            privileged_obs,
            reward,
            self.reset_buf,
            info,
        )

    def check_termination(self, state: EnvState, control: Control):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(
            torch.norm(
                state.contact_forces[:, self.termination_contact_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = (
            self.episode_step > self.max_episode_length
        )  # no terminal reward for time-outs
        # also reset if robot walks off the safe bounds
        walked_off_safe_bounds = torch.logical_or(
            (self.state.root_pos[:, :2] < self.safe_bounds[None, :, 0]).any(dim=1),
            (self.state.root_pos[:, :2] > self.safe_bounds[None, :, 1]).any(dim=1),
        )
        self.time_out_buf |= walked_off_safe_bounds
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        for task in self.tasks.values():
            task.reset_idx(env_ids)
        for constraint in self.constraints.values():
            constraint.reset_idx(env_ids)
        self.obs_history[env_ids] = 0.0

        # reset controllers
        if self.cfg.domain_rand.randomize_pd_params:
            self.controller.kp[env_ids] = (
                torch_rand_float(
                    lower=self.cfg.domain_rand.kp_ratio_range[0],
                    upper=self.cfg.domain_rand.kp_ratio_range[1],
                    shape=(len(env_ids), self.controller.control_dim),
                    device=self.device,
                    generator=self.generator,
                )
                * self.init_kp[env_ids]
            )
            self.controller.kd[env_ids] = (
                torch_rand_float(
                    lower=self.cfg.domain_rand.kd_ratio_range[0],
                    upper=self.cfg.domain_rand.kd_ratio_range[1],
                    shape=(len(env_ids), self.controller.control_dim),
                    device=self.device,
                    generator=self.generator,
                )
                * self.init_kd[env_ids]
            )
            self.setup.kp[env_ids] = self.controller.kp[env_ids]
            self.setup.kd[env_ids] = self.controller.kd[env_ids]

        # reset buffers
        self.ctrl.buffer[env_ids] = 0.0
        self.state.episode_time[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self, state: EnvState, control: Control):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        return_dict = {
            "total": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "env": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "constraint": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
            "task": torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float,
            ),
        }
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            return_dict[name] = (
                self.reward_functions[i](state=state, control=control)
                * self.reward_scales[name]
            )
            return_dict["total"] += return_dict[name]
            return_dict["env"] += return_dict[name]
        for constraint_name, constraint in self.constraints.items():
            constraint_rewards = {
                f"constraint/{constraint_name}/{k}": v
                for k, v in constraint.reward(state=state, control=control).items()
            }
            return_dict.update(constraint_rewards)
            return_dict["total"] += sum(constraint_rewards.values())
            return_dict["constraint"] += sum(constraint_rewards.values())
        for task_name, task in self.tasks.items():
            task_rewards = {
                f"task/{task_name}/{k}": v
                for k, v in task.reward(state=state, control=control).items()
            }
            return_dict.update(task_rewards)
            return_dict["total"] += sum(task_rewards.values())
            return_dict["task"] += sum(task_rewards.values())
        if self.cfg.rewards.only_positive_rewards:
            return_dict["total"][:] = torch.clip(return_dict["total"][:], min=0.0)
        return_dict["task_to_env_ratio"] = return_dict["task"].abs() / (
            return_dict["env"].abs() + 1e-10
        )
        return_dict["task_to_constraint_ratio"] = return_dict["task"].abs() / (
            return_dict["constraint"].abs() + 1e-10
        )
        return {f"reward/{k}": v * self.reward_dt_scale for k, v in return_dict.items()}

    def get_observations(
        self,
        state: EnvState,
        setup: EnvSetup,
        state_obs: Dict[str, EnvObservationAttribute],
        setup_obs: Dict[str, EnvSetupAttribute],
    ):
        obs_attrs = []
        for obs_attr in state_obs.values():
            value = obs_attr(struct=state, generator=self.generator)
            assert value.shape[-1] == obs_attr.dim
            obs_attrs.append(value)
        state_obs_tensor = torch.cat(
            obs_attrs,
            dim=1,
        )
        if len(self.tasks) > 0:
            all_task_obs = []
            for k, task in self.tasks.items():
                task_obs = task.observe(state=state)
                all_task_obs.append(task_obs)
            task_obs_tensor = torch.cat(
                all_task_obs,
                dim=1,
            )
        else:
            task_obs_tensor = torch.zeros(
                (self.num_envs, 0), dtype=torch.float, device=self.device
            )
        if len(setup_obs) > 0:
            obs_attrs = []
            for k, obs_attr in setup_obs.items():
                value = obs_attr(struct=setup, generator=self.generator).reshape(
                    self.num_envs, -1
                )
                assert value.shape[-1] == obs_attr.dim
                obs_attrs.append(value)
            setup_obs_tensor = torch.cat(obs_attrs, dim=1)
        else:
            setup_obs_tensor = torch.zeros(
                (self.num_envs, 0), dtype=torch.float, device=self.device
            )
        return torch.cat(
            (
                setup_obs_tensor,
                state_obs_tensor,
                task_obs_tensor,
                self.ctrl.action,
            ),
            dim=1,
        )

    def create_sim(self):
        """Creates simulation and evironments"""
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            gymapi.SIM_PHYSX,
            self.sim_params,
        )
        self._create_envs()
        self.safe_bounds = torch.tensor([[-10e8, 10e8]] * 2).to(self.device)
        if self.cfg.terrain.mode == "none":
            return
        elif self.cfg.terrain.mode == "plane":
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.cfg.terrain.static_friction
            plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            plane_params.restitution = self.cfg.terrain.restitution
            self.gym.add_ground(self.sim, plane_params)
        elif self.cfg.terrain.mode == "perlin":
            self.terrain = TerrainPerlin(
                tot_cols=self.cfg.terrain.tot_cols,
                tot_rows=self.cfg.terrain.tot_rows,
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                zScale=self.cfg.terrain.zScale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                slope_threshold=self.cfg.terrain.slope_threshold,
            )
            tm_params = gymapi.TriangleMeshParams()
            tm_params.nb_vertices = self.terrain.vertices.shape[0]
            tm_params.nb_triangles = self.terrain.triangles.shape[0]

            tm_params.transform.p.x = self.cfg.terrain.transform_x
            tm_params.transform.p.y = self.cfg.terrain.transform_y
            tm_params.transform.p.z = self.cfg.terrain.transform_z
            tm_params.static_friction = self.cfg.terrain.static_friction
            tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            tm_params.restitution = self.cfg.terrain.restitution
            self.gym.add_triangle_mesh(
                self.sim,
                self.terrain.vertices.flatten(order="C"),
                self.terrain.triangles.flatten(order="C"),
                tm_params,
            )
            self.height_points = self._init_height_points()
            self.height_samples = (
                torch.tensor(self.terrain.heightsamples)
                .view(self.terrain.tot_rows, self.terrain.tot_cols)
                .to(self.device)
            )
            bounds = np.array(
                (
                    self.terrain.vertices.min(axis=0),
                    self.terrain.vertices.max(axis=0),
                )
            )
            bounds[:, 0] += self.cfg.terrain.transform_x
            bounds[:, 1] += self.cfg.terrain.transform_y
            bounds[:, 2] += self.cfg.terrain.transform_z
            terrain_dims = bounds[1, :2] - bounds[0, :2]
            logging.info(
                f"Terrain dimensions: {terrain_dims[0]:.1f}m x {terrain_dims[1]:.1f}m"
            )
            assert (
                terrain_dims > self.cfg.terrain.safety_margin * 2
            ).all(), "Terrain too small for safety margin"
            self.env_origins = -self.env_origins
            self.env_origins[:, 0] += torch_rand_float(
                bounds[0, 0] + self.cfg.terrain.safety_margin,
                bounds[1, 0] - self.cfg.terrain.safety_margin,
                (self.num_envs, 1),
                device=self.device,
                generator=self.generator,
            )[:, 0]
            self.env_origins[:, 1] += torch_rand_float(
                bounds[0, 1] + self.cfg.terrain.safety_margin,
                bounds[1, 1] - self.cfg.terrain.safety_margin,
                (self.num_envs, 1),
                device=self.device,
                generator=self.generator,
            )[:, 0]
            self.env_origins[:, 2] += float(bounds[1, 2])
        else:
            raise ValueError(f"Unknown terrain mode {self.cfg.terrain.mode!r}")

    def set_camera(self, position, lookat):
        """Set camera position and direction"""
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            for i in range(len(props)):
                props[i].friction = self.setup.rigid_shape_friction[env_id, i]
        if len(self.cfg.domain_rand.randomize_restitution_rigid_bodies) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_restitution_rigid_body_ids
            ):
                props[body_id].restitution = torch_rand_float(
                    lower=self.cfg.domain_rand.restitution_coef_range[0],
                    upper=self.cfg.domain_rand.restitution_coef_range[1],
                    shape=(1,),
                    device=self.device,
                    generator=self.generator,
                ).item()
                self.setup.rigidbody_restitution_coef[env_id, idx] = props[
                    body_id
                ].restitution
        return props

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.asset_dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.curr_dof_pos_limits = self.asset_dof_pos_limits.clone()
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.asset_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.asset_dof_pos_limits[i, 1] = props["upper"][i].item()
            self._update_dof_limits(ratio=1.0)

        for i in range(len(props)):
            if self.cfg.domain_rand.randomize_dof_damping:
                props["damping"][i] = self.setup.dof_damping[env_id, i]
            if self.cfg.domain_rand.randomize_dof_friction:
                props["friction"][i] = self.setup.dof_friction[env_id, i]
            if self.cfg.domain_rand.randomize_dof_velocity:
                props["velocity"][i] = self.setup.dof_velocity[env_id, i]
        return props

    def _update_dof_limits(self, ratio: Union[float, torch.Tensor]):
        m = (self.asset_dof_pos_limits[:, 0] + self.asset_dof_pos_limits[:, 1]) / 2
        r = self.asset_dof_pos_limits[:, 1] - self.asset_dof_pos_limits[:, 0]
        # soft limits
        self.curr_dof_pos_limits[:, 0] = m - 0.5 * r * ratio
        self.curr_dof_pos_limits[:, 1] = m + 0.5 * r * ratio

    def _process_rigid_body_props(self, props, env_id):
        # from https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/domain_randomization.md
        # > Physx only allows 64000 unique physics materials in the
        # > scene at once. If more than 64000 materials are needed,
        # > increase num_buckets to allow materials to be shared
        # > between prims.
        if len(self.cfg.domain_rand.randomize_rigid_body_masses) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_rigid_body_masses_ids
            ):
                props[body_id].mass += torch_rand_float(
                    lower=self.cfg.domain_rand.added_mass_range[0],
                    upper=self.cfg.domain_rand.added_mass_range[1],
                    shape=(1,),
                    device=self.device,
                    generator=self.generator,
                ).item()
                props[body_id].mass = max(props[body_id].mass, 0.01)
                self.setup.rigidbody_mass[env_id, idx] = props[body_id].mass
        if len(self.cfg.domain_rand.randomize_rigid_body_com) > 0:
            for idx, body_id in enumerate(
                self.cfg.domain_rand.randomize_rigid_body_com_ids
            ):
                props[body_id].com += gymapi.Vec3(
                    *torch_rand_float(
                        lower=self.cfg.domain_rand.rigid_body_com_range[0],
                        upper=self.cfg.domain_rand.rigid_body_com_range[1],
                        shape=(3,),
                        device=self.device,
                        generator=self.generator,
                    )
                    .cpu()
                    .numpy()
                    .tolist()
                )
                self.setup.rigidbody_com_offset[env_id, idx, 0] = props[body_id].com.x
                self.setup.rigidbody_com_offset[env_id, idx, 1] = props[body_id].com.y
                self.setup.rigidbody_com_offset[env_id, idx, 2] = props[body_id].com.z
        return props

    def _reset_root_states(self, env_ids):
        # base position
        self.state.root_state[env_ids] = self.base_init_state
        if (self.init_pos_noise > 0).any():
            self.state.root_state[env_ids, 0:3] += torch_rand_float(
                -self.init_pos_noise,
                self.init_pos_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        if (self.init_euler_noise > 0).any():

            euler_displacement = torch_rand_float(
                -self.init_euler_noise,
                self.init_euler_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
            matrix = p3d.euler_angles_to_matrix(euler_displacement, "XYZ")
            quat_xyzw = p3d.matrix_to_quaternion(matrix)[..., [1, 2, 3, 0]]
            self.state.root_state[env_ids, 3:7] = quat_mul(
                self.state.root_state[env_ids, 3:7], quat_xyzw
            )
        if (self.init_lin_vel_noise > 0).any():
            self.state.root_state[env_ids, 7:10] += torch_rand_float(
                -self.init_lin_vel_noise,
                self.init_lin_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        if (self.init_ang_vel_noise > 0).any():
            self.state.root_state[env_ids, 10:13] += torch_rand_float(
                -self.init_ang_vel_noise,
                self.init_ang_vel_noise,
                (len(env_ids), 3),
                device=self.device,
                generator=self.generator,
            )
        self.state.root_state[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.state.root_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    # ----------------------------------------

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))
        logging.info("Reward functions: " + ", ".join(self.reward_names))

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = (
            self.cfg.asset.replace_cylinder_with_capsule
        )
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        if not hasattr(self.cfg.domain_rand, "randomize_restitution_rigid_bodies"):
            self.cfg.domain_rand.randomize_restitution_rigid_bodies = []
        self.setup = EnvSetup(
            kp=self.controller.kp.clone(),
            kd=self.controller.kd.clone(),
            rigidbody_mass=torch.ones(
                (self.num_envs, len(self.cfg.domain_rand.randomize_rigid_body_masses)),
                device=self.device,
            ),
            rigidbody_com_offset=torch.zeros(
                (self.num_envs, len(self.cfg.domain_rand.randomize_rigid_body_com), 3),
                device=self.device,
            ),
            rigidbody_restitution_coef=torch.ones(
                (
                    self.num_envs,
                    len(self.cfg.domain_rand.randomize_restitution_rigid_bodies),
                ),
                device=self.device,
            ),
            rigid_shape_friction=torch.zeros(
                (self.num_envs, len(rigid_shape_props_asset), 3), device=self.device
            ),
            dof_damping=torch.zeros((self.num_envs, self.num_dof), device=self.device),
            dof_friction=torch.zeros((self.num_envs, self.num_dof), device=self.device),
            dof_velocity=torch.zeros((self.num_envs, self.num_dof), device=self.device),
        )

        if self.cfg.domain_rand.randomize_friction:
            # prepare friction randomization
            friction_range = self.cfg.domain_rand.friction_range
            # from https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/domain_randomization.md
            # > Physx only allows 64000 unique physics materials in the
            # > scene at once. If more than 64000 materials are needed,
            # > increase num_buckets to allow materials to be shared
            # > between prims.
            # As far as I (huy) can tell, it only applies to friction
            # and restitution and not other properties (mass, com, etc.)
            # > material_properties (dim=3): Static friction, Dynamic
            # > friction, and Restitution.
            num_buckets = self.cfg.domain_rand.num_friction_buckets
            bucket_ids = torch.randint(
                low=0,
                high=num_buckets,
                size=(self.num_envs, len(rigid_shape_props_asset)),
                device=self.device,
                generator=self.generator,
            )
            friction_buckets = torch_rand_float(
                lower=friction_range[0],
                upper=friction_range[1],
                shape=(num_buckets, 1),
                device=self.device,
                generator=self.generator,
            )
            self.setup.rigid_shape_friction = friction_buckets[bucket_ids]
        if self.cfg.domain_rand.randomize_dof_damping:
            self.setup.dof_damping[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_damping_range[0],
                upper=self.cfg.domain_rand.dof_damping_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        if self.cfg.domain_rand.randomize_dof_friction:
            self.setup.dof_friction[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_friction_range[0],
                upper=self.cfg.domain_rand.dof_friction_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        if self.cfg.domain_rand.randomize_dof_velocity:
            self.setup.dof_velocity[:] = torch_rand_float(
                lower=self.cfg.domain_rand.dof_velocity_range[0],
                upper=self.cfg.domain_rand.dof_velocity_range[1],
                shape=(self.num_envs, self.num_dof),
                device=self.device,
                generator=self.generator,
            )
        self.cfg.domain_rand.randomize_rigid_body_masses_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_rigid_body_masses
        ]

        self.cfg.domain_rand.randomize_rigid_body_com_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_rigid_body_com
        ]
        self.cfg.domain_rand.randomize_restitution_rigid_body_ids = [
            self.gym.find_asset_rigid_body_index(robot_asset, name)
            for name in self.cfg.domain_rand.randomize_restitution_rigid_bodies
        ]

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        self.init_pos_noise = to_torch(
            self.cfg.init_state.pos_noise, device=self.device, requires_grad=False
        )
        self.init_euler_noise = to_torch(
            self.cfg.init_state.euler_noise, device=self.device, requires_grad=False
        )
        self.init_lin_vel_noise = to_torch(
            self.cfg.init_state.lin_vel_noise, device=self.device, requires_grad=False
        )
        self.init_ang_vel_noise = to_torch(
            self.cfg.init_state.ang_vel_noise, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

        sensor_pose = gymapi.Transform()
        if not hasattr(self.cfg.asset, "force_sensor_links"):
            self.cfg.asset.force_sensor_links = self.cfg.asset.feet_names
        for name in self.cfg.asset.force_sensor_links:
            """
            From Legged Gym:
            > The contact forces reported by `net_contact_force_tensor` are
            > unreliable when simulating on GPU with a triangle mesh terrain.
            > A workaround is to use force sensors, but the force are
            > propagated through the sensors of consecutive bodies resulting
            > in an undesireable behaviour. However, for a legged robot it is
            > possible to add sensors to the feet/end effector only and get the
            > expected results. When using the force sensors make sure to
            > exclude gravity from trhe reported forces with
            > `sensor_options.enable_forward_dynamics_forces`
            """
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False
            sensor_options.enable_constraint_solver_forces = True
            sensor_options.use_world_frame = True
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(
                robot_asset, index, sensor_pose, sensor_options
            )

        self.env_origins = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False
        )
        # create a grid of robots
        env_lower = gymapi.Vec3(
            -self.env_spacing,
            -self.env_spacing,
            0,
        )
        env_upper = gymapi.Vec3(
            self.env_spacing,
            self.env_spacing,
            self.env_spacing,
        )
        self.actor_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            origin = self.gym.get_env_origin(env_handle)
            self.env_origins[i, 0] = origin.x
            self.env_origins[i, 1] = origin.y
            self.env_origins[i, 2] = origin.z
            rigid_shape_props = self._process_rigid_shape_props(
                rigid_shape_props_asset, i
            )
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, actor_handle
            )
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        if (self.termination_contact_indices == -1).any():
            raise ValueError(
                f"Could not find all termination links in actor {self.gym.get_actor_name(self.envs[0], 0)!r}"
            )

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, state: EnvState, control: Control):
        # Penalize z axis base linear velocity
        return torch.square(state.local_root_lin_vel[:, 2])

    def _reward_ang_vel_xy(self, state: EnvState, control: Control):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(state.local_root_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self, state: EnvState, control: Control):
        # Penalize non flat base orientation
        return torch.sum(
            torch.square(state.local_root_gravity[:, :2]),
            dim=1,
        )

    def visualize(self, vis_env_ids: List[int]):
        """
        Draws all the trajectory position target lines.
        """
        self.gym.clear_lines(self.viewer)
        for task in self.tasks.values():
            task.visualize(
                state=self.state, viewer=self.viewer, vis_env_ids=vis_env_ids
            )
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        if self.cfg.terrain.mode in {"perlin"}:
            for i in vis_env_ids:
                base_pos = (self.state.root_pos[i, :3]).cpu().numpy()
                heights = self.state.measured_terrain_heights[i].cpu().numpy()
                height_points = (
                    quat_apply_yaw(
                        self.state.root_xyzw_quat[i].repeat(heights.shape[0]),
                        self.height_points[i],
                    )
                    .cpu()
                    .numpy()
                )
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(
                        sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                    )

    def _reset_dofs(self, env_ids):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_pos_range = self.curr_dof_pos_limits[:, 1] - self.curr_dof_pos_limits[:, 0]
        dof_pos_range[torch.isnan(dof_pos_range) | torch.isinf(dof_pos_range)] = 1.0
        self.state.dof_pos[env_ids] = torch.clip(
            self.init_dof_pos
            + (
                self.dof_pos_reset_range_scale
                * torch.randn(
                    len(env_ids),
                    self.state.dof_pos.shape[1],
                    device=self.device,
                    generator=self.generator,
                )
                * dof_pos_range
            ),
            min=self.curr_dof_pos_limits[:, 0],
            max=self.curr_dof_pos_limits[:, 1],
        )
        self.state.prev_dof_pos[env_ids] = self.state.dof_pos[env_ids].clone()
        self.state.dof_vel[env_ids] = 0.0
        self.state.prev_dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(
                torch.stack(
                    (
                        self.state.dof_pos,
                        self.state.dof_vel,
                    ),
                    dim=-1,
                )
            ),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_heights(self, env_ids=None):
        """Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mode == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mode == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(
                self.state.root_xyzw_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + self.state.root_pos.unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.state.root_xyzw_quat.repeat(1, self.num_height_points),
                self.height_points,
            ) + self.state.root_pos.unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points / self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    def _init_height_points(self):
        """Returns points at which the height measurements are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(
            self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False
        )
        x = torch.tensor(
            self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False
        )
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def __del__(self):
        # NOTE: this destructor still results in segfaults upon exit.
        # Need to investigate further.
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

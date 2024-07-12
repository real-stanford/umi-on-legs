# Modified from https://github.com/leggedrobotics/legged_gym/

import logging
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union
import typing

import numpy as np
import torch
from rich.progress import track

import wandb
from legged_gym.rsl_rl.algorithms import PPO
from legged_gym.rsl_rl.env import VecEnv
from legged_gym.rsl_rl.runners.utils import parse_rollout_stats
from legged_gym.rsl_rl.storage.rollout_storage import RolloutStorage

Policy = Callable[[torch.Tensor], torch.Tensor]
RenderCallback = Callable[[np.ndarray], None]


class OnPolicyRunner:
    def __init__(
        self,
        env: VecEnv,
        alg: PPO,
        ckpt_dir: str,
        device: str,
        num_transitions_per_env: int,
        ckpt_save_interval: int,
        max_iterations: int,
        init_at_random_ep_len: bool,
        vis_resolution: Tuple[int, int],
        eval_freq: int,
        num_eval_episode_per_env: int,
        eval_fn: Optional[
            Callable[
                [str],
                Dict[str, Union[float, int, bool]],
            ]
        ] = None,
        vis_steps: int = 200,
        vis_freq: int = 6000,
        eval_buf_device: str = "cpu",
    ):
        self.device = device
        self.init_at_random_ep_len = init_at_random_ep_len
        self.env = env
        self.alg = alg
        self.ckpt_save_interval = ckpt_save_interval
        self.num_transitions_per_env = num_transitions_per_env
        self.vis_steps = vis_steps
        self.vis_freq = vis_freq
        self.total_sim_steps = 0

        # init storage and model
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_transitions_per_env,
            obs_shape=self.env.num_obs,
            privileged_obs_shape=self.env.num_privileged_obs,
            action_shape=self.env.num_actions,
            actor_obs_history_len=self.env.obs_history_len,
        )

        # Log
        self.ckpt_dir = ckpt_dir
        self.max_iterations = max_iterations
        self.current_learning_iteration = 0
        self.vis_resolution = vis_resolution

        self.eval_freq = eval_freq
        self.eval_fn = eval_fn
        self.num_eval_episode_per_env = num_eval_episode_per_env
        self.eval_storage = RolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.env.max_episode_length
            * (self.num_eval_episode_per_env + 2),
            obs_shape=self.env.num_obs * self.env.obs_history_len,
            privileged_obs_shape=self.env.num_privileged_obs,
            action_shape=self.env.num_actions,
            device=eval_buf_device,
        )
        self.train_storage = self.alg.storage
        assert (
            self.eval_freq >= self.ckpt_save_interval
            and self.eval_freq % self.ckpt_save_interval == 0
        )

    def eval(self, use_pbar: bool = False):
        should_push_robots = self.env.cfg.domain_rand.push_robots
        should_transport_robots = self.env.cfg.domain_rand.transport_robots
        self.env.cfg.domain_rand.push_robots = False
        self.env.cfg.domain_rand.transport_robots = False
        obs, privileged_obs = self.env.reset()
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)
        policy = self.alg.get_inference_policy(self.device)
        vis_frames = []
        start = time.time()
        final_rollout_stats: Dict[str, Deque[float]] = {
            k: deque(maxlen=self.env.num_envs * self.num_eval_episode_per_env)
            for k in ["reward", "return", "episode_len"]
        }
        cum_rollout_stats: Dict[str, torch.Tensor] = {
            k: torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.eval_storage.device
            )
            for k in ["reward", "return", "episode_len"]
        }
        self.eval_storage.clear()
        self.alg.storage = self.eval_storage
        # Rollout
        eval_episodes = 0

        total_steps = self.env.max_episode_length * (self.num_eval_episode_per_env + 2)
        pbar = range(total_steps)
        if use_pbar:
            pbar = track(
                pbar,
                description="Evaluating",
                total=total_steps,
            )
        with torch.inference_mode(), torch.no_grad():
            stats = {}
            for _ in pbar:
                action = policy(obs)
                action = action.view(self.env.num_envs, -1)[:, : self.env.ctrl.ctrl_dim]
                obs, privileged_obs, rewards, dones, infos = self.env.step(
                    action=action,
                    return_vis=True,
                )
                eval_episodes += dones.sum().item()
                privileged_obs = privileged_obs if privileged_obs is not None else obs
                obs, privileged_obs, rewards, dones = (
                    obs.to(self.device),
                    privileged_obs.to(self.device),
                    rewards.to(self.device),
                    dones.to(self.device),
                )
                self.alg.process_env_step(rewards, dones, infos)
                if eval_episodes > self.env.num_envs * self.num_eval_episode_per_env:
                    break

            stop = time.time()
            stats["eval/time"] = stop - start
            stats["eval/num_episodes"] = eval_episodes
        stats.update(
            {
                f"eval/{k}": v
                for k, v in parse_rollout_stats(
                    storage=self.alg.storage,
                    cum_rollout_stats=cum_rollout_stats,
                    final_rollout_stats=final_rollout_stats,
                    gamma=self.alg.gamma,
                    vis_frames=vis_frames,
                    vis_resolution=self.vis_resolution,
                ).items()
            }
        )
        frames = np.stack(vis_frames)  # (num_frames, height, width, 3)
        # wandb requires (num_frames, height, width, 3) -> (num_frames, 3, height, width)
        stats["eval/vis"] = wandb.Video(
            frames.transpose(0, 3, 1, 2),
            fps=int(1 / self.env.gym_dt),
            format="mp4",
        )
        vis_frames.clear()
        self.alg.storage = self.train_storage
        self.env.cfg.domain_rand.push_robots = should_push_robots
        self.env.cfg.domain_rand.transport_robots = should_transport_robots
        return stats

    def learn(self):

        # initialize writer
        assert wandb.run is not None

        obs, privileged_obs = self.env.reset()
        if self.init_at_random_ep_len:
            self.env.state.episode_time = (
                torch.randint_like(
                    self.env.state.episode_time,
                    low=int(0),
                    high=int(self.env.max_episode_length_s / self.env.gym_dt),
                )
            ) * self.env.gym_dt
        obs, privileged_obs = obs.to(self.device), privileged_obs.to(self.device)

        final_rollout_stats: Dict[str, Deque[float]] = {
            k: deque(maxlen=self.env.num_envs * self.num_transitions_per_env)
            for k in ["reward", "return", "episode_len"]
        }
        cum_rollout_stats: Dict[str, torch.Tensor] = {
            k: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
            for k in ["reward", "return", "episode_len"]
        }

        tot_iter = self.current_learning_iteration + self.max_iterations
        vis_frames = []
        for it in track(
            range(self.current_learning_iteration, tot_iter),
            description="Training",
            total=tot_iter,
        ):
            start = time.time()
            self.alg.actor_critic.train()
            # Rollout
            with torch.inference_mode():
                stats = {}
                for _ in range(self.num_transitions_per_env):
                    action = self.alg.act(obs, privileged_obs)
                    return_vis = bool(
                        self.vis_freq != -1
                        and (
                            (self.total_sim_steps - np.arange(0, self.vis_steps))
                            % self.vis_freq
                            == 0
                        ).any()
                    )
                    obs, privileged_obs, rewards, dones, infos = self.env.step(
                        action=action,
                        return_vis=return_vis,
                    )
                    privileged_obs = (
                        privileged_obs if privileged_obs is not None else obs
                    )
                    obs, privileged_obs, rewards, dones = (
                        obs.to(self.device),
                        privileged_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    # Book keeping
                    self.total_sim_steps += 1

                stop = time.time()
                stats["system/collection_time"] = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(privileged_obs)
                assert self.alg.storage.step == self.num_transitions_per_env
            stats.update(
                {
                    f"episode/{k}": v
                    for k, v in parse_rollout_stats(
                        storage=self.alg.storage,
                        cum_rollout_stats=cum_rollout_stats,
                        final_rollout_stats=final_rollout_stats,
                        gamma=self.alg.gamma,
                        vis_frames=vis_frames,
                        vis_resolution=self.vis_resolution,
                    ).items()
                }
            )

            stats.update(
                {f"train/{k}": v for k, v in self.alg.update(learning_iter=it).items()}
            )
            self.alg.storage.clear()  # on policy, so we can clear the storage
            stop = time.time()
            stats["system/learn_time"] = stop - start
            stats["system/total_fps"] = int(
                self.num_transitions_per_env
                * self.env.num_envs
                / (stats["system/collection_time"] + stats["system/learn_time"])
            )

            if len(vis_frames) >= self.vis_steps:
                frames = np.stack(vis_frames)  # (num_frames, height, width, 3)
                # wandb requires (num_frames, height, width, 3) -> (num_frames, 3, height, width)
                stats["episode/vis"] = wandb.Video(
                    frames.transpose(0, 3, 1, 2),
                    fps=int(1 / self.env.gym_dt),
                    format="mp4",
                )
                vis_frames.clear()

            wandb.log(stats, step=it)
            self.current_learning_iteration += 1
            if it % self.ckpt_save_interval == 0 and it > 0:
                ckpt_path = os.path.join(self.ckpt_dir, "model_{}.pt".format(it))
                self.save(ckpt_path)
                if it % self.eval_freq == 0:
                    with torch.inference_mode():
                        if self.eval_fn is not None:
                            wandb.log(self.eval_fn(ckpt_path), step=it)
                        else:
                            wandb.log(self.eval(), step=it)
                            obs, privileged_obs = self.env.reset()
                            if self.init_at_random_ep_len:
                                self.env.state.episode_time = (
                                    torch.randint_like(
                                        self.env.state.episode_time,
                                        low=int(0),
                                        high=int(
                                            self.env.max_episode_length_s
                                            / self.env.gym_dt
                                        ),
                                    )
                                ) * self.env.gym_dt
                            obs, privileged_obs = obs.to(
                                self.device
                            ), privileged_obs.to(self.device)

        self.save(
            os.path.join(
                self.ckpt_dir, "model_{}.pt".format(self.current_learning_iteration)
            )
        )

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.get_model_state_dict(),
                "optimizer_state_dict": self.alg.get_optimizer_state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        # self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from git import Optional

from legged_gym.rsl_rl.modules import ActorCritic
from legged_gym.rsl_rl.storage import RolloutStorage


class PPO:
    def __init__(
        self,
        actor_critic: ActorCritic,
        num_learning_epochs: int,  # number of epochs to learn from batch
        num_mini_batches: int,  # number of mini batches to split the batch into
        clip_param: float,  # clip parameter for PPO
        gamma: float,  # discount factor
        lam: float,  # lambda for GAE-Lambda
        value_loss_coef: float,  # value loss coefficient
        entropy_coef: float,  # entropy coefficient
        learning_rate: float,  # learning rate
        max_grad_norm: float,  # max norm of gradients
        use_clipped_value_loss: float,  # whether to use clipped value loss
        schedule: str,  # learning rate schedule, either 'fixed' or 'adaptive'
        desired_kl: float,  # desired kl value for adaptive learning rate
        max_lr: float,  # maximum learning rate for adaptive learning rate
        min_lr: float,  # minimum learning rate for adaptive learning rate
        device: str,  # device to use for training
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage: RolloutStorage  # initialized later
        self.optimizer = optim.Adam(self.get_parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.max_lr = max_lr
        self.min_lr = min_lr

    def get_parameters(self):
        return self.actor_critic.parameters()

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: int,
        privileged_obs_shape: int,
        action_shape: int,
        actor_obs_history_len: int,
    ):
        self.storage = RolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=obs_shape * actor_obs_history_len,
            privileged_obs_shape=privileged_obs_shape,
            action_shape=action_shape,
            device=self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.infos = infos
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        return self.storage.compute_returns(last_values, self.gamma, self.lam)

    def get_ppo_actor_actions(
        self, obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor
    ):
        self.actor_critic.act(obs)
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(action)
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy
        return actions_log_prob_batch, mu_batch, sigma_batch, entropy_batch

    def update_per_batch(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        target_value: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
        old_action_log_prob: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        hid_state: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        mask: Optional[torch.Tensor],
        learning_iter: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats = {}
        value = self.actor_critic.evaluate(critic_obs)
        actions_log_prob, mu_batch, sigma_batch, entropy_batch = (
            self.get_ppo_actor_actions(obs, critic_obs, action)
        )

        # KL
        if self.desired_kl != None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma + 1.0e-5)
                    + (torch.square(old_sigma) + torch.square(old_mu - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_action_log_prob))
        stats["mean_raw_ratios"] = ratio.mean().item()
        stats["mean_clipped_ratio"] = (
            ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param))
            .float()
            .mean()
            .item()
        )
        surrogate = -torch.squeeze(advantage) * ratio
        surrogate_clipped = -torch.squeeze(advantage) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_value + (value - target_value).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns - value).pow(2).mean()

        stats["mean_value_loss"] = value_loss.item()
        stats["mean_surrogate_loss"] = surrogate_loss.item()

        loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy_batch.mean()
        )

        return loss, stats

    def update(self, learning_iter: int) -> Dict[str, float]:
        update_stats = {}
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        for (
            obs_batch,
            critic_obs,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            loss, stats = self.update_per_batch(
                obs=obs_batch,
                critic_obs=critic_obs,
                action=actions_batch,
                target_value=target_values_batch,
                advantage=advantages_batch,
                returns=returns_batch,
                old_action_log_prob=old_actions_log_prob_batch,
                old_mu=old_mu_batch,
                old_sigma=old_sigma_batch,
                hid_state=hid_states_batch,
                mask=masks_batch,
                learning_iter=learning_iter,
            )
            for k, v in stats.items():
                if k not in update_stats:
                    update_stats[k] = 0.0
                update_stats[k] += v

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            **{k: v / num_updates for k, v in update_stats.items()},
            **{
                "learning_rate": self.learning_rate,
                "action_std": self.actor_critic.std.mean().item(),
            },
        }

    def get_model_state_dict(self) -> Dict:
        return self.actor_critic.state_dict()

    def get_optimizer_state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def get_inference_policy(self, device: Optional[Union[torch.Tensor, str]] = None):
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic = self.actor_critic.to(device)

        def actor(obs: torch.Tensor):
            with torch.inference_mode():
                return self.actor_critic.act_inference(obs)

        return actor

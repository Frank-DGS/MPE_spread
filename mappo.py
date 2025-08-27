from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from config import train_cfg, net_cfg
from models.networks import CategoricalActor, CentralizedCritic


@dataclass
class MAPPOComponents:
    actors: List[CategoricalActor]
    critic: CentralizedCritic
    actor_optimizers: List[optim.Optimizer]
    critic_optimizer: optim.Optimizer


class MAPPO:
    def __init__(self, obs_dim: int, act_dim: int, central_obs_dim: int, num_agents: int, device: torch.device):
        self.num_agents = num_agents
        self.device = device

        # 同构智能体：共享策略或独立策略均可。这里采用参数共享的单一actor，执行时为每个agent计算分布。
        shared_actor = CategoricalActor(obs_dim, act_dim, net_cfg.actor_hidden_sizes, net_cfg.activation).to(device)
        self.actors = [shared_actor] * num_agents
        self.critic = CentralizedCritic(central_obs_dim, net_cfg.critic_hidden_sizes, net_cfg.activation).to(device)

        self.actor_params = list(shared_actor.parameters())
        self.critic_params = list(self.critic.parameters())

        self.actor_optimizer = optim.Adam(self.actor_params, lr=train_cfg.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=train_cfg.learning_rate)

        self.clip_range = train_cfg.ppo_clip
        self.entropy_coef = train_cfg.entropy_coef
        self.value_loss_coef = train_cfg.value_loss_coef
        self.max_grad_norm = train_cfg.max_grad_norm
        self.advantage_norm = train_cfg.advantage_norm
        self.value_clip = train_cfg.value_clip
        self.use_huber_loss = train_cfg.use_huber_loss
        self.huber_delta = train_cfg.huber_delta

        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss(beta=self.huber_delta)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        # obs: [N, obs_dim]; actions: [N]
        dist = self.actors[0](obs)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logprobs, entropy

    def evaluate_values(self, central_obs: torch.Tensor):
        values = self.critic(central_obs).squeeze(-1)
        return values

    def _value_loss(self, new_values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        if self.value_clip:
            clipped = old_values + (new_values - old_values).clamp(-self.clip_range, self.clip_range)
            loss1 = (new_values - returns).pow(2)
            loss2 = (clipped - returns).pow(2)
            value_loss = torch.max(loss1, loss2).mean()
        else:
            if self.use_huber_loss:
                value_loss = self.huber_loss(new_values, returns)
            else:
                value_loss = self.mse_loss(new_values, returns)
        return value_loss

    def update(self, batch_iterable, ppo_epoch: int):
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0, "num_updates": 0}

        for _ in range(ppo_epoch):
            for batch in batch_iterable:
                obs = batch["obs"]
                central_obs = batch["central_obs"]
                actions = batch["actions"]
                old_logprobs = batch["old_logprobs"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["old_values"]

                if self.advantage_norm:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

                # 策略部分
                new_logprobs, entropy = self.evaluate_actions(obs, actions)
                log_ratio = new_logprobs - old_logprobs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                approx_kl = 0.5 * (log_ratio.pow(2)).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()

                # 价值部分
                new_values = self.evaluate_values(central_obs)
                value_loss = self._value_loss(new_values, old_values, returns)

                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy)
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["num_updates"] += 1

        # 平均化
        if stats["num_updates"] > 0:
            n = stats["num_updates"]
            for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
                stats[k] /= n
        return stats



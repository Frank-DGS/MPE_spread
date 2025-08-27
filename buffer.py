from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class BufferSpec:
    num_steps: int
    num_envs: int
    num_agents: int
    obs_dim: int
    central_obs_dim: int


class OnPolicyBuffer:
    """
    按MAPPO收集：
    - obs: [T, E, A, obs_dim]
    - central_obs: [T, E, 1, central_obs_dim] 同时用于所有agent的价值评估
    - actions, logprobs, rewards, dones, values

    回放结束后计算GAE与returns。
    """

    def __init__(self, spec: BufferSpec, device: torch.device):
        self.spec = spec
        self.device = device
        T, E, A = spec.num_steps, spec.num_envs, spec.num_agents
        od, cd = spec.obs_dim, spec.central_obs_dim

        self.obs = torch.zeros(T, E, A, od, device=device)
        self.central_obs = torch.zeros(T, E, 1, cd, device=device)
        self.actions = torch.zeros(T, E, A, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(T, E, A, device=device)
        self.rewards = torch.zeros(T, E, A, device=device)
        self.dones = torch.zeros(T, E, A, device=device)
        self.values = torch.zeros(T, E, A, device=device)

        self.advantages = torch.zeros(T, E, A, device=device)
        self.returns = torch.zeros(T, E, A, device=device)

        self.step = 0

    def add(self, obs, central_obs, actions, logprobs, rewards, dones, values):
        t = self.step
        self.obs[t].copy_(obs)
        self.central_obs[t].copy_(central_obs)
        self.actions[t].copy_(actions)
        # 存储old_logprobs/old_values需与计算图分离
        self.logprobs[t].copy_(logprobs.detach())
        self.rewards[t].copy_(rewards)
        self.dones[t].copy_(dones)
        self.values[t].copy_(values.detach())
        self.step += 1

    @torch.no_grad()
    def compute_gae(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        T, E, A = self.spec.num_steps, self.spec.num_envs, self.spec.num_agents
        advantages = torch.zeros_like(self.rewards)
        last_adv = torch.zeros(E, A, device=self.device)

        for t in reversed(range(T)):
            # dones: 1.0 when done else 0.0
            next_values = last_values if t == T - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * not_done - self.values[t]
            last_adv = delta + gamma * gae_lambda * not_done * last_adv
            advantages[t] = last_adv

        self.advantages.copy_(advantages)
        self.returns.copy_(self.advantages + self.values)

    def get_training_batches(self, num_mini_batch: int):
        # 展平为 [T*E*A, ...]
        T, E, A = self.spec.num_steps, self.spec.num_envs, self.spec.num_agents
        N = T * E * A
        obs = self.obs.reshape(N, -1)
        central_obs = self.central_obs.expand(T, E, A, -1).reshape(N, -1)
        actions = self.actions.reshape(N)
        logprobs = self.logprobs.reshape(N)
        advantages = self.advantages.reshape(N)
        returns = self.returns.reshape(N)
        values = self.values.reshape(N)

        indices = torch.randperm(N, device=self.device)
        mini_size = N // num_mini_batch
        for i in range(num_mini_batch):
            idx = indices[i * mini_size : (i + 1) * mini_size]
            yield {
                "obs": obs[idx],
                "central_obs": central_obs[idx],
                "actions": actions[idx],
                "old_logprobs": logprobs[idx],
                "advantages": advantages[idx],
                "returns": returns[idx],
                "old_values": values[idx],
            }



from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from config import env_cfg, train_cfg
from mappo import MAPPO
from buffer import BufferSpec, OnPolicyBuffer


def make_env(render_mode=None):
    # PettingZoo simple_spread v3
    from pettingzoo.mpe import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=env_cfg.num_agents,
        local_ratio=env_cfg.local_ratio,
        max_cycles=env_cfg.max_cycles,
        continuous_actions=env_cfg.continuous_actions,
        render_mode=render_mode,
    )
    return env


def concat_central_obs(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    # 将所有agent的obs拼接为central obs
    agent_keys = sorted(obs_dict.keys())
    return np.concatenate([obs_dict[k] for k in agent_keys], axis=-1)


def reset_env(env):
    res = env.reset()
    # 兼容新旧API：新API返回 (obs, infos)
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], dict):
        obs, _infos = res
    else:
        obs = res
    return obs


def step_env(env, actions):
    res = env.step(actions)
    # 新API: (obs, rewards, terminations, truncations, infos)
    if isinstance(res, tuple) and len(res) == 5 and isinstance(res[0], dict):
        obs, rewards, terminations, truncations, infos = res
        dones = {aid: bool(terminations.get(aid, False) or truncations.get(aid, False)) for aid in rewards.keys()}
    else:
        # 旧API: (obs, rewards, dones, infos)
        obs, rewards, dones, infos = res
    return obs, rewards, dones, infos


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    env = make_env(render_mode=None)
    _ = reset_env(env)

    # 空间维度
    example_agent = env.agents[0]
    obs_dim = env.observation_space(example_agent).shape[0]
    act_dim = env.action_space(example_agent).n
    central_obs_dim = obs_dim * env_cfg.num_agents

    algo = MAPPO(obs_dim, act_dim, central_obs_dim, env_cfg.num_agents, device)

    # Buffer
    spec = BufferSpec(
        num_steps=train_cfg.rollout_length,
        num_envs=train_cfg.num_envs,
        num_agents=env_cfg.num_agents,
        obs_dim=obs_dim,
        central_obs_dim=central_obs_dim,
    )
    buffer = OnPolicyBuffer(spec, device)

    # 训练日志
    global_step = 0
    ep_returns = []
    recent_ret = 0.0
    episode_count = 0
    t_start = time.time()

    obs = reset_env(env)
    # 统一agent顺序
    agent_order = sorted(env.agents)

    while global_step < train_cfg.total_timesteps:
        buffer.step = 0

        for t in range(train_cfg.rollout_length):
            # obs dict -> tensors
            obs_mat = np.stack([obs[aid] for aid in agent_order], axis=0)  # [A, obs]
            central_obs = concat_central_obs(obs)  # [A*obs]

            obs_tensor = torch.tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(0)  # [E=1, A, obs]
            central_tensor = torch.tensor(central_obs, dtype=torch.float32, device=device).view(1, 1, -1)  # [E=1,1,centr]

            actions = {}
            logprobs = []
            values = []

            for i, aid in enumerate(agent_order):
                dist = algo.actors[0](obs_tensor[0, i])
                action = dist.sample()
                actions[aid] = int(action.item())
                logprobs.append(dist.log_prob(action).unsqueeze(0))

            # 值函数对central obs
            v = algo.evaluate_values(central_tensor.view(1, -1)).view(1, 1)
            v = v.expand(1, len(agent_order))  # broadcast to each agent

            next_obs, rewards, dones, infos = step_env(env, actions)

            reward_vec = torch.tensor([rewards[aid] for aid in agent_order], dtype=torch.float32, device=device).unsqueeze(0)
            done_vec = torch.tensor([float(dones[aid]) for aid in agent_order], dtype=torch.float32, device=device).unsqueeze(0)
            logprob_tensor = torch.cat(logprobs, dim=0).view(1, -1)

            buffer.add(
                obs=obs_tensor,
                central_obs=central_tensor,
                actions=torch.tensor([actions[aid] for aid in agent_order], device=device).view(1, -1),
                logprobs=logprob_tensor,
                rewards=reward_vec,
                dones=done_vec,
                values=v,
            )

            obs = next_obs
            global_step += env_cfg.num_agents

            # episode logging（simple_spread为稀疏回报，可累计）
            recent_ret += reward_vec.sum().item()
            if all(dones.values()):
                ep_returns.append(recent_ret)
                recent_ret = 0.0
                episode_count += 1
                obs = reset_env(env)

        # bootstrap value for last step central obs
        last_central_obs = concat_central_obs(obs)
        last_v = algo.evaluate_values(torch.tensor(last_central_obs, dtype=torch.float32, device=device).unsqueeze(0)).detach()
        last_v = last_v.view(1, 1).expand(1, env_cfg.num_agents)

        buffer.compute_gae(last_values=last_v, gamma=train_cfg.gamma, gae_lambda=train_cfg.gae_lambda)

        # 更新
        stats = algo.update(buffer.get_training_batches(train_cfg.num_mini_batch), ppo_epoch=train_cfg.ppo_epoch)

        # 日志输出
        if global_step % train_cfg.log_interval < env_cfg.num_agents:
            fps = int(global_step / (time.time() - t_start + 1e-8))
            avg_ret = np.mean(ep_returns[-10:]) if len(ep_returns) > 0 else 0.0
            print(
                f"step={global_step} | fps={fps} | avg_ret(10)={avg_ret:.2f} | "
                f"policy={stats['policy_loss']:.3f} value={stats['value_loss']:.3f} "
                f"ent={stats['entropy']:.3f} kl={stats['approx_kl']:.4f} clip={stats['clip_frac']:.2f}"
            )

        # 周期性保存
        if global_step % train_cfg.save_interval < env_cfg.num_agents:
            ckpt = os.path.join(train_cfg.model_dir, f"mappo_spread_step{global_step}.pt")
            torch.save({
                "actor": algo.actors[0].state_dict(),
                "critic": algo.critic.state_dict(),
                "cfg": {
                    "env": vars(env_cfg),
                    "train": vars(train_cfg),
                },
                "stats": stats,
                "ep_returns": ep_returns,
            }, ckpt)
            # 也保存最近一个简易快照
            torch.save(algo.actors[0].state_dict(), os.path.join(train_cfg.model_dir, "actor_latest.pt"))
            torch.save(algo.critic.state_dict(), os.path.join(train_cfg.model_dir, "critic_latest.pt"))

    # 最终保存
    torch.save(algo.actors[0].state_dict(), os.path.join(train_cfg.model_dir, "actor_final.pt"))
    torch.save(algo.critic.state_dict(), os.path.join(train_cfg.model_dir, "critic_final.pt"))

    # 保存训练曲线数据
    np.save(os.path.join(train_cfg.log_dir, "episode_returns.npy"), np.array(ep_returns, dtype=np.float32))


if __name__ == "__main__":
    main()



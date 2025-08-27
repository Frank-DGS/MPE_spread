from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch

from config import env_cfg
from models.networks import CategoricalActor


def make_env(render_mode="rgb_array"):
    from pettingzoo.mpe import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=env_cfg.num_agents,
        local_ratio=env_cfg.local_ratio,
        max_cycles=env_cfg.max_cycles,
        continuous_actions=env_cfg.continuous_actions,
        render_mode=render_mode,
    )
    return env


def reset_env(env):
    res = env.reset()
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[0], dict):
        obs, _infos = res
    else:
        obs = res
    return obs


def step_env(env, actions):
    res = env.step(actions)
    if isinstance(res, tuple) and len(res) == 5 and isinstance(res[0], dict):
        obs, rewards, terminations, truncations, infos = res
        dones = {aid: bool(terminations.get(aid, False) or truncations.get(aid, False)) for aid in rewards.keys()}
    else:
        obs, rewards, dones, infos = res
    return obs, rewards, dones, infos


def run_episode(actor_path: str, out_dir: str, video_prefix: str = "eval"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(render_mode="rgb_array")
    _ = reset_env(env)
    agent_id = env.agents[0]
    obs_dim = env.observation_space(agent_id).shape[0]
    act_dim = env.action_space(agent_id).n

    actor = CategoricalActor(obs_dim, act_dim, (64, 64), "relu").to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    frames = []
    obs = reset_env(env)
    total_reward = 0.0

    for t in range(env_cfg.max_cycles):
        actions = {}
        for aid in env.agents:
            o = torch.tensor(obs[aid], dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = actor(o)
                a = torch.argmax(dist.probs).item()
            actions[aid] = a

        obs, rewards, dones, infos = step_env(env, actions)
        total_reward += sum(rewards.values())

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if all(dones.values()):
            break

    # 保存帧为npz与若干关键帧png
    np.savez_compressed(os.path.join(out_dir, f"{video_prefix}_frames.npz"), frames=np.array(frames, dtype=object))
    # 抽取关键帧
    try:
        import imageio.v2 as imageio
        num_snap = min(8, len(frames))
        idxs = np.linspace(0, max(len(frames)-1, 0), num=num_snap, dtype=int)
        for i, idx in enumerate(idxs):
            imageio.imwrite(os.path.join(out_dir, f"{video_prefix}_snap_{i:02d}.png"), frames[idx])
    except Exception:
        pass

    print(f"Episode total reward: {total_reward:.2f}, frames saved: {len(frames)}")


if __name__ == "__main__":
    # 默认读取最新actor
    run_episode(actor_path=os.path.join("checkpoints", "actor_latest.pt"), out_dir="eval_outputs")



"""
MAPPO 在 MPE simple_spread 任务的配置项。
可直接修改变量进行调参；训练脚本会从此处读取。
"""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    env_name: str = "simple_spread_v3"
    num_agents: int = 3
    num_landmarks: int = 3
    max_cycles: int = 25  # 每个episode的步数（MPE默认25）
    local_ratio: float = 0.5
    continuous_actions: bool = False
    render_mode: str | None = None  # 训练时None；评估时可设为"rgb_array"


@dataclass
class TrainConfig:
    # MAPPO核心超参数（参考论文常用设置）
    learning_rate: float = 7e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epoch: int = 10
    num_mini_batch: int = 1  # MLP可使用>1的小批次

    # 训练时长与采样规模
    total_timesteps: int = 10_000_000
    rollout_length: int = 25  # 与max_cycles一致，按整局收集
    num_envs: int = 16  # 简化为单环境并行，可后续改大

    # 归一化与训练技巧
    advantage_norm: bool = True
    value_clip: bool = True
    use_huber_loss: bool = False
    huber_delta: float = 1.0
    reward_norm: bool = False  # 若需要，可在buffer中实现回报归一化

    # 日志与保存
    log_interval: int = 1000  # 每多少环境步打印一次
    save_interval: int = 50_000  # 每多少环境步保存一次
    model_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class NetConfig:
    actor_hidden_sizes: tuple[int, int] = (64, 64)
    critic_hidden_sizes: tuple[int, int] = (64, 64)
    activation: str = "tanh"  # relu | tanh


env_cfg = EnvConfig()
train_cfg = TrainConfig()
net_cfg = NetConfig()



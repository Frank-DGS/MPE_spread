# MAPPO多智能体强化学习项目

本项目实现了MAPPO（Multi-Agent Proximal Policy Optimization）算法在MPE（Multi-Agent Particle Environment）simple_spread任务上的应用。MAPPO是PPO算法在多智能体环境下的扩展，采用集中式训练、分散式执行的范式，有效解决了多智能体环境中的非平稳性问题。

## 项目概述

### 算法特点
- **集中式训练，分散式执行**：训练时使用集中式价值网络，执行时每个智能体独立决策
- **参数共享**：同构智能体共享策略网络参数，提高训练效率
- **PPO稳定性**：采用PPO截断机制，确保策略更新稳定性
- **GAE优势估计**：使用广义优势估计，平衡偏差与方差

### 任务描述
simple_spread是一个经典的多智能体协作任务：
- 3个智能体需要移动到3个不同的地标位置
- 智能体需要避免相互碰撞
- 目标是最小化总移动距离和碰撞惩罚
- 每个episode最多25步

## 项目结构

```
MPE_spread/
├── config.py                 # 配置文件，包含环境、训练和网络参数
├── mappo.py                  # MAPPO算法核心实现
├── train_spread.py           # 训练脚本
├── evaluate_spread.py        # 评估脚本
├── buffer.py                 # 经验回放缓冲区实现
├── models/
│   └── networks.py           # 神经网络架构定义
├── utils/
│   └── plotting.py           # 绘图工具
├── checkpoints/              # 模型检查点保存目录
├── logs/                     # 训练日志和曲线图
└── eval_outputs/             # 评估结果输出目录
```

## 核心文件说明

### 配置文件 (config.py)
- `EnvConfig`: 环境相关配置（智能体数量、地标数量、episode长度等）
- `TrainConfig`: 训练超参数（学习率、折扣因子、PPO参数等）
- `NetConfig`: 网络架构参数（隐藏层大小、激活函数等）

### 算法实现 (mappo.py)
- `MAPPO`: 主要的算法类，包含策略更新逻辑
- `CategoricalActor`: 离散动作策略网络
- `CentralizedCritic`: 集中式价值网络

### 训练脚本 (train_spread.py)
- 环境初始化和数据收集
- 训练循环和模型更新
- 日志记录和模型保存

### 评估脚本 (evaluate_spread.py)
- 加载训练好的模型
- 运行episode并生成可视化结果
- 保存关键帧图像

## 环境要求

### Python版本
- Python 3.8+

### 主要依赖
```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.3.0
pettingzoo>=1.17.0
imageio>=2.9.0
```

### 安装依赖
```bash
pip install torch numpy matplotlib pettingzoo imageio
```

## 使用方法

### 1. 训练模型

```bash
python train_spread.py
```

训练过程将：
- 自动创建checkpoints和logs目录
- 每1000步输出训练日志
- 每50000步保存模型检查点
- 训练完成后保存最终模型

### 2. 评估模型

```bash
python evaluate_spread.py
```

评估过程将：
- 加载最新的训练模型
- 运行一个完整的episode
- 生成关键帧图像保存到eval_outputs目录
- 输出episode总回报

### 3. 查看训练曲线

训练完成后，可以在logs目录中找到：
- `episode_returns.npy`: 训练回报数据
- `spread_mappo_curve.png`: 训练曲线图

### 4. 自定义配置

可以通过修改config.py中的参数来调整训练设置：

```python
# 调整学习率
train_cfg.learning_rate = 1e-3

# 修改网络架构
net_cfg.actor_hidden_sizes = (128, 128)

# 增加训练步数
train_cfg.total_timesteps = 20_000_000
```

## 训练参数说明

### 核心超参数
- `learning_rate`: 学习率，默认7e-4
- `gamma`: 折扣因子，默认0.99
- `gae_lambda`: GAE参数，默认0.95
- `ppo_clip`: PPO截断范围，默认0.2
- `entropy_coef`: 熵正则化系数，默认0.01

### 训练规模
- `total_timesteps`: 总训练步数，默认1000万步
- `rollout_length`: 每次收集的轨迹长度，默认25步
- `num_envs`: 并行环境数量，默认16个
- `ppo_epoch`: 每次更新的PPO轮数，默认10轮

### 网络架构
- `actor_hidden_sizes`: Actor网络隐藏层大小，默认(64, 64)
- `critic_hidden_sizes`: Critic网络隐藏层大小，默认(64, 64)
- `activation`: 激活函数，默认"tanh"

## 实验结果

### 训练性能
- 总训练episode: 133,334
- 最终平均回报: -70.36
- 最佳单次回报: -12.87
- 整体平均回报: -68.47

### 策略表现
- 智能体学会了基本的协作策略
- 能够有效避免相互碰撞
- 实现了目标导向的移动行为
- 在有限空间内进行协调

## 项目特色

### 1. 模块化设计
- 清晰的代码结构，便于理解和修改
- 独立的配置文件，方便参数调整
- 可复用的网络组件

### 2. 完整的训练流程
- 从数据收集到模型更新的完整实现
- 详细的日志记录和可视化
- 自动化的模型保存和加载

### 3. 学术严谨性
- 遵循MAPPO论文的原始设计
- 使用标准的强化学习技术
- 包含详细的实验分析

## 扩展方向

### 算法改进
- 引入注意力机制
- 实现分层强化学习
- 添加优先级经验回放

### 任务扩展
- 支持更多MPE任务
- 扩展到连续动作空间
- 增加智能体数量

### 性能优化
- 多GPU训练支持
- 更高效的并行化
- 内存优化

## 参考文献

1. Yu, J., et al. "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." NeurIPS 2021.
2. Schulman, J., et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347 (2017).
3. Lowe, R., et al. "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments." NeurIPS 2017.

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

---

**注意**: 本项目基于PettingZoo的MPE环境，请确保正确安装了相关依赖包。

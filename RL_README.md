# IC2 核反应堆强化学习指南

这是 IC2 核反应堆模拟器的强化学习扩展，让 AI 学习设计最优的核反应堆配置。

## 项目目标

使用强化学习训练 AI 自动设计核反应堆，目标是：
- **最大化发电量**：追求更高的 EU/t 输出
- **避免爆炸**：确保反应堆温度不超过 10000 HU
- **稳定运行**：设计能够长期稳定工作的反应堆

## 强化学习环境设计

### 状态空间（Observation Space）
- 9×6 的网格，每个位置用整数表示组件类型
- 共 18 种组件类型（包括空槽位）

### 动作空间（Action Space）
- 选择位置 (row, col) 和要放置的组件类型
- 每步放置一个组件，最多 54 步（填满整个反应堆）

### 奖励函数（Reward Function）

**主要奖励：**
- 平均发电功率（EU/t）× 1.0

**惩罚机制：**
- 爆炸：-1000 分
- 温度过高（>90% 阈值）：-200 × (heat_ratio - 0.9)
- 温度较高（>70% 阈值）：-50 × (heat_ratio - 0.7)

**额外奖励：**
- 完整运行模拟周期：+50 分
- 温度控制良好（<50%）且高功率（>100 EU/t）：+100 分

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `gymnasium`: OpenAI Gym 的继任者
- `stable-baselines3`: 强化学习算法库
- `torch`: PyTorch 深度学习框架
- `tensorboard`: 训练可视化

### 2. 训练模型

使用默认参数训练（PPO 算法）：

```bash
python rl_train.py
```

自定义训练参数：

```bash
python rl_train.py \
    --algorithm PPO \
    --timesteps 200000 \
    --n-envs 8 \
    --learning-rate 3e-4 \
    --simulation-ticks 1000 \
    --device cuda
```

参数说明：
- `--algorithm`: 算法选择（PPO, A2C, DQN）
- `--timesteps`: 总训练步数
- `--n-envs`: 并行环境数量（越多越快，但占用更多内存）
- `--learning-rate`: 学习率
- `--simulation-ticks`: 每次评估运行的 tick 数
- `--device`: 训练设备（cpu, cuda, auto）

### 3. 监控训练过程

训练时会自动保存日志到 `rl_logs` 目录，使用 TensorBoard 查看：

```bash
tensorboard --logdir rl_logs
```

然后在浏览器打开 http://localhost:6006

### 4. 评估模型

评估训练好的模型：

```bash
python rl_evaluate.py rl_models/PPO_20240330_120000/best/best_model.zip
```

自定义评估参数：

```bash
python rl_evaluate.py rl_models/PPO_20240330_120000/best/best_model.zip \
    --episodes 20 \
    --ticks 5000 \
    --output-dir my_results
```

评估会生成：
- 最佳设计的可视化图表
- 反应堆配置 YAML 文件
- 性能统计报告

## 环境详解

### ReactorEnv（逐步设计环境）

这是主要的训练环境，AI 逐步放置组件来设计反应堆。

**特点：**
- 每个 episode 最多 54 步（9×6 网格）
- 每步选择一个位置和组件类型
- 中间步骤有小奖励引导学习
- 最后一步运行完整模拟并计算最终奖励

**使用示例：**

```python
from model.rl_env import ReactorEnv

env = ReactorEnv(
    rows=9,
    cols=6,
    max_hull_heat=10000,
    simulation_ticks=1000
)

obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"平均功率: {info['avg_power']} EU/t")
print(f"是否爆炸: {info['exploded']}")
```

### ReactorEnvSimplified（一次性设计环境）

简化版环境，一次性设计整个反应堆。

**特点：**
- 单步决策：一次性指定所有 54 个位置的组件
- 适合进化算法或遗传算法
- 评估速度更快

**使用示例：**

```python
from model.rl_env import ReactorEnvSimplified

env = ReactorEnvSimplified(simulation_ticks=1000)

obs, info = env.reset()
action = env.action_space.sample()  # 随机设计
obs, reward, done, truncated, info = env.step(action)

print(f"奖励: {reward}")
print(f"平均功率: {info['avg_power']} EU/t")
```

## 算法选择

### PPO (Proximal Policy Optimization)
- **推荐用于此项目**
- 稳定性好，样本效率高
- 适合连续决策问题
- 训练速度适中

### A2C (Advantage Actor-Critic)
- 训练速度快
- 适合简单环境
- 可能不如 PPO 稳定

### DQN (Deep Q-Network)
- 经典算法
- 适合离散动作空间
- 需要更多训练步数

## 训练技巧

### 1. 调整模拟时长

较短的模拟（500-1000 ticks）：
- 训练速度快
- 适合初期探索
- 可能无法充分评估稳定性

较长的模拟（2000-5000 ticks）：
- 更准确的评估
- 训练速度慢
- 适合后期优化

### 2. 并行环境数量

- 更多并行环境 = 更快训练
- 但会占用更多 CPU/内存
- 推荐：4-8 个环境

### 3. 学习率调整

- 默认 3e-4 适合大多数情况
- 如果训练不稳定，降低到 1e-4
- 如果收敛太慢，提高到 1e-3

### 4. 使用 GPU 加速

如果有 NVIDIA GPU：

```bash
python rl_train.py --device cuda
```

可以显著加速训练（特别是使用大型神经网络时）。

## 高级用法

### 自定义奖励函数

编辑 `model/rl_env.py` 中的 `_calculate_reward` 方法：

```python
def _calculate_reward(self, avg_power, max_heat, exploded, ticks_run):
    reward = 0.0

    if exploded:
        reward = -1000.0
    else:
        # 自定义奖励逻辑
        reward = avg_power * 2.0  # 更重视发电量

        # 添加其他考虑因素
        # ...

    return reward
```

### 使用预训练模型继续训练

```python
from stable_baselines3 import PPO
from model.rl_env import ReactorEnv

env = ReactorEnv()
model = PPO.load("rl_models/PPO_xxx/best/best_model.zip", env=env)

# 继续训练
model.learn(total_timesteps=100000)
model.save("rl_models/continued_model")
```

### 集成到现有模拟器

训练好的模型可以导出为 YAML 配置，直接用于原有的模拟系统：

```bash
python rl_evaluate.py model.zip --episodes 1
# 生成 rl_results/best_design_config.yaml

# 使用原有模拟器运行
python examples/basic_simulation.py --config rl_results/best_design_config.yaml
```

## 项目结构

```
ic2-nuclear-reactor-proof/
├── model/
│   ├── rl_env.py              # 强化学习环境定义
│   ├── reactor.py             # 反应堆核心逻辑
│   ├── components.py          # 组件定义
│   └── ...
├── rl_train.py                # 训练脚本
├── rl_evaluate.py             # 评估脚本
├── rl_models/                 # 训练好的模型（自动创建）
├── rl_logs/                   # TensorBoard 日志（自动创建）
├── rl_results/                # 评估结果（自动创建）
└── requirements.txt           # 依赖列表
```

## 常见问题

### Q: 训练很慢怎么办？

A: 尝试以下方法：
1. 减少 `simulation_ticks`（如 500）
2. 增加并行环境数量 `--n-envs 8`
3. 使用 GPU `--device cuda`
4. 减少总训练步数先快速验证

### Q: 模型总是设计爆炸的反应堆？

A: 可能的原因：
1. 爆炸惩罚不够大，增加惩罚权重
2. 训练步数不够，继续训练
3. 学习率太高，降低学习率
4. 尝试不同的算法（如 PPO）

### Q: 如何提高发电量？

A: 调整奖励函数：
1. 增加发电量的奖励权重
2. 减少温度惩罚（但要小心爆炸）
3. 添加效率奖励（高功率 + 低温度）

### Q: 可以用其他强化学习库吗？

A: 可以，环境遵循 Gymnasium 标准接口，兼容：
- Ray RLlib
- TensorFlow Agents
- Keras-RL
- 等其他库

## 性能基准

在默认配置下（1000 ticks 模拟）：

| 算法 | 训练步数 | 平均功率 | 爆炸率 | 训练时间 |
|------|---------|---------|--------|---------|
| PPO  | 100k    | ~80 EU/t | ~20%  | ~30 分钟 |
| PPO  | 500k    | ~150 EU/t | ~5%  | ~2.5 小时 |
| A2C  | 100k    | ~60 EU/t | ~30%  | ~20 分钟 |

*注：实际性能取决于硬件配置和超参数设置*

## 进一步优化

### 1. 课程学习（Curriculum Learning）

逐步增加难度：
- 开始：短时间模拟（500 ticks）
- 中期：中等时间（1500 ticks）
- 后期：长时间（3000 ticks）

### 2. 奖励塑形（Reward Shaping）

添加中间奖励引导学习：
- 放置燃料棒附近有散热片：+小奖励
- 形成对称结构：+小奖励
- 避免孤立组件：+小奖励

### 3. 迁移学习

使用简化环境预训练，然后迁移到完整环境。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

与主项目相同

## 参考资料

- [Stable-Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [IC2 官方 Wiki](https://wiki.industrial-craft.net/)

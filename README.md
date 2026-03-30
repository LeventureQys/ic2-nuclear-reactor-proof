# IC2 核反应堆模拟器 ⚛️

这是一个用于模拟 Minecraft 工业2（IC2）核反应堆运行的完整系统。如果你曾经在游戏中因为反应堆设计不当而导致爆炸，或者想在实际建造前验证设计的可行性，这个工具可以帮到你。

**🆕 现已支持强化学习！** 让 AI 自动学习设计最优的核反应堆配置。详见 [强化学习扩展](#强化学习扩展-)

## 项目特色

- ✅ **完整物理模拟**：实现 IC2 核反应堆的所有核心机制
- 📊 **实时可视化**：发电曲线、温度监控、热量分布一目了然
- ⚙️ **灵活配置**：YAML 配置文件，无需修改代码
- 📈 **详细分析**：自动生成报告和 CSV 数据导出
- 🤖 **AI 设计**：强化学习自动优化反应堆配置

## 项目背景

在 IC2 中设计一个高效且安全的核反应堆并不容易。你需要平衡发电效率和散热系统，稍有不慎就会导致反应堆过热爆炸。这个模拟器可以让你在不消耗游戏资源的情况下，测试各种反应堆配置，找出最优方案。

## 快速开始

### 基础模拟

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **运行第一个模拟**
```bash
python examples/basic_simulation.py
```

输出文件（在 `output/` 目录）：
- `reactor_analysis.png` - 综合分析图表
- `simulation_report.txt` - 详细文本报告
- `simulation_data.csv` - 原始数据

3. **实时可视化**
```bash
python examples/realtime_simulation.py
```

4. **对比多个设计**
```bash
python examples/comparison_analysis.py
```

### 强化学习（AI 自动设计）

1. **测试环境**
```bash
python test_rl_env.py
```

2. **训练 AI**
```bash
python rl_train.py --algorithm PPO --timesteps 100000
```

3. **监控训练**
```bash
tensorboard --logdir rl_logs
```

4. **评估模型**
```bash
python rl_evaluate.py rl_models/PPO_xxx/best/best_model.zip
```

详细文档：[RL_README.md](RL_README.md)

## 项目结构

```
ic2-nuclear-reactor-proof/
├── config/                      # 配置文件目录
│   ├── reactor_config.yaml      # 标准反应堆配置
│   └── high_power_reactor.yaml  # 高功率反应堆配置
├── model/                       # 核心模型
│   ├── components.py            # 组件定义（燃料棒、散热片等）
│   ├── reactor.py               # 反应堆主类
│   ├── simulation.py            # 模拟引擎
│   ├── visualization.py         # 可视化系统
│   └── rl_env.py                # 强化学习环境 🆕
├── examples/                    # 示例脚本
│   ├── basic_simulation.py      # 基础模拟示例
│   ├── realtime_simulation.py   # 实时可视化示例
│   └── comparison_analysis.py   # 配置对比示例
├── documents/                   # 文档
│   └── IC2_Reactor_Info.md      # IC2 反应堆机制说明
├── rl_train.py                  # 强化学习训练脚本 🆕
├── rl_evaluate.py               # 强化学习评估脚本 🆕
├── test_rl_env.py               # 环境测试脚本 🆕
├── output/                      # 输出目录（自动创建）
├── rl_models/                   # 训练模型目录（自动创建）🆕
├── rl_logs/                     # 训练日志目录（自动创建）🆕
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
└── RL_README.md                 # 强化学习详细文档 🆕
```

## 配置文件详解

配置文件使用 YAML 格式，结构清晰易懂。

下面是一个典型的配置文件示例：

```yaml
# 给你的反应堆起个名字
reactor_name: "示例反应堆"

# 反应堆布局，9行6列，每个位置用代码表示组件类型
layout:
  - ["E", "E", "E", "E", "E", "E"]
  - ["E", "U", "H", "U", "H", "E"]
  - ["E", "H", "C60", "C60", "H", "E"]
  # ... 继续填写剩余的行

# 模拟参数
simulation:
  duration: 10000          # 模拟多少个 tick（1 tick = 1/20 秒）
  time_step: 1             # 时间步长，通常保持为 1
  speed_multiplier: 1.0    # 实时模拟的速度倍率，2.0 表示两倍速
  sample_interval: 20      # 每隔多少 tick 采样一次数据

# 可视化参数
visualization:
  enable_realtime: true    # 是否启用实时可视化
  update_interval: 100     # 图表更新间隔
  charts:                  # 要显示的图表类型
    - power_output
    - heat_distribution
    - reactor_heat
```

### 组件代码对照表

| 代码 | 组件名称 | 说明 |
|------|---------|------|
| E | 空格 | 空槽位 |
| **燃料棒** |||
| U | 单铀棒 | 基础燃料，发电 5 EU/t |
| D | 双联燃料棒 | 2倍效率 |
| Q | 四联燃料棒 | 4倍效率 |
| **散热片** |||
| H | 散热片 | 基础散热 6 HU/tick |
| R | 反应堆散热片 | 吸收堆温 5 HU/tick |
| C | 元件散热片 | 散发相邻元件热量 |
| A | 高级散热片 | 高效散热 12 HU/tick |
| O | 超频散热片 | 最强散热 20 HU/tick |
| **热交换器** |||
| HE | 热交换器 | 热量交换 |
| RH | 反应堆热交换器 | 与堆温交换 |
| CH | 元件热交换器 | 元件间交换 |
| AH | 高级热交换器 | 高效交换 |
| **其他** |||
| N | 中子反射板 | 增加发电效率 |
| TN | 加厚中子反射板 | 高耐久反射板 |
| C10 | 10k冷却单元 | 热量缓冲 10000 HU |
| C30 | 30k冷却单元 | 热量缓冲 30000 HU |
| C60 | 60k冷却单元 | 热量缓冲 60000 HU |

## 核心机制

详细机制说明请查看 [IC2_Reactor_Info.md](documents/IC2_Reactor_Info.md)

### 核脉冲机制
燃料棒工作时向四周发射核脉冲，接收到的核脉冲越多，发电量越高。中子反射板可以反射核脉冲，提升发电效率。

### 热量产生
燃料棒产生的热量取决于相邻的燃料棒或反射板数量：
```
热量 = 倍数 × (n+1) × (n+2)
```
其中 n 是相邻的燃料棒或反射板数量（0-4）

### 热量传递
- 燃料棒产生的热量均分给周围可储热组件
- 热交换器在组件间传递热量（高温→低温）
- 无法接受的热量传递给反应堆本体

### 散热系统
散热片将热量散发到环境中：
- 普通散热片：6 HU/tick
- 高级散热片：12 HU/tick
- 超频散热片：20 HU/tick

**关键：散热能力 ≥ 产热量，否则反应堆会持续升温直至爆炸（10000 HU）**

## 使用示例

### 方式一：使用配置文件

创建 `my_reactor.yaml`：
```yaml
reactor_name: "我的反应堆"
layout:
  - ["E", "E", "E", "E", "E", "E"]
  - ["E", "U", "N", "U", "N", "E"]
  - ["E", "O", "O", "O", "O", "E"]
  # ... 继续设计
```

运行模拟：
```bash
python examples/basic_simulation.py --config my_reactor.yaml
```

### 方式二：使用代码

```python
from model import Reactor, UraniumCell, HeatVent

# 创建反应堆
reactor = Reactor(rows=9, cols=6)

# 放置组件
reactor.grid[1][1] = UraniumCell((1, 1))
reactor.grid[1][2] = HeatVent((1, 2))

# 运行模拟
for tick in range(1000):
    result = reactor.simulate_tick()
    print(f"Tick {tick}: 功率={result['power']} EU/t")
```

### 方式三：使用 AI 自动设计

```bash
# 训练 AI
python rl_train.py --timesteps 100000

# 评估并导出最佳设计
python rl_evaluate.py rl_models/PPO_xxx/best/best_model.zip

# 使用 AI 设计的配置
python examples/basic_simulation.py --config rl_results/best_design_config.yaml
```

## 输出结果

### 可视化图表
- **发电功率曲线**：显示功率随时间变化，包含平均值
- **堆温曲线**：监控反应堆温度，标注爆炸阈值
- **热量分布图**：热力图显示各组件温度百分比
- **累计发电量**：总发电量随时间增长

### 文本报告
- 性能统计：总发电量、平均功率、最大功率
- 热量统计：最大堆温、最终堆温、是否爆炸
- 组件状态：燃料棒耐久、储热组件热量

### CSV 数据
逐 tick 记录：当前 tick、瞬时功率、堆温、累计发电量、爆炸状态

## 常见问题

### Q: 反应堆总是爆炸怎么办？
**A:** 散热系统不足
- 查看热量分布图，找出高温区域
- 在高温区域增加散热片
- 减少燃料棒数量或密度
- 使用热交换器转移热量

### Q: 发电量太低怎么办？
**A:** 优化燃料棒布局
- 增加燃料棒数量
- 让燃料棒紧密排列（增加核脉冲交互）
- 使用中子反射板提升效率
- 使用双联或四联燃料棒

### Q: 中子反射板很快损坏？
**A:** 耐久消耗过快
- 使用加厚中子反射板（4倍耐久）
- 减少反射板周围的燃料棒数量
- 避免四联燃料棒紧邻反射板

### Q: 如何提高训练效果？
**A:** 调整强化学习参数
- 增加训练步数（如 500k）
- 使用更多并行环境（8-16个）
- 调整奖励函数权重
- 尝试不同算法（PPO 通常最稳定）

## 项目贡献

如果你发现了 bug，或者有改进建议，欢迎提交 Issue。如果你想贡献代码，也欢迎提交 Pull Request。

## 参考资料

- IC2 官方 Wiki：https://wiki.industrial-craft.net/
- documents/IC2_Reactor_Info.md：本项目整理的详细机制说明文档

## 强化学习扩展 🤖

本项目支持使用强化学习训练 AI 自动设计最优核反应堆配置。

### 核心特性

- **自动优化**：AI 学习如何放置组件以最大化发电量
- **安全保障**：智能控制温度，避免反应堆爆炸
- **多种算法**：支持 PPO、A2C、DQN 等主流算法
- **实时监控**：TensorBoard 可视化训练过程
- **配置导出**：将 AI 设计导出为 YAML 文件

### 快速开始

```bash
# 1. 测试环境
python test_rl_env.py

# 2. 训练 AI（推荐使用 PPO）
python rl_train.py --algorithm PPO --timesteps 100000

# 3. 监控训练过程
tensorboard --logdir rl_logs

# 4. 评估最佳模型
python rl_evaluate.py rl_models/PPO_xxx/best/best_model.zip
```

### 训练参数

```bash
python rl_train.py \
    --algorithm PPO \           # 算法：PPO, A2C, DQN
    --timesteps 500000 \        # 训练步数
    --n-envs 8 \                # 并行环境数
    --learning-rate 3e-4 \      # 学习率
    --simulation-ticks 2000 \   # 每次评估的 tick 数
    --device cuda               # 使用 GPU 加速
```

### 奖励机制

- **主要奖励**：平均发电功率（EU/t）
- **爆炸惩罚**：-1000 分
- **温度惩罚**：温度越高惩罚越大
- **稳定奖励**：完整运行 +50 分
- **效率奖励**：低温高功率 +100 分

详细文档：[RL_README.md](RL_README.md)

## 注意事项

这个模拟器是基于 IC2 经典版本的机制实现的。不同的 IC2 版本（比如 IC2 Experimental）可能有不同的机制，使用前请确认你的游戏版本。

如果模拟结果和游戏中的实际表现有差异，可能是因为：
- 版本差异
- 模拟器实现的细节与游戏不完全一致
- 游戏中有其他 mod 影响了反应堆的行为

遇到这种情况，建议以游戏实际表现为准，并欢迎反馈问题帮助改进模拟器。

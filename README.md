# IC2 核反应堆模拟器

一个完整的 Minecraft 工业2（IC2）核反应堆模拟系统，支持实时可视化、性能分析和配置对比。

## 功能特性

- ✅ **完整的物理模拟**：精确模拟 IC2 核反应堆的所有机制
  - 核脉冲传递和发电
  - 热量产生和传递
  - 组件交互和散热
  - 反应堆爆炸检测

- 📊 **实时可视化**：
  - 发电功率曲线
  - 反应堆温度曲线
  - 热量分布热力图
  - 组件状态监控

- ⚙️ **灵活配置**：
  - YAML 配置文件驱动
  - 支持 54 格完整布局（9×6）
  - 可调节模拟速度
  - 多种预设配置

- 📈 **数据分析**：
  - 自动生成分析报告
  - CSV 数据导出
  - 多配置对比分析

## 项目结构

```
ic2-nuclear-reactor-proof/
├── config/                      # 配置文件目录
│   ├── reactor_config.yaml      # 标准反应堆配置
│   └── high_power_reactor.yaml  # 高功率反应堆配置
├── model/                       # 核心模型
│   ├── __init__.py
│   ├── components.py            # 组件定义（燃料棒、散热片等）
│   ├── reactor.py               # 反应堆主类
│   ├── simulation.py            # 模拟引擎
│   └── visualization.py         # 可视化系统
├── examples/                    # 示例脚本
│   ├── basic_simulation.py      # 基础模拟示例
│   ├── realtime_simulation.py   # 实时可视化示例
│   └── comparison_analysis.py   # 配置对比示例
├── documents/                   # 文档
│   └── IC2_Reactor_Info.md      # IC2 反应堆机制说明
├── output/                      # 输出目录（自动创建）
├── requirements.txt             # Python 依赖
└── README.md                    # 本文件
```

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd ic2-nuclear-reactor-proof
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础模拟

运行基础模拟，生成静态分析图表：

```bash
python examples/basic_simulation.py
```

输出文件：
- `output/reactor_analysis.png` - 综合分析图表
- `output/simulation_report.txt` - 文本报告
- `output/simulation_data.csv` - 原始数据

### 2. 实时可视化

运行实时模拟，查看动态变化：

```bash
python examples/realtime_simulation.py
```

### 3. 配置对比

对比不同配置的性能：

```bash
python examples/comparison_analysis.py
```

## 配置文件说明

配置文件使用 YAML 格式，示例：

```yaml
# 反应堆名称
reactor_name: "示例反应堆"

# 反应堆布局（9行6列）
layout:
  - ["E", "E", "E", "E", "E", "E"]
  - ["E", "U", "H", "U", "H", "E"]
  - ["E", "H", "C60", "C60", "H", "E"]
  # ... 更多行

# 模拟参数
simulation:
  duration: 10000          # 模拟时长（tick）
  time_step: 1             # 时间步长
  speed_multiplier: 1.0    # 速度倍率
  sample_interval: 20      # 采样间隔

# 可视化参数
visualization:
  enable_realtime: true
  update_interval: 100
  charts:
    - power_output
    - heat_distribution
    - reactor_heat
```

### 组件代码对照表

| 代码 | 组件名称 | 说明 |
|------|---------|------|
| E | 空格 | 空槽位 |
| U | 单铀棒 | 基础燃料 |
| D | 双联燃料棒 | 2倍效率 |
| Q | 四联燃料棒 | 4倍效率 |
| H | 散热片 | 基础散热 |
| R | 反应堆散热片 | 吸收堆温 |
| C | 元件散热片 | 散发相邻元件热量 |
| A | 高级散热片 | 高效散热 |
| O | 超频散热片 | 最强散热 |
| HE | 热交换器 | 热量交换 |
| RH | 反应堆热交换器 | 与堆温交换 |
| CH | 元件热交换器 | 元件间交换 |
| AH | 高级热交换器 | 高效交换 |
| N | 中子反射板 | 增加发电 |
| TN | 加厚中子反射板 | 高耐久反射板 |
| C10 | 10k冷却单元 | 热量缓冲 |
| C30 | 30k冷却单元 | 中型缓冲 |
| C60 | 60k冷却单元 | 大型缓冲 |

## 核心机制

### 1. 核脉冲机制
- 燃料棒向四周发射核脉冲
- 接收的核脉冲越多，发电量越高
- 中子反射板可以反射核脉冲

### 2. 热量产生
- 燃料棒根据相邻燃料棒/反射板数量产生热量
- 公式：`热量 = 倍数 × (n+1) × (n+2)`

### 3. 热量传递
- 燃料棒热量均分给周围可储热元件
- 热交换器在元件间传递热量
- 无法接受的热量传给反应堆本体

### 4. 散热系统
- 散热片将热量散发到环境
- 不同散热片有不同的散热速率
- 必须保证散热 ≥ 产热才能稳定运行

## 使用示例

### 自定义配置

创建新的配置文件 `my_reactor.yaml`：

```yaml
reactor_name: "我的反应堆"
layout:
  - ["E", "E", "E", "E", "E", "E"]
  - ["E", "U", "N", "U", "N", "E"]
  - ["E", "O", "O", "O", "O", "E"]
  # ... 继续设计
```

运行模拟：

```python
from model import SimulationEngine, ReactorVisualizer

engine = SimulationEngine("my_reactor.yaml")
engine.run_fast()

visualizer = ReactorVisualizer(engine.get_config())
visualizer.create_static_plots(engine.get_history(), engine.get_reactor())
```

### 编程接口

```python
from model import Reactor, UraniumCell, HeatVent

# 创建反应堆
reactor = Reactor(rows=9, cols=6)

# 手动放置组件
reactor.grid[1][1] = UraniumCell((1, 1))
reactor.grid[1][2] = HeatVent((1, 2))

# 运行模拟
for tick in range(1000):
    result = reactor.simulate_tick()
    print(f"Tick {tick}: Power={result['power']} EU/t")
```

## 输出说明

### 1. 分析图表
- 发电功率曲线：显示每个时刻的发电量
- 堆温曲线：监控反应堆温度变化
- 热量分布图：显示各组件的热量百分比
- 反应堆布局：可视化组件配置

### 2. 文本报告
包含：
- 性能统计（总发电量、平均功率等）
- 热量统计（最大堆温、是否爆炸等）
- 组件状态（燃料耐久、组件热量等）

### 3. CSV 数据
逐tick记录：
- 发电功率
- 堆温
- 总发电量
- 爆炸状态

## 高级功能

### 速度控制

```python
engine = SimulationEngine("config.yaml")
engine.set_speed(2.0)  # 2倍速
engine.run_realtime()
```

### 交互式控制

```python
from model import InteractiveSimulation

interactive = InteractiveSimulation(engine)
interactive.process_command("pause")
interactive.process_command("speed 0.5")
interactive.process_command("resume")
```

### 详细记录

```python
from model import SimulationRecorder

recorder = SimulationRecorder()
engine.register_tick_callback(recorder.record_tick)
engine.run_fast()
recorder.save_to_file("detailed_log.json")
```

## 性能优化建议

1. **快速模拟**：使用 `run_fast()` 而不是 `run_realtime()`
2. **减少采样**：增大 `sample_interval` 值
3. **缩短时长**：减小 `duration` 值进行测试

## 故障排查

### 反应堆爆炸
- 检查散热系统是否足够
- 查看热量分布图，找出热点
- 增加散热片或减少燃料棒

### 发电量低
- 增加燃料棒密度
- 使用中子反射板提升效率
- 检查燃料棒布局是否合理

### 组件损坏
- 中子反射板耐久耗尽：使用加厚版本
- 组件过热：增加散热或热交换器

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 参考资料

- [IC2 官方 Wiki](https://wiki.industrial-craft.net/)
- `documents/IC2_Reactor_Info.md` - 详细机制说明

---

**注意**：本模拟器基于 IC2 经典版本的机制，不同版本可能有差异。

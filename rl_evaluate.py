"""
IC2 核反应堆强化学习评估脚本

加载训练好的模型，评估其性能并可视化结果
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from model.rl_env import ReactorEnv
from model.reactor import Reactor
from model.visualization import ReactorVisualizer
import matplotlib.pyplot as plt


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    simulation_ticks: int = 2000,
    render: bool = True,
    save_results: bool = True,
    output_dir: str = "rl_results"
):
    """
    评估训练好的模型

    Args:
        model_path: 模型文件路径
        n_episodes: 评估的 episode 数量
        simulation_ticks: 每次模拟的 tick 数
        render: 是否渲染过程
        save_results: 是否保存结果
        output_dir: 结果保存目录
    """
    print("=" * 60)
    print("IC2 核反应堆 AI 评估")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"评估 episodes: {n_episodes}")
    print(f"模拟 ticks: {simulation_ticks}")
    print("=" * 60)

    # 加载模型
    print("\n加载模型...")
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        # 尝试自动检测
        try:
            model = PPO.load(model_path)
        except:
            try:
                model = A2C.load(model_path)
            except:
                model = DQN.load(model_path)

    print("模型加载成功")

    # 创建环境
    env = ReactorEnv(
        rows=9,
        cols=6,
        max_hull_heat=10000,
        simulation_ticks=simulation_ticks,
        render_mode="human" if render else None
    )

    # 评估
    results = []
    best_design = None
    best_power = 0

    print("\n开始评估...")

    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")

        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0

        while not (done or truncated):
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if render:
                env.render()

        # 记录结果
        result = {
            "episode": episode + 1,
            "reward": episode_reward,
            "avg_power": info.get("avg_power", 0),
            "max_hull_heat": info.get("max_hull_heat", 0),
            "exploded": info.get("exploded", False),
            "steps": steps
        }
        results.append(result)

        print(f"总奖励: {episode_reward:.2f}")
        print(f"平均功率: {result['avg_power']:.2f} EU/t")
        print(f"最大堆温: {result['max_hull_heat']:.2f}")
        print(f"是否爆炸: {'是' if result['exploded'] else '否'}")

        # 记录最佳设计
        if result['avg_power'] > best_power and not result['exploded']:
            best_power = result['avg_power']
            best_design = {
                "layout": env.current_layout.copy(),
                "result": result
            }

    # 统计结果
    print("\n" + "=" * 60)
    print("评估统计:")
    print("=" * 60)

    avg_reward = np.mean([r["reward"] for r in results])
    avg_power = np.mean([r["avg_power"] for r in results])
    max_power = np.max([r["avg_power"] for r in results])
    explosion_rate = np.mean([r["exploded"] for r in results]) * 100

    print(f"平均奖励: {avg_reward:.2f}")
    print(f"平均功率: {avg_power:.2f} EU/t")
    print(f"最大功率: {max_power:.2f} EU/t")
    print(f"爆炸率: {explosion_rate:.2f}%")

    # 保存结果
    if save_results and best_design is not None:
        os.makedirs(output_dir, exist_ok=True)

        # 保存最佳设计
        print(f"\n最佳设计 (功率: {best_power:.2f} EU/t):")
        print_layout(best_design["layout"], env.AVAILABLE_COMPONENTS)

        # 重新运行最佳设计以获取详细数据
        print("\n重新模拟最佳设计以生成可视化...")
        reactor, history = simulate_design(
            best_design["layout"],
            env.AVAILABLE_COMPONENTS,
            simulation_ticks
        )

        # 生成可视化
        visualize_results(reactor, history, output_dir)

        # 保存布局配置
        save_layout_config(
            best_design["layout"],
            env.AVAILABLE_COMPONENTS,
            output_dir,
            best_design["result"]
        )

    env.close()
    return results, best_design


def print_layout(layout: np.ndarray, component_list: list):
    """打印反应堆布局"""
    rows, cols = layout.shape
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            component_idx = layout[r, c]
            component_code = component_list[component_idx]
            row_str += f"{component_code:4s} "
        print(f"第{r+1}行: {row_str}")


def simulate_design(layout: np.ndarray, component_list: list, ticks: int):
    """重新模拟设计以获取详细数据"""
    rows, cols = layout.shape

    # 转换为代码列表
    layout_codes = []
    for r in range(rows):
        row_codes = []
        for c in range(cols):
            component_idx = layout[r, c]
            row_codes.append(component_list[component_idx])
        layout_codes.append(row_codes)

    # 创建反应堆
    reactor = Reactor(rows, cols, 10000)
    reactor.load_layout(layout_codes)

    # 运行模拟
    history = {
        "ticks": [],
        "power": [],
        "hull_heat": [],
        "hull_heat_percentage": [],
        "total_power": [],
        "exploded": []
    }

    for tick in range(ticks):
        result = reactor.simulate_tick()

        history["ticks"].append(result["tick"])
        history["power"].append(result["power"])
        history["hull_heat"].append(result["hull_heat"])
        history["hull_heat_percentage"].append(
            (result["hull_heat"] / reactor.max_hull_heat) * 100
        )
        history["total_power"].append(result["total_power"])
        history["exploded"].append(result["exploded"])

        if result["exploded"]:
            break

    return reactor, history


def visualize_results(reactor: Reactor, history: dict, output_dir: str):
    """生成可视化图表"""
    print("生成可视化图表...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 发电功率曲线
    ax1 = axes[0, 0]
    ax1.plot(history["ticks"], history["power"], 'b-', linewidth=2)
    ax1.set_xlabel("Tick")
    ax1.set_ylabel("功率 (EU/t)")
    ax1.set_title("发电功率曲线")
    ax1.grid(True, alpha=0.3)

    avg_power = np.mean(history["power"])
    ax1.axhline(y=avg_power, color='r', linestyle='--', label=f'平均: {avg_power:.2f} EU/t')
    ax1.legend()

    # 2. 堆温曲线
    ax2 = axes[0, 1]
    ax2.plot(history["ticks"], history["hull_heat"], 'r-', linewidth=2)
    ax2.axhline(y=reactor.max_hull_heat, color='k', linestyle='--', label='爆炸阈值')
    ax2.set_xlabel("Tick")
    ax2.set_ylabel("堆温 (HU)")
    ax2.set_title("反应堆温度曲线")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. 热量分布图
    ax3 = axes[1, 0]
    heat_map = reactor.get_heat_percentage_map()
    im = ax3.imshow(heat_map, cmap='hot', aspect='auto', vmin=0, vmax=100)
    ax3.set_xlabel("列")
    ax3.set_ylabel("行")
    ax3.set_title("组件热量分布 (%)")
    plt.colorbar(im, ax=ax3, label="热量百分比")

    # 4. 累计发电量
    ax4 = axes[1, 1]
    ax4.plot(history["ticks"], history["total_power"], 'g-', linewidth=2)
    ax4.set_xlabel("Tick")
    ax4.set_ylabel("累计发电量 (EU)")
    ax4.set_title("累计发电量")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, "best_design_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")

    plt.close()


def save_layout_config(
    layout: np.ndarray,
    component_list: list,
    output_dir: str,
    result: dict
):
    """保存布局配置为 YAML 文件"""
    import yaml

    rows, cols = layout.shape

    # 转换为代码列表
    layout_codes = []
    for r in range(rows):
        row_codes = []
        for c in range(cols):
            component_idx = layout[r, c]
            row_codes.append(component_list[component_idx])
        layout_codes.append(row_codes)

    # 创建配置
    config = {
        "reactor_name": f"AI设计反应堆 (功率: {result['avg_power']:.2f} EU/t)",
        "layout": layout_codes,
        "reactor_parameters": {
            "max_hull_heat": 10000
        },
        "simulation": {
            "duration": 10000,
            "time_step": 1,
            "speed_multiplier": 1.0,
            "sample_interval": 20
        },
        "visualization": {
            "enable_realtime": False,
            "update_interval": 100,
            "charts": ["power_output", "heat_distribution", "reactor_heat"]
        },
        "performance": {
            "avg_power": float(result['avg_power']),
            "max_hull_heat": float(result['max_hull_heat']),
            "exploded": bool(result['exploded'])
        }
    }

    # 保存
    output_path = os.path.join(output_dir, "best_design_config.yaml")
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"配置已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="评估 IC2 核反应堆 AI 模型")

    parser.add_argument(
        "model_path",
        type=str,
        help="模型文件路径"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="评估的 episode 数量"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=2000,
        help="每次模拟的 tick 数"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不渲染过程"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rl_results",
        help="结果保存目录"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        n_episodes=args.episodes,
        simulation_ticks=args.ticks,
        render=not args.no_render,
        save_results=True,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

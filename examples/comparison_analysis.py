#!/usr/bin/env python3
"""
IC2 核反应堆模拟器 - 对比分析示例

对比不同反应堆配置的性能
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimulationEngine, ReactorVisualizer, ReportGenerator
import matplotlib.pyplot as plt


def run_comparison():
    """运行对比分析"""
    print("=" * 80)
    print("IC2 核反应堆模拟器 - 对比分析")
    print("=" * 80)

    # 配置文件列表
    configs = [
        ("config/reactor_config.yaml", "标准配置"),
        ("config/high_power_reactor.yaml", "高功率配置")
    ]

    results = []

    # 运行每个配置
    for config_path, name in configs:
        print(f"\n{'=' * 80}")
        print(f"运行配置: {name}")
        print(f"配置文件: {config_path}")
        print(f"{'=' * 80}")

        engine = SimulationEngine(config_path)
        engine.run_fast()

        results.append({
            "name": name,
            "engine": engine,
            "history": engine.get_history(),
            "reactor": engine.get_reactor()
        })

    # 生成对比图表
    print("\n生成对比图表...")
    create_comparison_plots(results)

    # 生成对比报告
    print("\n生成对比报告...")
    generate_comparison_report(results)

    print("\n" + "=" * 80)
    print("对比分析完成！请查看 output 目录中的结果文件。")
    print("=" * 80)


def create_comparison_plots(results):
    """创建对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('反应堆配置对比分析', fontsize=16, fontweight='bold')

    # 1. 功率对比
    ax = axes[0, 0]
    for result in results:
        history = result["history"]
        ax.plot(history["ticks"], history["power"], linewidth=2, label=result["name"])
    ax.set_xlabel('时间 (tick)')
    ax.set_ylabel('功率 (EU/t)')
    ax.set_title('发电功率对比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 堆温对比
    ax = axes[0, 1]
    for result in results:
        history = result["history"]
        ax.plot(history["ticks"], history["hull_heat_percentage"],
               linewidth=2, label=result["name"])
    ax.set_xlabel('时间 (tick)')
    ax.set_ylabel('堆温百分比 (%)')
    ax.set_title('堆温对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='危险线')

    # 3. 平均功率对比（柱状图）
    ax = axes[1, 0]
    names = [r["name"] for r in results]
    avg_powers = []
    for result in results:
        reactor = result["reactor"]
        avg_power = reactor.total_power_output / reactor.current_tick if reactor.current_tick > 0 else 0
        avg_powers.append(avg_power)

    bars = ax.bar(names, avg_powers, color=['blue', 'orange'])
    ax.set_ylabel('平均功率 (EU/t)')
    ax.set_title('平均功率对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值
    for bar, power in zip(bars, avg_powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{power:.2f}',
               ha='center', va='bottom', fontweight='bold')

    # 4. 效率对比（总发电量）
    ax = axes[1, 1]
    total_powers = [r["reactor"].total_power_output for r in results]
    bars = ax.bar(names, total_powers, color=['green', 'red'])
    ax.set_ylabel('总发电量 (EU)')
    ax.set_title('总发电量对比')
    ax.grid(True, alpha=0.3, axis='y')

    # 在柱状图上添加数值
    for bar, power in zip(bars, total_powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{power:.0f}',
               ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/comparison_analysis.png', dpi=150, bbox_inches='tight')
    print("对比图表已保存到: output/comparison_analysis.png")
    plt.close()


def generate_comparison_report(results):
    """生成对比报告"""
    report_path = "output/comparison_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("IC2 核反应堆配置对比报告\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results, 1):
            reactor = result["reactor"]
            history = result["history"]
            name = result["name"]

            f.write(f"\n配置 {i}: {name}\n")
            f.write("-" * 80 + "\n")

            final_tick = reactor.current_tick
            avg_power = reactor.total_power_output / final_tick if final_tick > 0 else 0
            max_power = max(history['power']) if history['power'] else 0
            max_hull_heat = max(history['hull_heat']) if history['hull_heat'] else 0

            f.write(f"总tick数: {final_tick}\n")
            f.write(f"总发电量: {reactor.total_power_output:.2f} EU\n")
            f.write(f"平均功率: {avg_power:.2f} EU/t\n")
            f.write(f"最大功率: {max_power:.2f} EU/t\n")
            f.write(f"最大堆温: {max_hull_heat:.2f} HU ({(max_hull_heat / reactor.max_hull_heat * 100):.2f}%)\n")
            f.write(f"是否爆炸: {'是' if reactor.is_exploded else '否'}\n")

        # 综合对比
        f.write("\n" + "=" * 80 + "\n")
        f.write("综合对比\n")
        f.write("=" * 80 + "\n\n")

        # 找出最佳配置
        best_power_idx = max(range(len(results)),
                            key=lambda i: results[i]["reactor"].total_power_output)
        best_safety_idx = min(range(len(results)),
                             key=lambda i: max(results[i]["history"]["hull_heat_percentage"]))

        f.write(f"最高发电量配置: {results[best_power_idx]['name']}\n")
        f.write(f"最安全配置: {results[best_safety_idx]['name']}\n")

    print(f"对比报告已保存到: {report_path}")


if __name__ == "__main__":
    run_comparison()

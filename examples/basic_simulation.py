#!/usr/bin/env python3
"""
IC2 核反应堆模拟器 - 基础示例

演示如何使用模拟器进行基本的反应堆模拟
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimulationEngine, ReactorVisualizer, ReportGenerator


def main():
    """主函数"""
    print("=" * 80)
    print("IC2 核反应堆模拟器 - 基础示例")
    print("=" * 80)

    # 配置文件路径
    config_path = "config/reactor_config.yaml"

    # 创建模拟引擎
    print(f"\n加载配置: {config_path}")
    engine = SimulationEngine(config_path)

    # 运行模拟（快速模式）
    print("\n开始快速模拟...")
    engine.run_fast()

    # 获取结果
    history = engine.get_history()
    reactor = engine.get_reactor()

    # 生成可视化
    print("\n生成可视化图表...")
    visualizer = ReactorVisualizer(engine.get_config())
    visualizer.create_static_plots(history, reactor)

    # 生成报告
    print("\n生成模拟报告...")
    report_gen = ReportGenerator()
    report_gen.generate_text_report(engine)
    report_gen.generate_csv_data(history)

    print("\n" + "=" * 80)
    print("模拟完成！请查看 output 目录中的结果文件。")
    print("=" * 80)


if __name__ == "__main__":
    main()

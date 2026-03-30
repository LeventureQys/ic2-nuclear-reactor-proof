#!/usr/bin/env python3
"""
IC2 核反应堆模拟器 - 高级示例

演示实时可视化和交互式控制
"""

import sys
import os
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import SimulationEngine, ReactorVisualizer, ReportGenerator, InteractiveSimulation


def main():
    """主函数"""
    print("=" * 80)
    print("IC2 核反应堆模拟器 - 高级示例（实时可视化）")
    print("=" * 80)

    # 配置文件路径
    config_path = "config/reactor_config.yaml"

    # 创建模拟引擎
    print(f"\n加载配置: {config_path}")
    engine = SimulationEngine(config_path)

    # 创建可视化器
    visualizer = ReactorVisualizer(engine.get_config())

    # 设置实时可视化
    visualizer.create_realtime_visualization(engine)

    # 运行实时模拟
    print("\n开始实时模拟...")
    print("提示: 关闭图表窗口以停止模拟")

    try:
        engine.run_realtime()
    except KeyboardInterrupt:
        print("\n用户中断模拟")

    # 生成最终报告
    print("\n生成最终报告...")
    report_gen = ReportGenerator()
    report_gen.generate_text_report(engine)
    report_gen.generate_csv_data(engine.get_history())

    print("\n" + "=" * 80)
    print("模拟完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

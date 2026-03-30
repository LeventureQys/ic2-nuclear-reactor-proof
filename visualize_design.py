"""
可视化和复现评估出的反应堆设计
"""
import argparse
from stable_baselines3 import PPO
from model.rl_env import ReactorEnvV2
from model.reactor import Reactor
import numpy as np


def visualize_model_design(model_path):
    """可视化模型学到的反应堆设计"""
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建环境
    env = ReactorEnvV2(rows=9, cols=6, max_hull_heat=10000, simulation_ticks=1000)

    # 重置环境
    obs, _ = env.reset()

    # 收集所有动作
    actions = []
    done = False
    step = 0

    print("\n开始收集模型的动作序列...")
    while not done and step < 54:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)

        component_name = ReactorEnvV2.AVAILABLE_COMPONENTS[action]
        print(f"步骤 {step+1}: 动作={action} ({component_name})")

        obs, reward, done, truncated, info = env.step(action)
        step += 1

        if done or truncated:
            break

    # 打印最终布局
    print("\n" + "=" * 60)
    print("最终反应堆布局:")
    print("=" * 60)

    layout = []
    for r in range(9):
        row = []
        for c in range(6):
            idx = r * 6 + c
            if idx < len(actions):
                component_code = ReactorEnvV2.AVAILABLE_COMPONENTS[actions[idx]]
                row.append(component_code)
            else:
                row.append("E")
        layout.append(row)

    # 打印布局
    for r, row in enumerate(layout):
        row_str = " ".join(f"{comp:4s}" for comp in row)
        print(f"第{r+1}行: {row_str}")

    print("=" * 60)

    # 打印可复现的代码格式
    print("\n可复现的反应堆布局代码:")
    print("=" * 60)
    print("layout = [")
    for row in layout:
        print(f"    {row},")
    print("]")
    print("=" * 60)

    # 统计组件数量
    print("\n组件统计:")
    from collections import Counter
    all_components = [comp for row in layout for comp in row]
    component_counts = Counter(all_components)
    for comp, count in sorted(component_counts.items()):
        comp_name = {
            "E": "空槽位",
            "U": "单铀棒",
            "D": "双联燃料棒",
            "Q": "四联燃料棒",
            "H": "散热片",
            "R": "反应堆散热片",
            "C": "元件散热片",
            "A": "高级散热片",
            "O": "超频散热片",
            "HE": "热交换器",
            "RH": "反应堆热交换器",
            "CH": "元件热交换器",
            "AH": "高级热交换器",
            "N": "中子反射板",
            "TN": "加厚中子反射板",
            "C10": "10k冷却单元",
            "C30": "30k冷却单元",
            "C60": "60k冷却单元",
        }.get(comp, comp)
        print(f"  {comp} ({comp_name}): {count}个")

    # 模拟运行
    print("\n" + "=" * 60)
    print("模拟运行结果:")
    print("=" * 60)
    if done or truncated:
        avg_power = info.get('avg_power', 0)
        max_heat = info.get('max_hull_heat', 0)
        exploded = info.get('exploded', False)

        print(f"  平均功率: {avg_power:.2f} EU/t")
        print(f"  最大堆温: {max_heat:.2f}")
        print(f"  爆炸: {'是' if exploded else '否'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化模型学到的反应堆设计")
    parser.add_argument("model_path", type=str, help="模型路径")

    args = parser.parse_args()
    visualize_model_design(args.model_path)

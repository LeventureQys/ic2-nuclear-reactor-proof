"""
快速测试强化学习环境

验证环境是否正常工作
"""

from model.rl_env import ReactorEnv, ReactorEnvSimplified


def test_reactor_env():
    """测试 ReactorEnv 环境"""
    print("=" * 60)
    print("测试 ReactorEnv（逐步设计环境）")
    print("=" * 60)

    env = ReactorEnv(
        rows=9,
        cols=6,
        max_hull_heat=10000,
        simulation_ticks=500,
        render_mode="human"
    )

    print("\n环境信息:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  可用组件: {len(env.AVAILABLE_COMPONENTS)} 种")

    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观察形状: {obs.shape}")

    # 随机放置几个组件
    print("\n随机放置 10 个组件...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"步骤 {i+1}: 奖励={reward:.2f}, 已放置={info['components_placed']}")

        if terminated or truncated:
            print(f"\nEpisode 结束!")
            print(f"  平均功率: {info['avg_power']:.2f} EU/t")
            print(f"  最大堆温: {info['max_hull_heat']:.2f}")
            print(f"  是否爆炸: {'是' if info['exploded'] else '否'}")
            break

    env.close()
    print("\n✓ ReactorEnv 测试通过")


def test_reactor_env_simplified():
    """测试 ReactorEnvSimplified 环境"""
    print("\n" + "=" * 60)
    print("测试 ReactorEnvSimplified（一次性设计环境）")
    print("=" * 60)

    env = ReactorEnvSimplified(
        rows=9,
        cols=6,
        max_hull_heat=10000,
        simulation_ticks=500
    )

    print("\n环境信息:")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    # 重置环境
    obs, info = env.reset()

    # 随机设计一个反应堆
    print("\n生成随机反应堆设计...")
    action = env.action_space.sample()

    obs, reward, done, truncated, info = env.step(action)

    print(f"\n结果:")
    print(f"  奖励: {reward:.2f}")
    print(f"  平均功率: {info['avg_power']:.2f} EU/t")
    print(f"  总发电量: {info['total_power']:.2f} EU")
    print(f"  最大堆温: {info['max_hull_heat']:.2f}")
    print(f"  运行 ticks: {info['ticks_run']}")
    print(f"  是否爆炸: {'是' if info['exploded'] else '否'}")

    env.close()
    print("\n✓ ReactorEnvSimplified 测试通过")


def test_specific_design():
    """测试一个具体的反应堆设计"""
    print("\n" + "=" * 60)
    print("测试具体设计：简单的单铀棒反应堆")
    print("=" * 60)

    env = ReactorEnv(simulation_ticks=1000)

    obs, info = env.reset()

    # 设计一个简单的反应堆：中心放单铀棒，周围放散热片
    design = [
        (4, 2, env.AVAILABLE_COMPONENTS.index("U")),  # 中心放单铀棒
        (3, 2, env.AVAILABLE_COMPONENTS.index("H")),  # 上方散热片
        (5, 2, env.AVAILABLE_COMPONENTS.index("H")),  # 下方散热片
        (4, 1, env.AVAILABLE_COMPONENTS.index("H")),  # 左方散热片
        (4, 3, env.AVAILABLE_COMPONENTS.index("H")),  # 右方散热片
    ]

    print("\n放置组件:")
    for row, col, comp_idx in design:
        comp_name = env.AVAILABLE_COMPONENTS[comp_idx]
        print(f"  位置 ({row}, {col}): {comp_name}")
        action = [row, col, comp_idx]
        obs, reward, terminated, truncated, info = env.step(action)

    # 填充剩余位置为空
    print("\n填充剩余位置...")
    while not (terminated or truncated):
        # 找一个空位置
        for r in range(env.rows):
            for c in range(env.cols):
                if env.current_layout[r, c] == 0:
                    action = [r, c, 0]  # 放置空槽位
                    obs, reward, terminated, truncated, info = env.step(action)
                    break
            if terminated or truncated:
                break

    print("\n最终结果:")
    print(f"  平均功率: {info['avg_power']:.2f} EU/t")
    print(f"  最大堆温: {info['max_hull_heat']:.2f}")
    print(f"  是否爆炸: {'是' if info['exploded'] else '否'}")

    env.render()
    env.close()

    print("\n✓ 具体设计测试通过")


if __name__ == "__main__":
    print("\nIC2 核反应堆强化学习环境测试\n")

    try:
        test_reactor_env()
        test_reactor_env_simplified()
        test_specific_design()

        print("\n" + "=" * 60)
        print("所有测试通过！环境工作正常。")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 运行训练: python rl_train.py")
        print("  2. 查看日志: tensorboard --logdir rl_logs")
        print("  3. 评估模型: python rl_evaluate.py <model_path>")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

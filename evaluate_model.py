"""
评估训练好的模型 (V2环境)
"""
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from model.rl_env import ReactorEnv
import numpy as np


def make_env(simulation_ticks=1000):
    env = ReactorEnv(rows=9, cols=6, max_hull_heat=10000, simulation_ticks=simulation_ticks)
    return Monitor(env)


def evaluate_model(model_path, n_episodes=10, simulation_ticks=1000):
    """评估训练好的反应堆模型"""
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)
    print("✓ 模型加载成功")

    print(f"\n创建评估环境")
    env = DummyVecEnv([lambda: make_env(simulation_ticks)])
    print("✓ 环境创建成功")

    episode_rewards = []
    episode_powers = []
    episode_explosions = []

    print(f"\n开始评估 {n_episodes} 个episodes...")
    print("=" * 60)

    for episode in range(n_episodes):
        print(f"\n[Episode {episode+1}/{n_episodes}] 开始...")
        obs = env.reset()
        done = [False]
        episode_reward = 0
        step = 0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)

            # 调试：打印前几个动作
            if step < 5:
                component_name = ReactorEnv.AVAILABLE_COMPONENTS[action[0]]
                print(f"  步骤 {step+1}: 动作={action[0]} ({component_name})")

            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step += 1

            if done[0]:
                avg_power = info[0].get('avg_power', 0)
                max_heat = info[0].get('max_hull_heat', 0)
                exploded = info[0].get('exploded', False)

                episode_rewards.append(episode_reward)
                episode_powers.append(avg_power)
                episode_explosions.append(exploded)

                print(f"\n[Episode {episode+1}/{n_episodes}] 完成:")
                print(f"  奖励: {episode_reward:.2f}")
                print(f"  平均功率: {avg_power:.2f} EU/t")
                print(f"  最大堆温: {max_heat:.2f}")
                print(f"  爆炸: {'是' if exploded else '否'}")
                print(f"  步数: {step}")
                break

    env.close()

    # 统计
    print("\n" + "=" * 60)
    print("评估统计:")

    if len(episode_rewards) == 0:
        print("错误: 没有完成任何episode！")
        return

    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  平均功率: {np.mean(episode_powers):.2f} ± {np.std(episode_powers):.2f} EU/t")
    print(f"  最大功率: {np.max(episode_powers):.2f} EU/t")
    print(f"  爆炸率: {np.mean(episode_explosions) * 100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估训练好的反应堆模型")
    parser.add_argument("model_path", type=str, help="模型路径")
    parser.add_argument("--n-episodes", type=int, default=10, help="评估episodes数量")
    parser.add_argument("--simulation-ticks", type=int, default=1000, help="模拟tick数")

    args = parser.parse_args()
    evaluate_model(args.model_path, args.n_episodes, args.simulation_ticks)

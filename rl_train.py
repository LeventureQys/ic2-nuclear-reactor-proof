"""
IC2 核反应堆强化学习训练脚本

使用 Stable-Baselines3 训练 AI 设计最优反应堆
"""

import os
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from model.rl_env import ReactorEnv


class ReactorTrainingCallback(BaseCallback):
    """自定义训练回调，用于记录训练过程"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_powers = []
        self.episode_explosions = []

    def _on_step(self) -> bool:
        # 检查是否有 episode 结束
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    # 获取 info
                    info = self.locals["infos"][i]
                    if "avg_power" in info:
                        self.episode_powers.append(info["avg_power"])
                        self.episode_explosions.append(info["exploded"])

                        if self.verbose > 0:
                            print(f"\nEpisode 完成:")
                            print(f"  平均功率: {info['avg_power']:.2f} EU/t")
                            print(f"  最大堆温: {info['max_hull_heat']:.2f}")
                            print(f"  是否爆炸: {'是' if info['exploded'] else '否'}")

        return True

    def _on_training_end(self) -> None:
        """训练结束时的统计"""
        if len(self.episode_powers) > 0:
            print("\n" + "=" * 60)
            print("训练统计:")
            print(f"  总 episodes: {len(self.episode_powers)}")
            print(f"  平均功率: {np.mean(self.episode_powers):.2f} EU/t")
            print(f"  最大功率: {np.max(self.episode_powers):.2f} EU/t")
            print(f"  爆炸率: {np.mean(self.episode_explosions) * 100:.2f}%")
            print("=" * 60)


def train_reactor_agent(
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    save_dir: str = "rl_models",
    log_dir: str = "rl_logs",
    eval_freq: int = 5000,
    save_freq: int = 10000,
    simulation_ticks: int = 1000,
    device: str = "auto"
):
    """
    训练反应堆设计 AI

    Args:
        algorithm: 使用的算法 (PPO, A2C, DQN)
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        learning_rate: 学习率
        save_dir: 模型保存目录
        log_dir: 日志目录
        eval_freq: 评估频率
        save_freq: 保存频率
        simulation_ticks: 每次评估的模拟 tick 数
        device: 训练设备 (cpu, cuda, auto)
    """
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{algorithm}_{timestamp}"

    print("=" * 60)
    print("IC2 核反应堆强化学习训练")
    print("=" * 60)
    print(f"算法: {algorithm}")
    print(f"总步数: {total_timesteps}")
    print(f"并行环境: {n_envs}")
    print(f"学习率: {learning_rate}")
    print(f"模拟 ticks: {simulation_ticks}")
    print(f"设备: {device}")
    print(f"模型名称: {model_name}")
    print("=" * 60)

    # 创建环境
    def make_env():
        env = ReactorEnv(
            rows=9,
            cols=6,
            max_hull_heat=10000,
            simulation_ticks=simulation_ticks,
            render_mode=None
        )
        env = Monitor(env)
        return env

    # 创建向量化环境
    if n_envs > 1:
        # 多进程环境（更快但占用更多内存）
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env])

    # 创建评估环境
    eval_env = DummyVecEnv([make_env])

    # 创建模型
    print("\n创建模型...")

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
    elif algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}")

    # 创建回调
    callbacks = []

    # 训练回调
    training_callback = ReactorTrainingCallback(verbose=1)
    callbacks.append(training_callback)

    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, model_name, "best"),
        log_path=os.path.join(log_dir, model_name),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)

    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=os.path.join(save_dir, model_name, "checkpoints"),
        name_prefix="reactor_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # 开始训练
    print("\n开始训练...")
    print("提示: 可以使用 TensorBoard 查看训练过程:")
    print(f"  tensorboard --logdir {log_dir}")
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n训练被中断")

    # 保存最终模型
    final_model_path = os.path.join(save_dir, model_name, "final_model")
    model.save(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

    # 清理
    env.close()
    eval_env.close()

    return model, model_name


def main():
    parser = argparse.ArgumentParser(description="训练 IC2 核反应堆设计 AI")

    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C", "DQN"],
        help="强化学习算法"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="总训练步数"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="并行环境数量"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率"
    )
    parser.add_argument(
        "--simulation-ticks",
        type=int,
        default=1000,
        help="每次评估的模拟 tick 数"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="训练设备"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="rl_models",
        help="模型保存目录"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="rl_logs",
        help="日志目录"
    )

    args = parser.parse_args()

    # 检查 CUDA 可用性
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU")
        args.device = "cpu"

    # 开始训练
    train_reactor_agent(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        simulation_ticks=args.simulation_ticks,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )


if __name__ == "__main__":
    main()

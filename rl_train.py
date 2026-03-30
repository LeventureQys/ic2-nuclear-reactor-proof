"""
IC2 核反应堆强化学习训练脚本

使用 Stable-Baselines3 训练 AI 设计最优反应堆
"""

import os
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

from model.rl_env import ReactorEnv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ReactorTrainingCallback(BaseCallback):
    """自定义训练回调"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_powers = []
        self.episode_explosions = []

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    if "avg_power" in info:
                        self.episode_powers.append(info["avg_power"])
                        self.episode_explosions.append(info["exploded"])

                        if self.verbose > 0:
                            logger.info(f"Episode 完成: 平均功率={info['avg_power']:.2f} EU/t, "
                                      f"最大堆温={info['max_hull_heat']:.2f}, "
                                      f"爆炸={'是' if info['exploded'] else '否'}")
        return True

    def _on_training_end(self) -> None:
        if len(self.episode_powers) > 0:
            logger.info("=" * 60)
            logger.info("训练统计:")
            logger.info(f"  总 episodes: {len(self.episode_powers)}")
            logger.info(f"  平均功率: {np.mean(self.episode_powers):.2f} EU/t")
            logger.info(f"  最大功率: {np.max(self.episode_powers):.2f} EU/t")
            logger.info(f"  爆炸率: {np.mean(self.episode_explosions) * 100:.2f}%")
            logger.info("=" * 60)


def train_reactor_agent(
    total_timesteps: int = 200000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    save_dir: str = "rl_models",
    log_dir: str = "rl_logs",
    eval_freq: int = 10000,
    simulation_ticks: int = 1000,
):
    """训练反应堆设计 AI"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"PPO_{timestamp}"

    print("=" * 60)
    print("IC2 核反应堆强化学习训练")
    print("=" * 60)
    print(f"算法: PPO")
    print(f"策略: MlpPolicy")
    print(f"总步数: {total_timesteps}")
    print(f"并行环境: {n_envs}")
    print(f"学习率: {learning_rate}")
    print(f"动作空间: Discrete(18) - 只选择组件，无无效动作")
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
        return Monitor(env)

    logger.info(f"创建 {n_envs} 个并行环境")
    env = DummyVecEnv([make_env for _ in range(n_envs)])

    logger.info("创建评估环境")
    eval_env = DummyVecEnv([make_env])

    logger.info("创建PPO模型...")
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
        device="auto"
    )

    # 创建回调
    callbacks = []

    training_callback = ReactorTrainingCallback(verbose=1)
    callbacks.append(training_callback)

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

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(save_dir, model_name, "checkpoints"),
        name_prefix="reactor_model",
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    # 开始训练
    logger.info("开始训练...")
    logger.info(f"提示: 可以使用 TensorBoard 查看训练过程: tensorboard --logdir {log_dir}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
        raise

    # 保存最终模型
    final_model_path = os.path.join(save_dir, model_name, "final_model")
    model.save(final_model_path)
    logger.info(f"最终模型已保存到: {final_model_path}")

    # 清理
    env.close()
    eval_env.close()
    logger.info("环境已关闭")

    return model, model_name


def main():
    parser = argparse.ArgumentParser(description="训练 IC2 核反应堆设计 AI")

    parser.add_argument("--timesteps", type=int, default=200000, help="总训练步数")
    parser.add_argument("--n-envs", type=int, default=4, help="并行环境数量")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--simulation-ticks", type=int, default=1000, help="模拟tick数")
    parser.add_argument("--eval-freq", type=int, default=10000, help="评估频率")

    args = parser.parse_args()

    train_reactor_agent(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        simulation_ticks=args.simulation_ticks,
        eval_freq=args.eval_freq
    )


if __name__ == "__main__":
    main()

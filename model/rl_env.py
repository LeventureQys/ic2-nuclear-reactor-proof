"""
IC2 核反应堆强化学习环境 V2

改进：
1. 使用Discrete动作空间，避免无效动作问题
2. 动作 = 位置索引(0-53) * 18 + 组件索引(0-17)
3. 自动跳过已占用的位置
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from model.reactor import Reactor
from model.components import ComponentFactory, ComponentType, EmptySlot


class ReactorEnvV2(gym.Env):
    """核反应堆强化学习环境 V2 - 简化动作空间"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # 可用的组件类型
    AVAILABLE_COMPONENTS = [
        "E",   # 空槽位
        "U",   # 单铀棒
        "D",   # 双联燃料棒
        "Q",   # 四联燃料棒
        "H",   # 散热片
        "R",   # 反应堆散热片
        "C",   # 元件散热片
        "A",   # 高级散热片
        "O",   # 超频散热片
        "HE",  # 热交换器
        "RH",  # 反应堆热交换器
        "CH",  # 元件热交换器
        "AH",  # 高级热交换器
        "N",   # 中子反射板
        "TN",  # 加厚中子反射板
        "C10", # 10k冷却单元
        "C30", # 30k冷却单元
        "C60", # 60k冷却单元
    ]

    def __init__(
        self,
        rows: int = 9,
        cols: int = 6,
        max_hull_heat: float = 10000,
        simulation_ticks: int = 1000,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.max_hull_heat = max_hull_heat
        self.simulation_ticks = simulation_ticks
        self.render_mode = render_mode

        # 动作空间：只选择组件类型，位置自动按顺序填充
        # 这样避免了"选择已占用位置"的问题
        self.action_space = spaces.Discrete(len(self.AVAILABLE_COMPONENTS))

        # 状态空间：反应堆网格的编码表示
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(len(self.AVAILABLE_COMPONENTS), self.rows, self.cols),
            dtype=np.float32
        )

        # 内部状态
        self.reactor = None
        self.current_layout = None
        self.steps_taken = 0
        self.max_steps = rows * cols
        self.current_position = 0  # 当前要填充的位置

        # 统计信息
        self.episode_stats = {
            "total_power": 0,
            "avg_power": 0,
            "max_hull_heat": 0,
            "exploded": False,
            "components_placed": 0
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        self.reactor = Reactor(self.rows, self.cols, self.max_hull_heat)
        self.current_layout = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.steps_taken = 0
        self.current_position = 0

        self.episode_stats = {
            "total_power": 0,
            "avg_power": 0,
            "max_hull_heat": 0,
            "exploded": False,
            "components_placed": 0
        }

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个动作

        Args:
            action: 组件类型索引 (0-17)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 动作就是组件类型
        component_idx = action

        # 计算当前位置
        row = self.current_position // self.cols
        col = self.current_position % self.cols

        # 放置组件
        component_code = self.AVAILABLE_COMPONENTS[component_idx]
        self.current_layout[row, col] = component_idx
        self.steps_taken += 1
        self.current_position += 1

        if component_code != "E":
            self.episode_stats["components_placed"] += 1

        # 检查是否达到最大步数
        truncated = self.steps_taken >= self.max_steps

        if not truncated:
            # 中间步骤：小奖励
            reward = self._calculate_intermediate_reward(component_code)
            terminated = False
        else:
            # 达到最大步数，运行模拟并计算最终奖励
            reward, terminated = self._evaluate_reactor()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_intermediate_reward(self, component_code: str) -> float:
        """计算中间步骤的奖励"""
        if component_code == "E":
            return 0.0
        elif component_code in ["U", "D", "Q"]:
            return 0.1  # 燃料棒
        elif component_code in ["H", "R", "A", "O"]:
            return 0.05  # 散热片
        else:
            return 0.02  # 其他组件

    def _evaluate_reactor(self) -> Tuple[float, bool]:
        """评估反应堆设计并计算奖励"""
        import logging
        logger = logging.getLogger(__name__)

        # 从当前布局重建反应堆
        layout_codes = []
        for r in range(self.rows):
            row_codes = []
            for c in range(self.cols):
                component_idx = self.current_layout[r, c]
                row_codes.append(self.AVAILABLE_COMPONENTS[component_idx])
            layout_codes.append(row_codes)

        # 创建新反应堆并加载布局
        self.reactor = Reactor(self.rows, self.cols, self.max_hull_heat)
        try:
            self.reactor.load_layout(layout_codes)
        except Exception as e:
            logger.warning(f"布局加载失败: {e}")
            return -100.0, True

        # 运行模拟
        total_power = 0.0
        max_heat = 0.0
        exploded = False

        for tick in range(self.simulation_ticks):
            try:
                result = self.reactor.simulate_tick()
                total_power += result["power"]
                max_heat = max(max_heat, result["hull_heat"])

                if result["exploded"]:
                    exploded = True
                    break
            except Exception as e:
                logger.error(f"模拟tick {tick}时出错: {e}")
                exploded = True
                break

        # 计算平均功率
        ticks_run = tick + 1
        avg_power = total_power / ticks_run if ticks_run > 0 else 0

        # 更新统计
        self.episode_stats["total_power"] = total_power
        self.episode_stats["avg_power"] = avg_power
        self.episode_stats["max_hull_heat"] = max_heat
        self.episode_stats["exploded"] = exploded

        # 计算奖励
        reward = self._calculate_reward(avg_power, max_heat, exploded, ticks_run)

        return reward, exploded

    def _calculate_reward(
        self,
        avg_power: float,
        max_heat: float,
        exploded: bool,
        ticks_run: int
    ) -> float:
        """计算最终奖励"""
        reward = 0.0

        if exploded:
            # 降低爆炸惩罚，鼓励探索高功率设计
            reward = -500.0
            reward += avg_power * 0.2
        else:
            # 提高功率权重，更重视发电量
            reward = avg_power * 1.5

            # 保持严格的堆温安全限制
            heat_ratio = max_heat / self.max_hull_heat
            if heat_ratio > 0.9:
                reward -= 200 * (heat_ratio - 0.9)
            elif heat_ratio > 0.7:
                reward -= 50 * (heat_ratio - 0.7)

            if ticks_run >= self.simulation_ticks:
                reward += 50.0

            # 删除低温奖励，避免过度散热
            # 旧代码: if heat_ratio < 0.5 and avg_power > 100: reward += 100.0

        return reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        obs = np.zeros((len(self.AVAILABLE_COMPONENTS), self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            for c in range(self.cols):
                component_idx = self.current_layout[r, c]
                obs[component_idx, r, c] = 1.0
        return obs

    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            "steps_taken": self.steps_taken,
            "components_placed": self.episode_stats["components_placed"],
            "avg_power": self.episode_stats["avg_power"],
            "max_hull_heat": self.episode_stats["max_hull_heat"],
            "exploded": self.episode_stats["exploded"]
        }

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            print("\n" + "=" * 60)
            print(f"步数: {self.steps_taken}/{self.max_steps}")
            print(f"已放置组件: {self.episode_stats['components_placed']}")
            print("\n当前布局:")

            for r in range(self.rows):
                row_str = ""
                for c in range(self.cols):
                    component_idx = self.current_layout[r, c]
                    component_code = self.AVAILABLE_COMPONENTS[component_idx]
                    row_str += f"{component_code:4s} "
                print(f"第{r+1}行: {row_str}")

            print("=" * 60)

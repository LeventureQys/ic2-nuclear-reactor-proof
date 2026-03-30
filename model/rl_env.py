"""
IC2 核反应堆强化学习环境

使用 Gymnasium 接口，让 AI 学习设计最优的核反应堆配置
目标：最大化 EU/t 发电量，同时避免反应堆爆炸（温度超过 10000）
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from model.reactor import Reactor
from model.components import ComponentFactory, ComponentType, EmptySlot


class ReactorEnv(gym.Env):
    """核反应堆强化学习环境"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # 可用的组件类型（按成本/复杂度排序）
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
        """
        初始化环境

        Args:
            rows: 反应堆行数
            cols: 反应堆列数
            max_hull_heat: 最大堆温（爆炸阈值）
            simulation_ticks: 每次评估运行的 tick 数
            render_mode: 渲染模式
        """
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.max_hull_heat = max_hull_heat
        self.simulation_ticks = simulation_ticks
        self.render_mode = render_mode

        # 动作空间：选择位置 (row, col) 和组件类型
        # 使用 MultiDiscrete: [row, col, component_type]
        self.action_space = spaces.MultiDiscrete([
            self.rows,  # 行索引
            self.cols,  # 列索引
            len(self.AVAILABLE_COMPONENTS)  # 组件类型索引
        ])

        # 状态空间：反应堆网格的编码表示
        # 每个位置用一个整数表示组件类型
        self.observation_space = spaces.Box(
            low=0,
            high=len(self.AVAILABLE_COMPONENTS) - 1,
            shape=(self.rows, self.cols),
            dtype=np.int32
        )

        # 内部状态
        self.reactor = None
        self.current_layout = None
        self.steps_taken = 0
        self.max_steps = rows * cols  # 最多放置所有格子

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

        # 创建新的空反应堆
        self.reactor = Reactor(self.rows, self.cols, self.max_hull_heat)

        # 初始化布局为全空
        self.current_layout = np.zeros((self.rows, self.cols), dtype=np.int32)

        # 重置计数器
        self.steps_taken = 0

        # 重置统计
        self.episode_stats = {
            "total_power": 0,
            "avg_power": 0,
            "max_hull_heat": 0,
            "exploded": False,
            "components_placed": 0
        }

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个动作

        Args:
            action: [row, col, component_type_idx]

        Returns:
            observation: 新的状态
            reward: 奖励值
            terminated: 是否终止（爆炸或完成）
            truncated: 是否截断（超过最大步数）
            info: 额外信息
        """
        row, col, component_idx = action

        # 检查动作是否有效
        if not self._is_valid_action(row, col, component_idx):
            # 无效动作：小惩罚，不改变状态
            return self._get_observation(), -1.0, False, False, self._get_info()

        # 放置组件
        component_code = self.AVAILABLE_COMPONENTS[component_idx]
        self.current_layout[row, col] = component_idx
        self.steps_taken += 1

        if component_code != "E":
            self.episode_stats["components_placed"] += 1

        # 检查是否达到最大步数或用户选择结束
        truncated = self.steps_taken >= self.max_steps

        # 如果还没结束，返回中间奖励
        if not truncated:
            # 中间步骤：小奖励鼓励放置有用的组件
            reward = self._calculate_intermediate_reward(component_code)
            terminated = False
        else:
            # 达到最大步数，运行模拟并计算最终奖励
            reward, terminated = self._evaluate_reactor()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _is_valid_action(self, row: int, col: int, component_idx: int) -> bool:
        """检查动作是否有效"""
        # 检查索引范围
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if not (0 <= component_idx < len(self.AVAILABLE_COMPONENTS)):
            return False

        # 检查位置是否已被占用（除非是放置空槽位）
        if self.current_layout[row, col] != 0 and component_idx != 0:
            return False

        return True

    def _calculate_intermediate_reward(self, component_code: str) -> float:
        """计算中间步骤的奖励"""
        # 简单的启发式奖励
        if component_code == "E":
            return 0.0  # 空槽位无奖励
        elif component_code in ["U", "D", "Q"]:
            return 0.1  # 放置燃料棒有小奖励
        elif component_code in ["H", "R", "A", "O"]:
            return 0.05  # 放置散热片有小奖励
        else:
            return 0.02  # 其他组件有微小奖励

    def _evaluate_reactor(self) -> Tuple[float, bool]:
        """
        评估反应堆设计并计算奖励

        Returns:
            reward: 奖励值
            terminated: 是否因爆炸而终止
        """
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
            # 布局加载失败
            return -100.0, True

        # 运行模拟
        total_power = 0.0
        max_heat = 0.0
        exploded = False

        for tick in range(self.simulation_ticks):
            result = self.reactor.simulate_tick()
            total_power += result["power"]
            max_heat = max(max_heat, result["hull_heat"])

            if result["exploded"]:
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
        """
        计算最终奖励

        奖励设计：
        1. 主要奖励：平均发电功率（EU/t）
        2. 惩罚：温度过高
        3. 严重惩罚：爆炸
        4. 奖励：运行时间长（稳定性）
        """
        reward = 0.0

        if exploded:
            # 爆炸：大惩罚
            reward = -1000.0
            # 但如果在爆炸前产生了一些电力，给予部分奖励
            reward += avg_power * 0.1
        else:
            # 主要奖励：平均发电功率
            reward = avg_power * 1.0

            # 温度惩罚：温度越接近爆炸阈值，惩罚越大
            heat_ratio = max_heat / self.max_hull_heat
            if heat_ratio > 0.9:
                # 非常危险
                reward -= 200 * (heat_ratio - 0.9)
            elif heat_ratio > 0.7:
                # 比较危险
                reward -= 50 * (heat_ratio - 0.7)

            # 稳定性奖励：运行完整个模拟周期
            if ticks_run >= self.simulation_ticks:
                reward += 50.0

            # 效率奖励：如果温度控制得好且发电量高
            if heat_ratio < 0.5 and avg_power > 100:
                reward += 100.0

        return reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        return self.current_layout.copy()

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

            if self.episode_stats["avg_power"] > 0:
                print(f"\n平均功率: {self.episode_stats['avg_power']:.2f} EU/t")
                print(f"最大堆温: {self.episode_stats['max_hull_heat']:.2f} / {self.max_hull_heat}")
                print(f"是否爆炸: {'是' if self.episode_stats['exploded'] else '否'}")

            print("=" * 60)

    def close(self):
        """关闭环境"""
        pass


class ReactorEnvSimplified(gym.Env):
    """
    简化版核反应堆环境

    一次性设计整个反应堆，而不是逐步放置组件
    适合使用进化算法或遗传算法
    """

    metadata = {"render_modes": ["human"]}

    AVAILABLE_COMPONENTS = ReactorEnv.AVAILABLE_COMPONENTS

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

        # 动作空间：整个反应堆的布局
        # 每个位置选择一个组件类型
        self.action_space = spaces.MultiDiscrete(
            [len(self.AVAILABLE_COMPONENTS)] * (rows * cols)
        )

        # 状态空间：只有一个初始状态（空反应堆）
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.episode_stats = {}

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        self.episode_stats = {}
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一个动作（设计整个反应堆）

        Args:
            action: 长度为 rows*cols 的数组，每个元素是组件类型索引
        """
        # 将一维动作转换为二维布局
        layout_codes = []
        for r in range(self.rows):
            row_codes = []
            for c in range(self.cols):
                idx = r * self.cols + c
                component_idx = action[idx]
                row_codes.append(self.AVAILABLE_COMPONENTS[component_idx])
            layout_codes.append(row_codes)

        # 创建反应堆并评估
        reactor = Reactor(self.rows, self.cols, self.max_hull_heat)

        try:
            reactor.load_layout(layout_codes)
        except Exception as e:
            # 布局无效
            return np.array([0.0], dtype=np.float32), -100.0, True, False, {"error": str(e)}

        # 运行模拟
        total_power = 0.0
        max_heat = 0.0
        exploded = False

        for tick in range(self.simulation_ticks):
            result = reactor.simulate_tick()
            total_power += result["power"]
            max_heat = max(max_heat, result["hull_heat"])

            if result["exploded"]:
                exploded = True
                break

        ticks_run = tick + 1
        avg_power = total_power / ticks_run if ticks_run > 0 else 0

        # 计算奖励
        if exploded:
            reward = -1000.0 + avg_power * 0.1
        else:
            reward = avg_power

            heat_ratio = max_heat / self.max_hull_heat
            if heat_ratio > 0.9:
                reward -= 200 * (heat_ratio - 0.9)
            elif heat_ratio > 0.7:
                reward -= 50 * (heat_ratio - 0.7)

            if ticks_run >= self.simulation_ticks:
                reward += 50.0

            if heat_ratio < 0.5 and avg_power > 100:
                reward += 100.0

        # 保存统计
        self.episode_stats = {
            "avg_power": avg_power,
            "total_power": total_power,
            "max_hull_heat": max_heat,
            "exploded": exploded,
            "ticks_run": ticks_run
        }

        # 一次性环境，执行完就结束
        return np.array([0.0], dtype=np.float32), reward, True, False, self.episode_stats

    def render(self):
        """渲染环境"""
        if self.render_mode == "human" and self.episode_stats:
            print(f"平均功率: {self.episode_stats['avg_power']:.2f} EU/t")
            print(f"最大堆温: {self.episode_stats['max_hull_heat']:.2f}")
            print(f"是否爆炸: {'是' if self.episode_stats['exploded'] else '否'}")

    def close(self):
        """关闭环境"""
        pass

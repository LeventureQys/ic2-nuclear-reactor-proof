"""
IC2 核反应堆模拟器 - 反应堆主类

实现反应堆的核心逻辑，包括热量传递、组件交互等
"""

from typing import List, Tuple, Dict
import numpy as np
from model.components import (
    Component, EmptySlot, FuelRod, HeatVent, ReactorHeatVent, ComponentHeatVent,
    AdvancedHeatVent, OverclockedHeatVent, HeatExchanger, ReactorHeatExchanger,
    ComponentHeatExchanger, AdvancedHeatExchanger, CoolantCell, ComponentFactory
)


class Reactor:
    """核反应堆主类"""

    def __init__(self, rows: int = 9, cols: int = 6, max_hull_heat: float = 10000):
        self.rows = rows
        self.cols = cols
        self.max_hull_heat = max_hull_heat
        self.hull_heat = 0.0  # 反应堆本体热量

        # 初始化网格
        self.grid: List[List[Component]] = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(EmptySlot((r, c)))
            self.grid.append(row)

        # 统计数据
        self.total_power_output = 0.0
        self.total_heat_generated = 0.0
        self.is_exploded = False
        self.current_tick = 0

    def load_layout(self, layout: List[List[str]]):
        """从布局配置加载反应堆"""
        if len(layout) != self.rows:
            raise ValueError(f"布局行数不匹配: 期望 {self.rows}, 实际 {len(layout)}")

        for r, row in enumerate(layout):
            if len(row) != self.cols:
                raise ValueError(f"布局第 {r} 行列数不匹配: 期望 {self.cols}, 实际 {len(row)}")

            for c, component_code in enumerate(row):
                self.grid[r][c] = ComponentFactory.create_component(component_code, (r, c))

    def get_component(self, row: int, col: int) -> Component:
        """获取指定位置的组件"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return None

    def get_neighbors(self, row: int, col: int) -> List[Component]:
        """获取相邻的四个组件"""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append(self.grid[nr][nc])

        return neighbors

    def simulate_tick(self) -> Dict:
        """模拟一个tick"""
        if self.is_exploded:
            return {
                "tick": self.current_tick,
                "power": 0,
                "hull_heat": self.hull_heat,
                "exploded": True
            }

        self.current_tick += 1
        tick_power = 0.0
        tick_heat = 0.0

        # 第一阶段：燃料棒产生热量和电力
        fuel_heat_map = {}  # 记录每个燃料棒产生的热量

        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]
                if isinstance(component, FuelRod):
                    result = component.simulate_tick(self)
                    tick_power += result.get("power", 0)
                    heat_generated = result.get("heat_generated", 0)
                    fuel_heat_map[(r, c)] = heat_generated
                    tick_heat += heat_generated

        # 第二阶段：燃料棒热量分配给周围元件
        self._distribute_fuel_heat(fuel_heat_map)

        # 第三阶段：热交换器工作
        self._process_heat_exchangers()

        # 第四阶段：散热片工作
        self._process_heat_vents()

        # 第五阶段：检查组件状态和反应堆状态
        self._check_component_status()

        # 检查反应堆是否爆炸
        if self.hull_heat >= self.max_hull_heat:
            self.is_exploded = True

        self.total_power_output += tick_power
        self.total_heat_generated += tick_heat

        return {
            "tick": self.current_tick,
            "power": tick_power,
            "hull_heat": self.hull_heat,
            "exploded": self.is_exploded,
            "total_power": self.total_power_output,
            "total_heat": self.total_heat_generated
        }

    def _distribute_fuel_heat(self, fuel_heat_map: Dict[Tuple[int, int], float]):
        """分配燃料棒产生的热量给周围元件"""
        for (r, c), heat in fuel_heat_map.items():
            if heat <= 0:
                continue

            # 获取周围可以接受热量的元件
            heat_accepting_neighbors = []
            neighbors = self.get_neighbors(r, c)

            for neighbor in neighbors:
                if self._can_accept_heat(neighbor):
                    heat_accepting_neighbors.append(neighbor)

            if len(heat_accepting_neighbors) == 0:
                # 没有元件接受热量，传递给反应堆本体
                self.hull_heat += heat
            else:
                # 均分热量
                heat_per_component = heat / len(heat_accepting_neighbors)
                remainder = heat % len(heat_accepting_neighbors)

                for i, neighbor in enumerate(heat_accepting_neighbors):
                    heat_to_add = heat_per_component

                    # 处理余数：传给下方元件
                    if i == 0 and remainder > 0:
                        # 简化处理：余数给第一个元件
                        heat_to_add += remainder

                    # 添加热量，但不超过最大值
                    if neighbor.max_heat > 0:
                        available_capacity = neighbor.max_heat - neighbor.heat
                        actual_heat = min(heat_to_add, available_capacity)
                        neighbor.heat += actual_heat

                        # 超出部分传给反应堆
                        if heat_to_add > actual_heat:
                            self.hull_heat += (heat_to_add - actual_heat)
                    else:
                        # 无法储存热量的元件，热量传给反应堆
                        self.hull_heat += heat_to_add

    def _can_accept_heat(self, component: Component) -> bool:
        """检查组件是否可以接受热量"""
        if isinstance(component, EmptySlot):
            return False
        if component.max_heat > 0 and component.heat < component.max_heat:
            return True
        return False

    def _process_heat_exchangers(self):
        """处理热交换器的热量交换"""
        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]

                if isinstance(component, HeatExchanger):
                    # 与相邻元件交换
                    self._exchange_heat_with_neighbors(component, component.component_transfer)
                    # 与反应堆交换
                    self._exchange_heat_with_reactor(component, component.reactor_transfer)

                elif isinstance(component, ReactorHeatExchanger):
                    # 只与反应堆交换
                    self._exchange_heat_with_reactor(component, component.reactor_transfer)

                elif isinstance(component, ComponentHeatExchanger):
                    # 只与相邻元件交换
                    self._exchange_heat_with_neighbors(component, component.component_transfer)

                elif isinstance(component, AdvancedHeatExchanger):
                    # 与相邻元件和反应堆都交换
                    self._exchange_heat_with_neighbors(component, component.component_transfer)
                    self._exchange_heat_with_reactor(component, component.reactor_transfer)

    def _exchange_heat_with_neighbors(self, component: Component, transfer_rate: float):
        """与相邻元件交换热量"""
        neighbors = self.get_neighbors(component.position[0], component.position[1])

        for neighbor in neighbors:
            if isinstance(neighbor, EmptySlot):
                continue

            if neighbor.max_heat > 0:
                # 计算热量差
                heat_diff = component.heat - neighbor.heat

                if abs(heat_diff) > 0:
                    # 热量从高温侧流向低温侧
                    transfer_amount = min(abs(heat_diff), transfer_rate)

                    if heat_diff > 0:
                        # component 温度高，传给 neighbor
                        actual_transfer = min(transfer_amount, component.heat)
                        actual_transfer = min(actual_transfer, neighbor.max_heat - neighbor.heat)

                        component.heat -= actual_transfer
                        neighbor.heat += actual_transfer
                    else:
                        # neighbor 温度高，传给 component
                        actual_transfer = min(transfer_amount, neighbor.heat)
                        actual_transfer = min(actual_transfer, component.max_heat - component.heat)

                        neighbor.heat -= actual_transfer
                        component.heat += actual_transfer

    def _exchange_heat_with_reactor(self, component: Component, transfer_rate: float):
        """与反应堆本体交换热量"""
        if component.max_heat <= 0:
            return

        heat_diff = component.heat - self.hull_heat

        if abs(heat_diff) > 0:
            transfer_amount = min(abs(heat_diff), transfer_rate)

            if heat_diff > 0:
                # component 温度高，传给反应堆
                actual_transfer = min(transfer_amount, component.heat)
                component.heat -= actual_transfer
                self.hull_heat += actual_transfer
            else:
                # 反应堆温度高，传给 component
                actual_transfer = min(transfer_amount, self.hull_heat)
                actual_transfer = min(actual_transfer, component.max_heat - component.heat)
                self.hull_heat -= actual_transfer
                component.heat += actual_transfer

    def _process_heat_vents(self):
        """处理散热片的散热"""
        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]
                result = component.simulate_tick(self)

                # 自身散热
                if "self_cooling" in result:
                    cooling = min(result["self_cooling"], component.heat)
                    component.heat -= cooling

                # 从反应堆吸热并散发
                if "reactor_cooling" in result:
                    cooling = min(result["reactor_cooling"], self.hull_heat)
                    self.hull_heat -= cooling

                    # 吸收到自身
                    if component.max_heat > 0:
                        available = component.max_heat - component.heat
                        absorbed = min(cooling, available)
                        component.heat += absorbed

                        # 立即散发
                        if "self_cooling" in result:
                            dissipate = min(result["self_cooling"], component.heat)
                            component.heat -= dissipate

                # 从相邻元件散热
                if "component_cooling" in result:
                    neighbors = self.get_neighbors(r, c)
                    for neighbor in neighbors:
                        if neighbor.max_heat > 0 and neighbor.heat > 0:
                            cooling = min(result["component_cooling"], neighbor.heat)
                            neighbor.heat -= cooling

    def _check_component_status(self):
        """检查组件状态"""
        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]
                if component.is_broken():
                    # 组件损坏，可以在这里添加处理逻辑
                    pass

    def get_heat_map(self) -> np.ndarray:
        """获取热量分布图"""
        heat_map = np.zeros((self.rows, self.cols))

        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]
                if component.max_heat > 0:
                    heat_map[r][c] = component.heat
                else:
                    heat_map[r][c] = 0

        return heat_map

    def get_heat_percentage_map(self) -> np.ndarray:
        """获取热量百分比分布图"""
        heat_map = np.zeros((self.rows, self.cols))

        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]
                heat_map[r][c] = component.get_heat_percentage()

        return heat_map

    def get_status_summary(self) -> Dict:
        """获取反应堆状态摘要"""
        fuel_rods = []
        heat_components = []
        broken_components = []

        for r in range(self.rows):
            for c in range(self.cols):
                component = self.grid[r][c]

                if isinstance(component, FuelRod):
                    fuel_rods.append({
                        "position": component.position,
                        "name": component.get_name(),
                        "durability": component.get_durability_percentage()
                    })

                if component.max_heat > 0:
                    heat_components.append({
                        "position": component.position,
                        "name": component.get_name(),
                        "heat": component.heat,
                        "max_heat": component.max_heat,
                        "percentage": component.get_heat_percentage()
                    })

                if component.is_broken():
                    broken_components.append({
                        "position": component.position,
                        "name": component.get_name()
                    })

        return {
            "tick": self.current_tick,
            "hull_heat": self.hull_heat,
            "hull_heat_percentage": (self.hull_heat / self.max_hull_heat) * 100,
            "is_exploded": self.is_exploded,
            "total_power": self.total_power_output,
            "fuel_rods": fuel_rods,
            "heat_components": heat_components,
            "broken_components": broken_components
        }

    def print_layout(self):
        """打印反应堆布局"""
        print(f"\n反应堆布局 ({self.rows}x{self.cols}):")
        print("=" * 60)

        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                component = self.grid[r][c]
                name = component.get_name()
                row_str += f"{name:8s} "
            print(f"第{r+1}行: {row_str}")

        print("=" * 60)

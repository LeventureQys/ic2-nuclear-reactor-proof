"""
IC2 核反应堆模拟器 - 核心组件模块

定义所有反应堆组件的基类和具体实现
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from enum import Enum


class ComponentType(Enum):
    """组件类型枚举"""
    EMPTY = "E"
    # 燃料棒
    URANIUM_CELL = "U"
    DUAL_URANIUM_CELL = "D"
    QUAD_URANIUM_CELL = "Q"
    # 散热片
    HEAT_VENT = "H"
    REACTOR_HEAT_VENT = "R"
    COMPONENT_HEAT_VENT = "C"
    ADVANCED_HEAT_VENT = "A"
    OVERCLOCKED_HEAT_VENT = "O"
    # 热交换器
    HEAT_EXCHANGER = "HE"
    REACTOR_HEAT_EXCHANGER = "RH"
    COMPONENT_HEAT_EXCHANGER = "CH"
    ADVANCED_HEAT_EXCHANGER = "AH"
    # 中子反射板
    NEUTRON_REFLECTOR = "N"
    THICK_NEUTRON_REFLECTOR = "TN"
    # 冷却单元
    COOLANT_CELL_10K = "C10"
    COOLANT_CELL_30K = "C30"
    COOLANT_CELL_60K = "C60"


class Component(ABC):
    """反应堆组件基类"""

    def __init__(self, component_type: ComponentType, position: Tuple[int, int]):
        self.component_type = component_type
        self.position = position  # (row, col)
        self.heat = 0.0  # 当前热量
        self.max_heat = 0.0  # 最大热量
        self.durability = 0  # 耐久度（用于中子反射板）
        self.max_durability = 0

    @abstractmethod
    def get_name(self) -> str:
        """获取组件名称"""
        pass

    @abstractmethod
    def simulate_tick(self, reactor: 'Reactor') -> dict:
        """
        模拟一个tick的行为
        返回：包含组件状态变化的字典
        """
        pass

    def get_heat_percentage(self) -> float:
        """获取热量百分比"""
        if self.max_heat == 0:
            return 0.0
        return (self.heat / self.max_heat) * 100

    def get_durability_percentage(self) -> float:
        """获取耐久度百分比"""
        if self.max_durability == 0:
            return 100.0
        return (self.durability / self.max_durability) * 100

    def is_broken(self) -> bool:
        """检查组件是否损坏"""
        if self.max_heat > 0 and self.heat >= self.max_heat:
            return True
        if self.max_durability > 0 and self.durability <= 0:
            return True
        return False

    def __repr__(self):
        return f"{self.get_name()}@{self.position}"


class EmptySlot(Component):
    """空槽位"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.EMPTY, position)

    def get_name(self) -> str:
        return "空槽位"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {}


# ==================== 燃料棒 ====================

class FuelRod(Component):
    """燃料棒基类"""

    def __init__(self, component_type: ComponentType, position: Tuple[int, int],
                 base_output: int, pulse_count: int, heat_multiplier: int):
        super().__init__(component_type, position)
        self.base_output = base_output  # 基础输出（无核脉冲时）
        self.pulse_count = pulse_count  # 发射的核脉冲数
        self.heat_multiplier = heat_multiplier  # 热量倍数
        self.max_durability = 400000  # 工作时长
        self.durability = 400000
        self.received_pulses = 0  # 接收到的核脉冲数

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        """模拟燃料棒工作"""
        if self.durability <= 0:
            return {"power": 0, "heat_generated": 0}

        # 计算接收到的核脉冲数
        self.received_pulses = self._count_received_pulses(reactor)

        # 计算发电量
        power_output = 5 * (self.base_output + self.received_pulses)

        # 计算产生的热量
        adjacent_count = self._count_adjacent_fuel_or_reflector(reactor)
        heat_generated = self.heat_multiplier * (adjacent_count + 1) * (adjacent_count + 2)

        # 消耗耐久
        self.durability -= 1

        return {
            "power": power_output,
            "heat_generated": heat_generated,
            "received_pulses": self.received_pulses,
            "adjacent_count": adjacent_count
        }

    def _count_received_pulses(self, reactor: 'Reactor') -> int:
        """计算接收到的核脉冲数"""
        pulses = 0
        row, col = self.position

        # 检查四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < reactor.rows and 0 <= nc < reactor.cols:
                neighbor = reactor.grid[nr][nc]
                if isinstance(neighbor, FuelRod):
                    pulses += neighbor.pulse_count
                elif isinstance(neighbor, NeutronReflector):
                    pulses += self.pulse_count  # 反射板反射自己的脉冲

        return pulses

    def _count_adjacent_fuel_or_reflector(self, reactor: 'Reactor') -> int:
        """计算相邻的燃料棒或中子反射板数量"""
        count = 0
        row, col = self.position

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < reactor.rows and 0 <= nc < reactor.cols:
                neighbor = reactor.grid[nr][nc]
                if isinstance(neighbor, (FuelRod, NeutronReflector)):
                    count += 1

        return count


class UraniumCell(FuelRod):
    """单铀棒"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.URANIUM_CELL, position,
                        base_output=1, pulse_count=1, heat_multiplier=2)

    def get_name(self) -> str:
        return "单铀棒"


class DualUraniumCell(FuelRod):
    """双联燃料棒"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.DUAL_URANIUM_CELL, position,
                        base_output=4, pulse_count=2, heat_multiplier=4)

    def get_name(self) -> str:
        return "双联燃料棒"


class QuadUraniumCell(FuelRod):
    """四联燃料棒"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.QUAD_URANIUM_CELL, position,
                        base_output=12, pulse_count=4, heat_multiplier=8)

    def get_name(self) -> str:
        return "四联燃料棒"


# ==================== 散热片 ====================

class HeatVent(Component):
    """散热片"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.HEAT_VENT, position)
        self.max_heat = 1000
        self.self_cooling = 6  # 自身散热

    def get_name(self) -> str:
        return "散热片"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        # 散热片在热量传递阶段处理
        return {"self_cooling": self.self_cooling}


class ReactorHeatVent(Component):
    """反应堆散热片"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.REACTOR_HEAT_VENT, position)
        self.max_heat = 1000
        self.self_cooling = 5
        self.reactor_cooling = 5  # 从反应堆吸热

    def get_name(self) -> str:
        return "反应堆散热片"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {
            "self_cooling": self.self_cooling,
            "reactor_cooling": self.reactor_cooling
        }


class ComponentHeatVent(Component):
    """元件散热片"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.COMPONENT_HEAT_VENT, position)
        self.max_heat = 0  # 不储存热量
        self.component_cooling = 4  # 从相邻元件散热

    def get_name(self) -> str:
        return "元件散热片"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {"component_cooling": self.component_cooling}


class AdvancedHeatVent(Component):
    """高级散热片"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.ADVANCED_HEAT_VENT, position)
        self.max_heat = 1000
        self.self_cooling = 12

    def get_name(self) -> str:
        return "高级散热片"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {"self_cooling": self.self_cooling}


class OverclockedHeatVent(Component):
    """超频散热片"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.OVERCLOCKED_HEAT_VENT, position)
        self.max_heat = 1000
        self.self_cooling = 20
        self.reactor_cooling = 36

    def get_name(self) -> str:
        return "超频散热片"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {
            "self_cooling": self.self_cooling,
            "reactor_cooling": self.reactor_cooling
        }


# ==================== 热交换器 ====================

class HeatExchanger(Component):
    """热交换器"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.HEAT_EXCHANGER, position)
        self.max_heat = 2500
        self.component_transfer = 12  # 与相邻元件交换
        self.reactor_transfer = 4  # 与反应堆交换

    def get_name(self) -> str:
        return "热交换器"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {
            "component_transfer": self.component_transfer,
            "reactor_transfer": self.reactor_transfer
        }


class ReactorHeatExchanger(Component):
    """反应堆热交换器"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.REACTOR_HEAT_EXCHANGER, position)
        self.max_heat = 5000
        self.reactor_transfer = 72

    def get_name(self) -> str:
        return "反应堆热交换器"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {"reactor_transfer": self.reactor_transfer}


class ComponentHeatExchanger(Component):
    """元件热交换器"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.COMPONENT_HEAT_EXCHANGER, position)
        self.max_heat = 5000
        self.component_transfer = 36

    def get_name(self) -> str:
        return "元件热交换器"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {"component_transfer": self.component_transfer}


class AdvancedHeatExchanger(Component):
    """高级热交换器"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.ADVANCED_HEAT_EXCHANGER, position)
        self.max_heat = 10000
        self.component_transfer = 24
        self.reactor_transfer = 8

    def get_name(self) -> str:
        return "高级热交换器"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        return {
            "component_transfer": self.component_transfer,
            "reactor_transfer": self.reactor_transfer
        }


# ==================== 中子反射板 ====================

class NeutronReflector(Component):
    """中子反射板"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.NEUTRON_REFLECTOR, position)
        self.max_durability = 30000
        self.durability = 30000

    def get_name(self) -> str:
        return "中子反射板"

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        # 计算相邻燃料棒消耗的耐久
        durability_loss = self._calculate_durability_loss(reactor)
        self.durability -= durability_loss

        return {"durability_loss": durability_loss}

    def _calculate_durability_loss(self, reactor: 'Reactor') -> int:
        """计算耐久消耗"""
        loss = 0
        row, col = self.position

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if 0 <= nr < reactor.rows and 0 <= nc < reactor.cols:
                neighbor = reactor.grid[nr][nc]
                if isinstance(neighbor, UraniumCell):
                    loss += 1
                elif isinstance(neighbor, DualUraniumCell):
                    loss += 2
                elif isinstance(neighbor, QuadUraniumCell):
                    loss += 4

        return loss


class ThickNeutronReflector(NeutronReflector):
    """加厚中子反射板"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(position)
        self.component_type = ComponentType.THICK_NEUTRON_REFLECTOR
        self.max_durability = 120000
        self.durability = 120000

    def get_name(self) -> str:
        return "加厚中子反射板"


# ==================== 冷却单元 ====================

class CoolantCell(Component):
    """冷却单元基类"""

    def __init__(self, component_type: ComponentType, position: Tuple[int, int], max_heat: int):
        super().__init__(component_type, position)
        self.max_heat = max_heat

    def simulate_tick(self, reactor: 'Reactor') -> dict:
        # 冷却单元被动接受热量
        return {}


class CoolantCell10k(CoolantCell):
    """10k冷却单元"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.COOLANT_CELL_10K, position, 10000)

    def get_name(self) -> str:
        return "10k冷却单元"


class CoolantCell30k(CoolantCell):
    """30k冷却单元"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.COOLANT_CELL_30K, position, 30000)

    def get_name(self) -> str:
        return "30k冷却单元"


class CoolantCell60k(CoolantCell):
    """60k冷却单元"""

    def __init__(self, position: Tuple[int, int]):
        super().__init__(ComponentType.COOLANT_CELL_60K, position, 60000)

    def get_name(self) -> str:
        return "60k冷却单元"


# ==================== 组件工厂 ====================

class ComponentFactory:
    """组件工厂类"""

    @staticmethod
    def create_component(component_code: str, position: Tuple[int, int]) -> Component:
        """根据代码创建组件"""
        component_map = {
            "E": EmptySlot,
            "U": UraniumCell,
            "D": DualUraniumCell,
            "Q": QuadUraniumCell,
            "H": HeatVent,
            "R": ReactorHeatVent,
            "C": ComponentHeatVent,
            "A": AdvancedHeatVent,
            "O": OverclockedHeatVent,
            "HE": HeatExchanger,
            "RH": ReactorHeatExchanger,
            "CH": ComponentHeatExchanger,
            "AH": AdvancedHeatExchanger,
            "N": NeutronReflector,
            "TN": ThickNeutronReflector,
            "C10": CoolantCell10k,
            "C30": CoolantCell30k,
            "C60": CoolantCell60k,
        }

        component_class = component_map.get(component_code)
        if component_class is None:
            raise ValueError(f"未知的组件代码: {component_code}")

        return component_class(position)

"""
IC2 核反应堆模拟器 - 模拟引擎

负责运行模拟、收集数据、控制模拟速度
"""

import time
from typing import Dict, List, Callable, Optional
import yaml
from model.reactor import Reactor


class SimulationEngine:
    """模拟引擎"""

    def __init__(self, config_path: str):
        """
        初始化模拟引擎

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.reactor = self._create_reactor()

        # 模拟参数
        self.duration = self.config["simulation"]["duration"]
        self.time_step = self.config["simulation"]["time_step"]
        self.speed_multiplier = self.config["simulation"]["speed_multiplier"]
        self.sample_interval = self.config["simulation"]["sample_interval"]

        # 数据收集
        self.history = {
            "ticks": [],
            "power": [],
            "hull_heat": [],
            "hull_heat_percentage": [],
            "total_power": [],
            "exploded": []
        }

        # 回调函数
        self.tick_callbacks: List[Callable] = []
        self.sample_callbacks: List[Callable] = []

        # 状态
        self.is_running = False
        self.is_paused = False
        self.current_speed = 1.0

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _create_reactor(self) -> Reactor:
        """创建反应堆实例"""
        reactor_params = self.config["reactor_parameters"]
        reactor = Reactor(
            rows=9,
            cols=6,
            max_hull_heat=reactor_params["max_hull_heat"]
        )

        # 加载布局
        layout = self.config["layout"]
        reactor.load_layout(layout)

        return reactor

    def register_tick_callback(self, callback: Callable):
        """注册每个tick的回调函数"""
        self.tick_callbacks.append(callback)

    def register_sample_callback(self, callback: Callable):
        """注册采样回调函数"""
        self.sample_callbacks.append(callback)

    def run(self, realtime: bool = False):
        """
        运行模拟

        Args:
            realtime: 是否实时模拟（考虑速度倍率）
        """
        self.is_running = True
        print(f"\n开始模拟: {self.config['reactor_name']}")
        print(f"模拟时长: {self.duration} ticks")
        print(f"采样间隔: {self.sample_interval} ticks")
        print(f"速度倍率: {self.speed_multiplier}x")
        print("=" * 60)

        self.reactor.print_layout()

        start_time = time.time()
        last_sample_tick = 0

        for tick in range(self.duration):
            if not self.is_running:
                break

            while self.is_paused:
                time.sleep(0.1)
                if not self.is_running:
                    break

            # 模拟一个tick
            result = self.reactor.simulate_tick()

            # 调用tick回调
            for callback in self.tick_callbacks:
                callback(self.reactor, result)

            # 采样
            if tick - last_sample_tick >= self.sample_interval or tick == 0:
                self._collect_sample(result)
                last_sample_tick = tick

                # 调用采样回调
                for callback in self.sample_callbacks:
                    callback(self.reactor, result, self.history)

            # 检查是否爆炸
            if result["exploded"]:
                print(f"\n⚠️  反应堆在第 {tick} tick 爆炸！")
                break

            # 实时模拟延迟
            if realtime and self.speed_multiplier > 0:
                # 1 tick = 1/20 秒 = 0.05 秒
                tick_duration = 0.05 / self.speed_multiplier
                time.sleep(tick_duration)

        end_time = time.time()
        elapsed = end_time - start_time

        self._print_summary(elapsed)
        self.is_running = False

    def run_fast(self):
        """快速运行模拟（不考虑实时）"""
        self.run(realtime=False)

    def run_realtime(self):
        """实时运行模拟"""
        self.run(realtime=True)

    def pause(self):
        """暂停模拟"""
        self.is_paused = True

    def resume(self):
        """恢复模拟"""
        self.is_paused = False

    def stop(self):
        """停止模拟"""
        self.is_running = False

    def set_speed(self, multiplier: float):
        """设置速度倍率"""
        self.speed_multiplier = multiplier
        self.current_speed = multiplier

    def _collect_sample(self, result: Dict):
        """收集采样数据"""
        self.history["ticks"].append(result["tick"])
        self.history["power"].append(result["power"])
        self.history["hull_heat"].append(result["hull_heat"])
        self.history["hull_heat_percentage"].append(
            (result["hull_heat"] / self.reactor.max_hull_heat) * 100
        )
        self.history["total_power"].append(result["total_power"])
        self.history["exploded"].append(result["exploded"])

    def _print_summary(self, elapsed_time: float):
        """打印模拟摘要"""
        print("\n" + "=" * 60)
        print("模拟完成")
        print("=" * 60)

        final_tick = self.reactor.current_tick
        avg_power = self.reactor.total_power_output / final_tick if final_tick > 0 else 0

        print(f"总tick数: {final_tick}")
        print(f"总发电量: {self.reactor.total_power_output:.2f} EU")
        print(f"平均功率: {avg_power:.2f} EU/t")
        print(f"最终堆温: {self.reactor.hull_heat:.2f} / {self.reactor.max_hull_heat} HU")
        print(f"堆温百分比: {(self.reactor.hull_heat / self.reactor.max_hull_heat * 100):.2f}%")
        print(f"是否爆炸: {'是' if self.reactor.is_exploded else '否'}")
        print(f"实际用时: {elapsed_time:.2f} 秒")
        print("=" * 60)

    def get_history(self) -> Dict:
        """获取历史数据"""
        return self.history

    def get_reactor(self) -> Reactor:
        """获取反应堆实例"""
        return self.reactor

    def get_config(self) -> Dict:
        """获取配置"""
        return self.config


class SimulationRecorder:
    """模拟记录器 - 用于记录详细的模拟数据"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.detailed_history = []

    def record_tick(self, reactor: Reactor, result: Dict):
        """记录每个tick的详细数据"""
        record = {
            "tick": result["tick"],
            "power": result["power"],
            "hull_heat": result["hull_heat"],
            "heat_map": reactor.get_heat_map().tolist(),
            "status": reactor.get_status_summary()
        }
        self.detailed_history.append(record)

    def save_to_file(self, filename: str):
        """保存记录到文件"""
        import json
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.detailed_history, f, indent=2, ensure_ascii=False)

        print(f"详细记录已保存到: {filepath}")


class InteractiveSimulation:
    """交互式模拟 - 支持用户控制"""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.commands = {
            "pause": self.pause,
            "resume": self.resume,
            "stop": self.stop,
            "speed": self.set_speed,
            "status": self.print_status,
            "help": self.print_help
        }

    def pause(self):
        """暂停"""
        self.engine.pause()
        print("模拟已暂停")

    def resume(self):
        """恢复"""
        self.engine.resume()
        print("模拟已恢复")

    def stop(self):
        """停止"""
        self.engine.stop()
        print("模拟已停止")

    def set_speed(self, multiplier: str):
        """设置速度"""
        try:
            speed = float(multiplier)
            self.engine.set_speed(speed)
            print(f"速度已设置为: {speed}x")
        except ValueError:
            print("无效的速度值")

    def print_status(self):
        """打印状态"""
        reactor = self.engine.get_reactor()
        status = reactor.get_status_summary()

        print("\n当前状态:")
        print(f"Tick: {status['tick']}")
        print(f"堆温: {status['hull_heat']:.2f} / {reactor.max_hull_heat} HU ({status['hull_heat_percentage']:.2f}%)")
        print(f"总发电: {status['total_power']:.2f} EU")
        print(f"是否爆炸: {'是' if status['is_exploded'] else '否'}")

    def print_help(self):
        """打印帮助"""
        print("\n可用命令:")
        print("  pause  - 暂停模拟")
        print("  resume - 恢复模拟")
        print("  stop   - 停止模拟")
        print("  speed <倍率> - 设置速度倍率")
        print("  status - 显示当前状态")
        print("  help   - 显示帮助")

    def process_command(self, command: str):
        """处理命令"""
        parts = command.strip().split()
        if len(parts) == 0:
            return

        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if cmd in self.commands:
            if len(args) > 0:
                self.commands[cmd](*args)
            else:
                self.commands[cmd]()
        else:
            print(f"未知命令: {cmd}")
            self.print_help()

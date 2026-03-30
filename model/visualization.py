"""
IC2 核反应堆模拟器 - 可视化模块

提供实时可视化功能，包括发电曲线、热量分布、组件状态等
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Dict, Optional
import os


class ReactorVisualizer:
    """反应堆可视化器"""

    def __init__(self, config: Dict, output_dir: str = "output"):
        self.config = config
        self.output_dir = output_dir
        self.colormap = config.get("visualization", {}).get("temperature_colormap", "hot")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 图表设置
        self.fig = None
        self.axes = {}

    def create_static_plots(self, history: Dict, reactor):
        """创建静态图表"""
        print("\n生成可视化图表...")

        # 创建图表布局
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 发电功率曲线
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_power_curve(ax1, history)

        # 2. 堆温曲线
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_hull_heat_curve(ax2, history)

        # 3. 热量分布图
        ax3 = fig.add_subplot(gs[0:2, 2])
        self._plot_heat_distribution(ax3, reactor)

        # 4. 反应堆布局
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_reactor_layout(ax4, reactor)

        # 保存图表
        output_path = os.path.join(self.output_dir, "reactor_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")

        plt.close()

    def _plot_power_curve(self, ax, history: Dict):
        """绘制发电功率曲线"""
        ticks = history["ticks"]
        power = history["power"]

        ax.plot(ticks, power, 'b-', linewidth=2, label='瞬时功率')

        # 计算平均功率
        if len(power) > 0:
            avg_power = np.mean(power)
            ax.axhline(y=avg_power, color='r', linestyle='--',
                      linewidth=1.5, label=f'平均功率: {avg_power:.2f} EU/t')

        ax.set_xlabel('时间 (tick)', fontsize=12)
        ax.set_ylabel('功率 (EU/t)', fontsize=12)
        ax.set_title('发电功率曲线', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

    def _plot_hull_heat_curve(self, ax, history: Dict):
        """绘制堆温曲线"""
        ticks = history["ticks"]
        hull_heat = history["hull_heat"]
        hull_heat_pct = history["hull_heat_percentage"]

        ax1 = ax
        ax2 = ax1.twinx()

        # 绘制绝对温度
        line1 = ax1.plot(ticks, hull_heat, 'r-', linewidth=2, label='堆温 (HU)')
        ax1.set_xlabel('时间 (tick)', fontsize=12)
        ax1.set_ylabel('堆温 (HU)', fontsize=12, color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        # 绘制百分比
        line2 = ax2.plot(ticks, hull_heat_pct, 'orange', linewidth=2,
                        linestyle='--', label='堆温百分比')
        ax2.set_ylabel('堆温百分比 (%)', fontsize=12, color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylim(0, 100)

        # 危险线
        ax2.axhline(y=80, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.text(ticks[-1] if len(ticks) > 0 else 0, 82, '危险',
                color='red', fontsize=10)

        ax1.set_title('反应堆本体温度', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')

    def _plot_heat_distribution(self, ax, reactor):
        """绘制热量分布图"""
        heat_map = reactor.get_heat_percentage_map()

        im = ax.imshow(heat_map, cmap=self.colormap, aspect='auto',
                      vmin=0, vmax=100, interpolation='nearest')

        # 添加数值标注
        for i in range(reactor.rows):
            for j in range(reactor.cols):
                component = reactor.grid[i][j]
                if component.max_heat > 0:
                    text = ax.text(j, i, f'{heat_map[i, j]:.0f}%',
                                 ha="center", va="center", color="white",
                                 fontsize=8, fontweight='bold')

        ax.set_xticks(range(reactor.cols))
        ax.set_yticks(range(reactor.rows))
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        ax.set_title('热量分布图', fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('热量百分比 (%)', fontsize=10)

    def _plot_reactor_layout(self, ax, reactor):
        """绘制反应堆布局"""
        # 创建布局矩阵
        layout_matrix = np.zeros((reactor.rows, reactor.cols))

        # 组件类型映射到颜色
        component_colors = {
            "空槽位": 0,
            "单铀棒": 1,
            "双联燃料棒": 2,
            "四联燃料棒": 3,
            "散热片": 4,
            "反应堆散热片": 5,
            "元件散热片": 6,
            "高级散热片": 7,
            "超频散热片": 8,
            "热交换器": 9,
            "反应堆热交换器": 10,
            "元件热交换器": 11,
            "高级热交换器": 12,
            "中子反射板": 13,
            "加厚中子反射板": 14,
            "10k冷却单元": 15,
            "30k冷却单元": 16,
            "60k冷却单元": 17,
        }

        for i in range(reactor.rows):
            for j in range(reactor.cols):
                component = reactor.grid[i][j]
                name = component.get_name()
                layout_matrix[i, j] = component_colors.get(name, 0)

        # 绘制布局
        im = ax.imshow(layout_matrix, cmap='tab20', aspect='auto',
                      interpolation='nearest')

        # 添加组件名称
        for i in range(reactor.rows):
            for j in range(reactor.cols):
                component = reactor.grid[i][j]
                name = component.get_name()

                # 缩写显示
                abbrev = self._get_component_abbrev(name)
                ax.text(j, i, abbrev, ha="center", va="center",
                       color="white", fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        ax.set_xticks(range(reactor.cols))
        ax.set_yticks(range(reactor.rows))
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        ax.set_title('反应堆布局', fontsize=14, fontweight='bold')

    def _get_component_abbrev(self, name: str) -> str:
        """获取组件缩写"""
        abbrev_map = {
            "空槽位": "空",
            "单铀棒": "U",
            "双联燃料棒": "D",
            "四联燃料棒": "Q",
            "散热片": "H",
            "反应堆散热片": "RH",
            "元件散热片": "CH",
            "高级散热片": "AH",
            "超频散热片": "OH",
            "热交换器": "HE",
            "反应堆热交换器": "RHE",
            "元件热交换器": "CHE",
            "高级热交换器": "AHE",
            "中子反射板": "N",
            "加厚中子反射板": "TN",
            "10k冷却单元": "C10",
            "30k冷却单元": "C30",
            "60k冷却单元": "C60",
        }
        return abbrev_map.get(name, name[:2])

    def create_realtime_visualization(self, engine):
        """创建实时可视化"""
        print("\n启动实时可视化...")

        # 创建图表
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        self.axes['power'] = self.fig.add_subplot(gs[0, :2])
        self.axes['heat'] = self.fig.add_subplot(gs[1, :2])
        self.axes['distribution'] = self.fig.add_subplot(gs[:, 2])

        # 初始化数据
        self.realtime_data = {
            'ticks': [],
            'power': [],
            'hull_heat': []
        }

        # 注册回调
        def update_callback(reactor, result, history):
            self._update_realtime_plots(reactor, history)

        engine.register_sample_callback(update_callback)

        plt.ion()
        plt.show()

    def _update_realtime_plots(self, reactor, history: Dict):
        """更新实时图表"""
        # 更新功率曲线
        ax = self.axes['power']
        ax.clear()
        ax.plot(history['ticks'], history['power'], 'b-', linewidth=2)
        ax.set_xlabel('时间 (tick)')
        ax.set_ylabel('功率 (EU/t)')
        ax.set_title('实时发电功率')
        ax.grid(True, alpha=0.3)

        # 更新堆温曲线
        ax = self.axes['heat']
        ax.clear()
        ax.plot(history['ticks'], history['hull_heat'], 'r-', linewidth=2)
        ax.set_xlabel('时间 (tick)')
        ax.set_ylabel('堆温 (HU)')
        ax.set_title('实时堆温')
        ax.grid(True, alpha=0.3)

        # 更新热量分布
        ax = self.axes['distribution']
        ax.clear()
        heat_map = reactor.get_heat_percentage_map()
        im = ax.imshow(heat_map, cmap=self.colormap, aspect='auto',
                      vmin=0, vmax=100, interpolation='nearest')
        ax.set_title('热量分布')

        plt.pause(0.01)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_text_report(self, engine, filename: str = "simulation_report.txt"):
        """生成文本报告"""
        reactor = engine.get_reactor()
        history = engine.get_history()
        config = engine.get_config()

        report_path = os.path.join(self.output_dir, filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("IC2 核反应堆模拟报告\n")
            f.write("=" * 80 + "\n\n")

            # 基本信息
            f.write(f"反应堆名称: {config['reactor_name']}\n")
            f.write(f"模拟时长: {config['simulation']['duration']} ticks\n")
            f.write(f"采样间隔: {config['simulation']['sample_interval']} ticks\n\n")

            # 性能统计
            f.write("-" * 80 + "\n")
            f.write("性能统计\n")
            f.write("-" * 80 + "\n")

            final_tick = reactor.current_tick
            avg_power = reactor.total_power_output / final_tick if final_tick > 0 else 0
            max_power = max(history['power']) if history['power'] else 0
            min_power = min(history['power']) if history['power'] else 0

            f.write(f"总tick数: {final_tick}\n")
            f.write(f"总发电量: {reactor.total_power_output:.2f} EU\n")
            f.write(f"平均功率: {avg_power:.2f} EU/t\n")
            f.write(f"最大功率: {max_power:.2f} EU/t\n")
            f.write(f"最小功率: {min_power:.2f} EU/t\n\n")

            # 热量统计
            f.write("-" * 80 + "\n")
            f.write("热量统计\n")
            f.write("-" * 80 + "\n")

            max_hull_heat = max(history['hull_heat']) if history['hull_heat'] else 0
            final_hull_heat = reactor.hull_heat

            f.write(f"最终堆温: {final_hull_heat:.2f} HU\n")
            f.write(f"最大堆温: {max_hull_heat:.2f} HU\n")
            f.write(f"堆温上限: {reactor.max_hull_heat} HU\n")
            f.write(f"最大堆温百分比: {(max_hull_heat / reactor.max_hull_heat * 100):.2f}%\n")
            f.write(f"是否爆炸: {'是' if reactor.is_exploded else '否'}\n\n")

            # 组件状态
            f.write("-" * 80 + "\n")
            f.write("组件状态\n")
            f.write("-" * 80 + "\n")

            status = reactor.get_status_summary()

            f.write(f"\n燃料棒 ({len(status['fuel_rods'])} 个):\n")
            for fuel in status['fuel_rods']:
                f.write(f"  位置 {fuel['position']}: {fuel['name']} - "
                       f"耐久 {fuel['durability']:.2f}%\n")

            f.write(f"\n发热组件 ({len(status['heat_components'])} 个):\n")
            for comp in status['heat_components']:
                f.write(f"  位置 {comp['position']}: {comp['name']} - "
                       f"{comp['heat']:.2f}/{comp['max_heat']} HU "
                       f"({comp['percentage']:.2f}%)\n")

            if status['broken_components']:
                f.write(f"\n损坏组件 ({len(status['broken_components'])} 个):\n")
                for comp in status['broken_components']:
                    f.write(f"  位置 {comp['position']}: {comp['name']}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"文本报告已保存到: {report_path}")

    def generate_csv_data(self, history: Dict, filename: str = "simulation_data.csv"):
        """生成CSV数据文件"""
        import csv

        csv_path = os.path.join(self.output_dir, filename)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Tick', 'Power (EU/t)', 'Hull Heat (HU)',
                           'Hull Heat (%)', 'Total Power (EU)', 'Exploded'])

            for i in range(len(history['ticks'])):
                writer.writerow([
                    history['ticks'][i],
                    history['power'][i],
                    history['hull_heat'][i],
                    history['hull_heat_percentage'][i],
                    history['total_power'][i],
                    history['exploded'][i]
                ])

        print(f"CSV数据已保存到: {csv_path}")

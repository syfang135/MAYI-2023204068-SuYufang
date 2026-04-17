"""
数据可视化模块：基于模拟结果生成核心图表。

包含5个核心图表：
1. 策略对比分组柱状图
2. 等待时间分布箱线图
3. 波峰场景队列动态折线图（双Y轴）
4. 故障恢复折线图（双Y轴）
5. 抢占热力图/散点图

所有图表自动保存到 results/ 目录，并包含防御性编程处理空数据。
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
# 设置中文字体，避免警告
import matplotlib
try:
    # 尝试设置中文字体（Windows）
    matplotlib.font_manager.fontManager.addfont("C:\\Windows\\Fonts\\msyh.ttc")
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    # 如果失败，忽略字体设置
    pass

from models import (
    EventLogEntry, MetricsSnapshot, JobLifecycleRecord,
    JobStatus, EventType, FinalMetricsReport
)


# ============================================================================
# 数据准备与辅助函数
# ============================================================================

def ensure_results_dir() -> str:
    """确保 results/ 目录存在，返回目录路径"""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def calculate_final_metrics(
    job_records: List[JobLifecycleRecord],
    metrics_snapshots: List[MetricsSnapshot],
    strategy_name: str,
    simulation_duration: float,
    random_seed: Optional[int] = None
) -> FinalMetricsReport:
    """
    根据作业记录和指标快照计算最终指标报告。

    参数:
        job_records: 作业生命周期记录列表
        metrics_snapshots: 指标快照列表
        strategy_name: 策略名称
        simulation_duration: 模拟总时长
        random_seed: 随机种子

    返回:
        FinalMetricsReport 对象
    """
    if not job_records:
        # 返回空报告
        return FinalMetricsReport(
            high_avg_wait_time=0.0,
            high_p95_wait_time=0.0,
            high_completion_rate=0.0,
            normal_max_wait_time=0.0,
            normal_wait_time_variance=0.0,
            starving_jobs_count=0,
            normal_completion_rate=0.0,
            avg_utilization=0.0,
            peak_utilization=0.0,
            total_preemptions=0,
            total_failures=0,
            utilization_timeline=[],
            queue_length_timeline=[],
            strategy_name=strategy_name,
            simulation_duration=simulation_duration,
            random_seed=random_seed
        )

    # 分离高优和普通作业
    high_jobs = [j for j in job_records if j.priority == "high"]
    normal_jobs = [j for j in job_records if j.priority == "normal"]

    # 计算高优作业指标
    high_wait_times = [j.wait_time for j in high_jobs if j.wait_time is not None]
    high_avg_wait = np.mean(high_wait_times) if high_wait_times else 0.0
    high_p95 = np.percentile(high_wait_times, 95) if high_wait_times else 0.0

    # 计算完成率（完成状态为 COMPLETED）
    high_completed = sum(1 for j in high_jobs if j.final_status == JobStatus.COMPLETED)
    high_completion_rate = high_completed / len(high_jobs) if high_jobs else 0.0

    # 计算普通作业指标
    normal_wait_times = [j.wait_time for j in normal_jobs if j.wait_time is not None]
    normal_max_wait = max(normal_wait_times) if normal_wait_times else 0.0
    normal_variance = np.var(normal_wait_times) if len(normal_wait_times) > 1 else 0.0

    # 饥饿作业计数（等待超过720分钟的普通作业）
    starving_threshold = 720.0
    starving_count = sum(1 for j in normal_jobs
                        if j.wait_time is not None and j.wait_time > starving_threshold)

    normal_completed = sum(1 for j in normal_jobs if j.final_status == JobStatus.COMPLETED)
    normal_completion_rate = normal_completed / len(normal_jobs) if normal_jobs else 0.0

    # 计算集群效率指标
    utilizations = [s.utilization for s in metrics_snapshots]
    avg_utilization = np.mean(utilizations) if utilizations else 0.0
    peak_utilization = max(utilizations) if utilizations else 0.0

    # 统计抢占和故障次数
    total_preemptions = sum(j.preempt_count for j in job_records)
    total_failures = sum(j.retry_count for j in job_records)

    # 提取时间序列数据
    utilization_timeline = [(s.timestamp, s.utilization) for s in metrics_snapshots]

    # 队列长度时间序列（普通作业等待数 + 高优作业等待数）
    queue_length_timeline = []
    for snapshot in metrics_snapshots:
        queue_length = snapshot.pending_normal_count + snapshot.pending_high_count
        queue_length_timeline.append((snapshot.timestamp, queue_length))

    return FinalMetricsReport(
        high_avg_wait_time=high_avg_wait,
        high_p95_wait_time=high_p95,
        high_completion_rate=high_completion_rate,
        normal_max_wait_time=normal_max_wait,
        normal_wait_time_variance=normal_variance,
        starving_jobs_count=starving_count,
        normal_completion_rate=normal_completion_rate,
        avg_utilization=avg_utilization,
        peak_utilization=peak_utilization,
        total_preemptions=total_preemptions,
        total_failures=total_failures,
        utilization_timeline=utilization_timeline,
        queue_length_timeline=queue_length_timeline,
        strategy_name=strategy_name,
        simulation_duration=simulation_duration,
        random_seed=random_seed
    )


def extract_preemption_data(job_records: List[JobLifecycleRecord]) -> Tuple[List[float], List[int]]:
    """
    提取抢占相关数据用于可视化。

    返回:
        (extra_time_list, preempt_count_list) - 额外成本列表和抢占次数列表
    """
    if not job_records:
        return [], []

    extra_time_list = []
    preempt_count_list = []

    for job in job_records:
        if job.priority == "normal" and job.preempt_count > 0:
            extra_time_list.append(job.extra_time_due_to_preempt)
            preempt_count_list.append(job.preempt_count)

    return extra_time_list, preempt_count_list


# ============================================================================
# 核心可视化函数
# ============================================================================

def plot_strategy_comparison(strategy_reports: Dict[str, FinalMetricsReport],
                           save: bool = True) -> Optional[plt.Figure]:
    """
    策略对比分组柱状图。

    对比不同策略的关键指标：
    - 高优作业P95等待时间
    - 普通作业最长等待时间
    - 平均利用率
    - 总抢占次数
    - 饥饿作业数

    参数:
        strategy_reports: 策略名称到FinalMetricsReport的映射
        save: 是否保存图表到文件

    返回:
        matplotlib Figure对象（如果save=False）
    """
    if not strategy_reports:
        print("警告: 策略对比图无数据，跳过生成")
        return None

    strategies = list(strategy_reports.keys())
    if len(strategies) < 1:
        return None

    # 提取指标数据
    high_p95 = [strategy_reports[s].high_p95_wait_time for s in strategies]
    normal_max_wait = [strategy_reports[s].normal_max_wait_time for s in strategies]
    avg_utilization = [strategy_reports[s].avg_utilization * 100 for s in strategies]  # 转换为百分比
    total_preemptions = [strategy_reports[s].total_preemptions for s in strategies]
    starving_jobs = [strategy_reports[s].starving_jobs_count for s in strategies]

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('调度策略对比分析', fontsize=16, fontweight='bold')

    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))

    # 子图1: 高优作业P95等待时间
    ax1 = axes[0, 0]
    bars1 = ax1.bar(strategies, high_p95, color=colors)
    ax1.set_title('高优作业P95等待时间 (分钟)')
    ax1.set_ylabel('等待时间 (分钟)')
    ax1.grid(True, alpha=0.3)
    # 在柱子上添加数值
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 子图2: 普通作业最长等待时间
    ax2 = axes[0, 1]
    bars2 = ax2.bar(strategies, normal_max_wait, color=colors)
    ax2.set_title('普通作业最长等待时间 (分钟)')
    ax2.set_ylabel('等待时间 (分钟)')
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 子图3: 平均利用率
    ax3 = axes[0, 2]
    bars3 = ax3.bar(strategies, avg_utilization, color=colors)
    ax3.set_title('平均利用率')
    ax3.set_ylabel('利用率 (%)')
    ax3.yaxis.set_major_formatter(PercentFormatter())
    ax3.grid(True, alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    # 子图4: 总抢占次数
    ax4 = axes[1, 0]
    bars4 = ax4.bar(strategies, total_preemptions, color=colors)
    ax4.set_title('总抢占次数')
    ax4.set_ylabel('次数')
    ax4.grid(True, alpha=0.3)
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 子图5: 饥饿作业数
    ax5 = axes[1, 1]
    bars5 = ax5.bar(strategies, starving_jobs, color=colors)
    ax5.set_title('饥饿作业数 (等待>720分钟)')
    ax5.set_ylabel('作业数量')
    ax5.grid(True, alpha=0.3)
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    # 子图6: 完成率对比
    ax6 = axes[1, 2]
    high_completion = [strategy_reports[s].high_completion_rate * 100 for s in strategies]
    normal_completion = [strategy_reports[s].normal_completion_rate * 100 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35
    bars6a = ax6.bar(x - width/2, high_completion, width, label='高优作业', color='#FF6B6B')
    bars6b = ax6.bar(x + width/2, normal_completion, width, label='普通作业', color='#4ECDC4')
    ax6.set_title('作业完成率对比')
    ax6.set_ylabel('完成率 (%)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(strategies)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.yaxis.set_major_formatter(PercentFormatter())

    plt.tight_layout()

    if save:
        results_dir = ensure_results_dir()
        filepath = os.path.join(results_dir, 'strategy_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"策略对比图已保存: {filepath}")
        plt.close()
        return None
    else:
        return fig


def plot_wait_time_distribution(job_records: List[JobLifecycleRecord],
                               save: bool = True) -> Optional[plt.Figure]:
    """
    等待时间分布箱线图。

    分别展示高优作业和普通作业的等待时间分布，
    包含异常值识别和统计摘要。

    参数:
        job_records: 作业生命周期记录列表
        save: 是否保存图表到文件

    返回:
        matplotlib Figure对象（如果save=False）
    """
    if not job_records:
        print("警告: 等待时间分布图无数据，跳过生成")
        return None

    # 分离高优和普通作业的等待时间
    high_wait_times = [j.wait_time for j in job_records
                      if j.priority == "high" and j.wait_time is not None]
    normal_wait_times = [j.wait_time for j in job_records
                        if j.priority == "normal" and j.wait_time is not None]

    if not high_wait_times and not normal_wait_times:
        print("警告: 无有效的等待时间数据，跳过生成")
        return None

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('作业等待时间分布分析', fontsize=14, fontweight='bold')

    # 左侧：箱线图
    ax1 = axes[0]
    data_to_plot = []
    labels = []

    if high_wait_times:
        data_to_plot.append(high_wait_times)
        labels.append('高优作业')
    if normal_wait_times:
        data_to_plot.append(normal_wait_times)
        labels.append('普通作业')

    if data_to_plot:
        box = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        # 设置颜色
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(box['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)

        ax1.set_title('等待时间箱线图')
        ax1.set_ylabel('等待时间 (分钟)')
        ax1.grid(True, alpha=0.3)

        # 添加统计信息文本
        stats_text = "统计摘要:\n"
        for i, (label, data) in enumerate(zip(labels, data_to_plot)):
            if data:
                stats_text += f"\n{label}:\n"
                stats_text += f"  数量: {len(data)}\n"
                stats_text += f"  中位数: {np.median(data):.1f} min\n"
                stats_text += f"  平均值: {np.mean(data):.1f} min\n"
                stats_text += f"  标准差: {np.std(data):.1f} min\n"
                stats_text += f"  P95: {np.percentile(data, 95):.1f} min"

        # 右侧：统计信息
        ax2 = axes[1]
        ax2.axis('off')
        ax2.text(0.1, 0.95, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 添加分布直方图（如果数据量足够）
        if len(high_wait_times) > 10 or len(normal_wait_times) > 10:
            # 可以在箱线图下方添加小直方图，这里简化为只显示箱线图
            pass

        plt.tight_layout()

        if save:
            results_dir = ensure_results_dir()
            filepath = os.path.join(results_dir, 'wait_time_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"等待时间分布图已保存: {filepath}")
            plt.close()
            return None
        else:
            return fig
    else:
        plt.close()
        return None


def plot_queue_dynamics(metrics_snapshots: List[MetricsSnapshot],
                       event_logs: Optional[List[EventLogEntry]] = None,
                       save: bool = True) -> Optional[plt.Figure]:
    """
    波峰场景的双Y轴队列动态折线图。

    展示普通作业和高优作业的队列长度随时间变化，
    并标记关键事件（如批量到达、抢占等）。

    参数:
        metrics_snapshots: 指标快照列表
        event_logs: 事件日志列表（可选，用于标记事件）
        save: 是否保存图表到文件

    返回:
        matplotlib Figure对象（如果save=False）
    """
    if not metrics_snapshots:
        print("警告: 队列动态图无数据，跳过生成")
        return None

    # 提取时间序列数据
    timestamps = [s.timestamp for s in metrics_snapshots]
    pending_normal = [s.pending_normal_count for s in metrics_snapshots]
    pending_high = [s.pending_high_count for s in metrics_snapshots]
    utilization = [s.utilization * 100 for s in metrics_snapshots]  # 转换为百分比

    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle('队列动态与系统利用率', fontsize=14, fontweight='bold')

    # 左侧Y轴：队列长度
    color1 = '#1f77b4'
    color2 = '#ff7f0e'

    line1, = ax1.plot(timestamps, pending_normal, color=color1,
                     linewidth=2, label='普通作业队列')
    line2, = ax1.plot(timestamps, pending_high, color=color2,
                     linewidth=2, label='高优作业队列')

    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('队列长度 (作业数)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # 右侧Y轴：利用率
    ax2 = ax1.twinx()
    color3 = '#2ca02c'
    line3, = ax2.plot(timestamps, utilization, color=color3,
                     linewidth=1.5, linestyle='--', alpha=0.7, label='利用率')

    ax2.set_ylabel('利用率 (%)', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # 标记关键事件
    if event_logs:
        # 找出批量提交事件（08:00普通作业批量）
        batch_events = [e for e in event_logs
                       if e.event_type == EventType.JOB_SUBMIT and
                       e.job_priority == "normal" and
                       e.timestamp % 1440 >= 479.9 and e.timestamp % 1440 <= 480.1]

        if batch_events:
            # 按时间分组，每个08:00只标记一次
            batch_times = set()
            for event in batch_events:
                day = int(event.timestamp // 1440)
                if day not in batch_times:
                    batch_times.add(day)
                    ax1.axvline(x=event.timestamp, color='red',
                               linestyle=':', alpha=0.5, linewidth=1)
                    ax1.text(event.timestamp, ax1.get_ylim()[1] * 0.9,
                            '08:00批量', rotation=90, fontsize=8,
                            verticalalignment='top', color='red')

        # 标记抢占事件
        preempt_events = [e for e in event_logs if e.event_type == EventType.JOB_PREEMPT]
        if preempt_events:
            for event in preempt_events:
                ax1.scatter(event.timestamp, ax1.get_ylim()[1] * 0.95,
                          color='purple', marker='x', s=50, zorder=5)

    # 合并图例
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.tight_layout()

    if save:
        results_dir = ensure_results_dir()
        filepath = os.path.join(results_dir, 'queue_dynamics.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"队列动态图已保存: {filepath}")
        plt.close()
        return None
    else:
        return fig


def plot_failure_recovery(metrics_snapshots: List[MetricsSnapshot],
                         event_logs: Optional[List[EventLogEntry]] = None,
                         save: bool = True) -> Optional[plt.Figure]:
    """
    故障恢复的双Y轴折线图。

    展示故障机器数和利用率随时间变化，
    并标记故障点和恢复点。

    参数:
        metrics_snapshots: 指标快照列表
        event_logs: 事件日志列表（可选，用于标记事件）
        save: 是否保存图表到文件

    返回:
        matplotlib Figure对象（如果save=False）
    """
    if not metrics_snapshots:
        print("警告: 故障恢复图无数据，跳过生成")
        return None

    # 提取时间序列数据
    timestamps = [s.timestamp for s in metrics_snapshots]
    down_machines = [s.down_machines for s in metrics_snapshots]
    utilization = [s.utilization * 100 for s in metrics_snapshots]  # 转换为百分比

    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle('故障恢复分析', fontsize=14, fontweight='bold')

    # 左侧Y轴：故障机器数
    color1 = '#d62728'
    line1, = ax1.plot(timestamps, down_machines, color=color1,
                     linewidth=2, label='故障机器数')

    ax1.set_xlabel('时间 (分钟)')
    ax1.set_ylabel('故障机器数', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 右侧Y轴：利用率
    ax2 = ax1.twinx()
    color2 = '#2ca02c'
    line2, = ax2.plot(timestamps, utilization, color=color2,
                     linewidth=1.5, linestyle='--', alpha=0.7, label='利用率')

    ax2.set_ylabel('利用率 (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.yaxis.set_major_formatter(PercentFormatter())

    # 标记故障和恢复事件
    if event_logs:
        failure_events = [e for e in event_logs if e.event_type == EventType.MACHINE_FAILURE]
        repair_events = [e for e in event_logs if e.event_type == EventType.MACHINE_REPAIR]

        # 标记故障点
        for event in failure_events:
            ax1.scatter(event.timestamp, 0, color='red', marker='^',
                       s=100, zorder=5, label='故障发生')
            ax1.text(event.timestamp, ax1.get_ylim()[1] * 0.1,
                    f'故障{event.machine_id}', rotation=90, fontsize=8,
                    verticalalignment='bottom', color='red')

        # 标记恢复点
        for event in repair_events:
            ax1.scatter(event.timestamp, 0, color='green', marker='v',
                       s=100, zorder=5, label='维修完成')
            ax1.text(event.timestamp, ax1.get_ylim()[1] * 0.2,
                    f'恢复{event.machine_id}', rotation=90, fontsize=8,
                    verticalalignment='bottom', color='green')

    # 合并图例（去重）
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.tight_layout()

    if save:
        results_dir = ensure_results_dir()
        filepath = os.path.join(results_dir, 'failure_recovery.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"故障恢复图已保存: {filepath}")
        plt.close()
        return None
    else:
        return fig


def plot_preemption_heatmap(job_records: List[JobLifecycleRecord],
                           save: bool = True) -> Optional[plt.Figure]:
    """
    抢占热力图/散点图。

    展示被抢占作业的额外成本积累情况，
    横轴为作业ID，纵轴为额外成本，颜色/大小表示抢占次数。

    参数:
        job_records: 作业生命周期记录列表
        save: 是否保存图表到文件

    返回:
        matplotlib Figure对象（如果save=False）
    """
    if not job_records:
        print("警告: 抢占热力图无数据，跳过生成")
        return None

    # 提取被抢占的普通作业数据
    preempted_jobs = [j for j in job_records
                     if j.priority == "normal" and j.preempt_count > 0]

    if not preempted_jobs:
        print("警告: 无被抢占作业，跳过生成")
        return None

    # 准备数据
    job_ids = [j.job_id for j in preempted_jobs]
    extra_times = [j.extra_time_due_to_preempt for j in preempted_jobs]
    preempt_counts = [j.preempt_count for j in preempted_jobs]
    wait_times = [j.wait_time for j in preempted_jobs if j.wait_time is not None]
    # 如果有些作业没有wait_time，用0填充
    if len(wait_times) < len(preempted_jobs):
        wait_times = [j.wait_time or 0.0 for j in preempted_jobs]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('作业抢占分析', fontsize=16, fontweight='bold')

    # 子图1: 散点图 - 额外成本 vs 等待时间，颜色表示抢占次数
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(wait_times, extra_times, c=preempt_counts,
                          cmap='YlOrRd', s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('等待时间 (分钟)')
    ax1.set_ylabel('额外成本 (分钟)')
    ax1.set_title('额外成本 vs 等待时间（颜色=抢占次数）')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='抢占次数')

    # 子图2: 柱状图 - 每个作业的抢占次数
    ax2 = axes[0, 1]
    # 只显示前20个作业（如果太多）
    if len(job_ids) > 20:
        show_job_ids = job_ids[:20]
        show_preempt_counts = preempt_counts[:20]
        ax2.set_title(f'抢占次数（前20个作业，共{len(job_ids)}个）')
    else:
        show_job_ids = job_ids
        show_preempt_counts = preempt_counts
        ax2.set_title('抢占次数')

    bars = ax2.bar(range(len(show_job_ids)), show_preempt_counts,
                   color=plt.cm.YlOrRd(np.array(show_preempt_counts)/max(show_preempt_counts)))
    ax2.set_xlabel('作业索引')
    ax2.set_ylabel('抢占次数')
    ax2.set_xticks(range(len(show_job_ids)))
    ax2.set_xticklabels([str(id) for id in show_job_ids], rotation=45, fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # 子图3: 额外成本分布直方图
    ax3 = axes[1, 0]
    if extra_times:
        ax3.hist(extra_times, bins=min(20, len(set(extra_times))),
                color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('额外成本 (分钟)')
        ax3.set_ylabel('作业数')
        ax3.set_title('额外成本分布')
        ax3.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"统计:\n"
        stats_text += f"平均额外成本: {np.mean(extra_times):.1f} min\n"
        stats_text += f"最大额外成本: {max(extra_times):.1f} min\n"
        stats_text += f"总额外成本: {sum(extra_times):.1f} min"
        ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 子图4: 抢占次数与额外成本的关系
    ax4 = axes[1, 1]
    # 计算每个抢占次数的平均额外成本
    preempt_to_extra = {}
    for count, extra in zip(preempt_counts, extra_times):
        if count not in preempt_to_extra:
            preempt_to_extra[count] = []
        preempt_to_extra[count].append(extra)

    if preempt_to_extra:
        avg_extra = [np.mean(preempt_to_extra[count]) for count in sorted(preempt_to_extra.keys())]
        counts = sorted(preempt_to_extra.keys())

        bars4 = ax4.bar(counts, avg_extra, color='lightcoral', edgecolor='black')
        ax4.set_xlabel('抢占次数')
        ax4.set_ylabel('平均额外成本 (分钟)')
        ax4.set_title('抢占次数 vs 平均额外成本')
        ax4.set_xticks(counts)
        ax4.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save:
        results_dir = ensure_results_dir()
        filepath = os.path.join(results_dir, 'preemption_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"抢占分析图已保存: {filepath}")
        plt.close()
        return None
    else:
        return fig


def generate_all_plots(strategy_results: Dict[str, Tuple[List[EventLogEntry],
                                                        List[MetricsSnapshot],
                                                        List[JobLifecycleRecord]]],
                      simulation_duration: float,
                      random_seeds: Optional[Dict[str, int]] = None) -> Dict[str, FinalMetricsReport]:
    """
    为所有策略生成完整的可视化图表。

    参数:
        strategy_results: 策略名称到(事件日志, 指标快照, 作业记录)的映射
        simulation_duration: 模拟总时长
        random_seeds: 策略名称到随机种子的映射（可选）

    返回:
        策略名称到FinalMetricsReport的映射
    """
    print("开始生成可视化图表...")

    # 确保结果目录存在
    ensure_results_dir()

    # 计算每个策略的最终指标
    strategy_reports = {}
    for strategy_name, (event_logs, metrics_snapshots, job_records) in strategy_results.items():
        print(f"处理策略: {strategy_name}")

        random_seed = random_seeds.get(strategy_name) if random_seeds else None
        report = calculate_final_metrics(
            job_records, metrics_snapshots, strategy_name,
            simulation_duration, random_seed
        )
        strategy_reports[strategy_name] = report

        # 生成该策略的专属图表
        if job_records:
            plot_wait_time_distribution(job_records, save=True)
            plot_queue_dynamics(metrics_snapshots, event_logs, save=True)
            plot_failure_recovery(metrics_snapshots, event_logs, save=True)
            plot_preemption_heatmap(job_records, save=True)

    # 生成策略对比图（需要至少2个策略）
    if len(strategy_reports) >= 2:
        plot_strategy_comparison(strategy_reports, save=True)
    else:
        print("警告: 策略数量不足，跳过策略对比图")

    print("所有图表生成完成！")
    return strategy_reports


# ============================================================================
# 主函数（测试用）
# ============================================================================

if __name__ == "__main__":
    print("可视化模块测试")

    # 创建测试数据
    from dataclasses import make_dataclass

    # 创建一个简单的MetricsSnapshot用于测试
    TestSnapshot = make_dataclass('TestSnapshot', [
        ('timestamp', float), ('utilization', float),
        ('pending_normal_count', int), ('pending_high_count', int),
        ('down_machines', int)
    ])

    # 创建测试数据
    test_snapshots = [
        TestSnapshot(timestamp=100.0, utilization=0.6, pending_normal_count=10,
                    pending_high_count=2, down_machines=0),
        TestSnapshot(timestamp=200.0, utilization=0.8, pending_normal_count=5,
                    pending_high_count=1, down_machines=1),
        TestSnapshot(timestamp=300.0, utilization=0.7, pending_normal_count=8,
                    pending_high_count=0, down_machines=0),
    ]

    # 测试图表生成
    print("测试图表生成函数...")
    try:
        fig = plot_queue_dynamics(test_snapshots, save=False)
        if fig is not None:
            plt.show()
        print("队列动态图测试通过")
    except Exception as e:
        print(f"队列动态图测试失败: {e}")

    print("可视化模块测试完成")
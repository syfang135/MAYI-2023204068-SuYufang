"""
集群调度模拟器的核心数据模型定义。

包含事件、作业、机器状态枚举，以及用于指标收集和日志记录的数据类。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random


# ============================================================================
# 事件类型枚举
# ============================================================================

class EventType:
    """事件类型常量定义"""
    # 作业相关事件
    JOB_SUBMIT = "job_submit"          # 作业提交
    JOB_START = "job_start"            # 作业开始执行
    JOB_COMPLETE = "job_complete"      # 作业成功完成
    JOB_FAIL = "job_fail"              # 作业失败
    JOB_PREEMPT = "job_preempt"        # 作业被抢占

    # 机器相关事件
    MACHINE_FAILURE = "machine_failure"  # 机器故障
    MACHINE_REPAIR = "machine_repair"    # 机器维修完成

    # 系统事件
    SCHEDULING_TICK = "scheduling_tick"  # 调度触发点
    SIMULATION_END = "simulation_end"    # 模拟结束


# ============================================================================
# 状态枚举
# ============================================================================

class JobStatus:
    """作业状态常量定义"""
    PENDING = "pending"      # 已提交，等待调度
    RUNNING = "running"      # 已分配机器，执行中
    COMPLETED = "completed"  # 成功完成
    FAILED = "failed"        # 执行失败
    PREEMPTED = "preempted"  # 被抢占


class MachineStatus:
    """机器状态常量定义"""
    IDLE = "idle"    # 空闲可用
    BUSY = "busy"    # 正在执行作业
    DOWN = "down"    # 故障维修中


# ============================================================================
# 核心事件类
# ============================================================================

@dataclass(order=True)
class Event:
    """
    离散事件基类，用于优先队列排序。

    属性:
        timestamp: 事件发生时间戳（分钟）
        event_type: 事件类型，取 EventType 中的常量
        data: 事件附加数据字典

    注意：排序仅基于 timestamp，相同时间戳的事件顺序由 heapq 稳定排序决定。
    """
    timestamp: float
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __repr__(self) -> str:
        return f"Event(t={self.timestamp:.2f}, type={self.event_type}, data={self.data})"


# ============================================================================
# 日志与指标数据类
# ============================================================================

@dataclass
class EventLogEntry:
    """
    事件日志记录，用于系统复盘和调试。

    每次状态转换或重要事件都会生成一条记录，包含完整的上下文信息。
    """
    timestamp: float                     # 事件发生时间
    event_type: str                     # 事件类型
    job_id: Optional[int] = None        # 相关作业ID
    machine_id: Optional[int] = None    # 相关机器ID
    job_priority: Optional[str] = None  # 作业优先级（'high'/'normal'）
    job_resource_num: Optional[int] = None  # 作业需求机器数
    old_status: Optional[str] = None    # 状态变更前状态
    new_status: Optional[str] = None    # 状态变更后状态
    extra_data: Dict[str, Any] = field(default_factory=dict)  # 扩展数据

    # 关键指标预计算字段（可选）
    wait_time: Optional[float] = None    # 等待时间（用于统计）
    utilization: Optional[float] = None  # 当前利用率

    def __repr__(self) -> str:
        return (f"EventLogEntry(t={self.timestamp:.2f}, type={self.event_type}, "
                f"job={self.job_id}, machine={self.machine_id})")


@dataclass
class MetricsSnapshot:
    """
    系统指标快照，定期采集用于时间序列分析。

    每经过固定时间间隔（如10分钟）采集一次，记录系统全局状态。
    """
    timestamp: float          # 采样时间点（分钟）

    # 集群状态
    total_machines: int       # 总机器数 M=100
    busy_machines: int        # 忙碌机器数
    idle_machines: int        # 空闲机器数
    down_machines: int        # 故障机器数
    utilization: float        # 利用率 = (busy + down) / total

    # 队列状态
    pending_normal_count: int  # 等待普通作业数
    pending_high_count: int    # 等待高优作业数
    running_normal_count: int  # 运行中普通作业数
    running_high_count: int    # 运行中高优作业数

    # 饥饿检测
    starving_jobs: int        # 等待超过阈值的普通作业数
    max_wait_time: float      # 当前最长等待时间（分钟）

    # 完成统计
    completed_normal: int     # 已完成的普通作业数
    completed_high: int       # 已完成的高优作业数

    # 性能统计
    preempt_count: int = 0    # 总抢占次数（累计）
    failure_count: int = 0    # 总故障次数（累计）

    def __repr__(self) -> str:
        return (f"MetricsSnapshot(t={self.timestamp:.2f}, "
                f"util={self.utilization:.1%}, "
                f"pending=[N:{self.pending_normal_count}, H:{self.pending_high_count}])")


@dataclass
class JobLifecycleRecord:
    """
    作业生命周期完整记录，用于最终指标计算。

    每个作业从提交到最终完成（或失败）生成一条记录，包含所有关键时间点。
    """
    job_id: int
    priority: str                     # 'high' 或 'normal'
    submit_time: float                # 提交时间戳
    estimated_time: float             # 预估执行时间

    # 执行时间线
    start_time: Optional[float] = None  # 首次开始执行时间
    end_time: Optional[float] = None    # 最终完成/失败时间
    actual_time: Optional[float] = None      # 实际执行时间（最后一次）
    extra_time_due_to_preempt: float = 0.0   # 累计抢占额外成本

    # 状态历史
    preempt_count: int = 0            # 被抢占次数
    retry_count: int = 0              # 重试次数（含故障重试）
    final_status: str = JobStatus.PENDING  # 最终状态

    # 资源信息
    resource_num: int = 1             # 需求机器数
    allocated_machines: List[int] = field(default_factory=list)  # 分配的机器ID列表

    def __repr__(self) -> str:
        return (f"JobLifecycleRecord(id={self.job_id}, priority={self.priority}, "
                f"status={self.final_status}, wait={self.wait_time:.1f})")

    @property
    def wait_time(self) -> float:
        """
        等待时间 = 开始时间 - 提交时间（不含抢占额外成本）。

        注意：这是调度器视角的等待时间，用于公平性指标计算。
        """
        if self.start_time is None:
            return 0.0
        return self.start_time - self.submit_time

    @property
    def total_execution_time(self) -> float:
        """
        总执行时间 = 实际执行时间 + 抢占额外成本。

        注意：实际执行时间可能为 None（作业未完成）。
        """
        actual = self.actual_time or 0.0
        return actual + self.extra_time_due_to_preempt

    @property
    def turnaround_time(self) -> float:
        """
        周转时间 = 完成时间 - 提交时间（含所有成本）。

        注意：如果作业未完成，返回 0。
        """
        if self.end_time is None:
            return 0.0
        return self.end_time - self.submit_time

    @property
    def hrrn_score(self) -> float:
        """
        计算 HRRN（响应比）得分：R = (等待时间 + 预估时间) / 预估时间。

        注意：等待时间需要包含抢占额外成本，因为这部分时间也是作业已经付出的等待。
        """
        wait_with_extra = self.wait_time + self.extra_time_due_to_preempt
        return (wait_with_extra + self.estimated_time) / self.estimated_time


# ============================================================================
# 最终指标报告
# ============================================================================

@dataclass
class FinalMetricsReport:
    """
    模拟结束后的最终指标报告，用于策略对比和评估。
    """
    # 高优作业指标
    high_avg_wait_time: float      # 平均等待时间
    high_p95_wait_time: float      # P95等待时间
    high_completion_rate: float    # 完成率

    # 普通作业公平性指标
    normal_max_wait_time: float    # 最长等待时间
    normal_wait_time_variance: float  # 等待时间方差
    starving_jobs_count: int       # 饥饿作业总数（等待超过阈值）
    normal_completion_rate: float  # 完成率

    # 集群效率指标
    avg_utilization: float         # 平均利用率
    peak_utilization: float        # 峰值利用率
    total_preemptions: int         # 总抢占次数
    total_failures: int            # 总故障次数

    # 时间窗口指标（可输出时间序列）
    utilization_timeline: List[Tuple[float, float]]  # (时间点, 利用率)
    queue_length_timeline: List[Tuple[float, int]]   # (时间点, 队列长度)

    # 策略信息
    strategy_name: str             # 调度策略名称
    simulation_duration: float     # 模拟总时长（分钟）
    random_seed: Optional[int] = None  # 随机种子

    def __repr__(self) -> str:
        return (f"FinalMetricsReport(strategy={self.strategy_name}, "
                f"high_p95={self.high_p95_wait_time:.1f}, "
                f"util={self.avg_utilization:.1%}, "
                f"starving={self.starving_jobs_count})")


# ============================================================================
# 辅助函数
# ============================================================================

def generate_actual_time(estimated_time: float, random_seed: Optional[int] = None) -> float:
    """
    根据预估时间生成实际执行时间，包含 ±20% 的随机误差。

    参数:
        estimated_time: 预估执行时间（分钟）
        random_seed: 随机种子，用于可复现性

    返回:
        actual_time: 实际执行时间，范围在 [estimated_time×0.8, estimated_time×1.2]
    """
    if random_seed is not None:
        # 为每次调用创建独立随机状态，避免影响全局随机性
        local_random = random.Random(random_seed)
    else:
        local_random = random

    # 生成 0.8 到 1.2 之间的均匀分布随机因子
    factor = local_random.uniform(0.8, 1.2)
    return estimated_time * factor


def is_08_00_time(timestamp: float, tolerance: float = 0.01) -> bool:
    """
    判断给定时间戳是否为每天 08:00（480分钟）。

    参数:
        timestamp: 时间戳（分钟）
        tolerance: 浮点数比较容差（分钟）

    返回:
        是否为 08:00（考虑浮点误差）
    """
    # 计算相对于天起始的时间（分钟）
    time_in_day = timestamp % (24 * 60)  # 24小时 = 1440分钟

    # 检查是否接近 480 分钟（08:00）
    return abs(time_in_day - 480) < tolerance


if __name__ == "__main__":
    # 简单测试数据类定义
    print("测试数据类创建...")

    # 测试 Event
    event = Event(timestamp=100.5, event_type=EventType.JOB_SUBMIT,
                  data={"job_id": 1, "priority": "high"})
    print(f"Event: {event}")

    # 测试 EventLogEntry
    log_entry = EventLogEntry(
        timestamp=100.5,
        event_type=EventType.JOB_START,
        job_id=1,
        old_status=JobStatus.PENDING,
        new_status=JobStatus.RUNNING,
        wait_time=10.5
    )
    print(f"EventLogEntry: {log_entry}")

    # 测试 MetricsSnapshot
    snapshot = MetricsSnapshot(
        timestamp=100.5,
        total_machines=100,
        busy_machines=60,
        idle_machines=35,
        down_machines=5,
        utilization=0.65,
        pending_normal_count=50,
        pending_high_count=2,
        running_normal_count=40,
        running_high_count=20,
        starving_jobs=3,
        max_wait_time=120.5,
        completed_normal=100,
        completed_high=50
    )
    print(f"MetricsSnapshot: {snapshot}")

    # 测试 JobLifecycleRecord
    job_record = JobLifecycleRecord(
        job_id=1,
        priority="normal",
        submit_time=90.0,
        start_time=100.5,
        end_time=220.5,
        estimated_time=100.0,
        actual_time=120.0,
        extra_time_due_to_preempt=10.0,
        preempt_count=1,
        final_status=JobStatus.COMPLETED,
        resource_num=3
    )
    print(f"JobLifecycleRecord: {job_record}")
    print(f"  wait_time: {job_record.wait_time:.1f}")
    print(f"  total_execution_time: {job_record.total_execution_time:.1f}")
    print(f"  turnaround_time: {job_record.turnaround_time:.1f}")
    print(f"  hrrn_score: {job_record.hrrn_score:.2f}")

    # 测试 actual_time 生成
    estimated = 100.0
    actual = generate_actual_time(estimated, random_seed=42)
    print(f"actual_time生成: {estimated} -> {actual:.1f} (误差: {actual/estimated:.1%})")

    # 测试 08:00 判断
    test_times = [479.99, 480.0, 480.01, 960.0, 1440.0]
    for t in test_times:
        result = is_08_00_time(t)
        print(f"is_08_00_time({t:.2f}) = {result}")

    print("\n所有数据类测试完成！")
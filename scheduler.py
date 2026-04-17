"""
调度器核心实现。

实现 IScheduler 接口，维护作业和机器的状态机流转。
负责处理所有事件，做出调度决策，并更新系统状态。
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import heapq
from enum import Enum

from models import (
    Event, EventType, JobStatus, MachineStatus,
    EventLogEntry, MetricsSnapshot, JobLifecycleRecord,
    generate_actual_time
)
from strategies import ISchedulingStrategy, StrategyFactory


# ============================================================================
# 内部数据类（不导出）
# ============================================================================

@dataclass
class _Machine:
    """内部机器表示"""
    machine_id: int
    status: str = MachineStatus.IDLE  # 'idle', 'busy', 'down'
    current_job_id: Optional[int] = None  # 当前运行的作业ID（仅当 status='busy' 时有效）
    failure_time: Optional[float] = None  # 故障发生时间
    repair_complete_time: Optional[float] = None  # 维修完成时间

    def __repr__(self) -> str:
        return (f"Machine(id={self.machine_id}, status={self.status}, "
                f"job={self.current_job_id})")


@dataclass
class _Job:
    """内部作业表示"""
    job_id: int
    priority: str  # 'high' 或 'normal'
    submit_time: float
    estimated_time: float
    resource_num: int
    status: str = JobStatus.PENDING

    # 时间相关字段
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    actual_time: Optional[float] = None  # 实际执行时间（生成于开始执行时）
    extra_time_due_to_preempt: float = 0.0  # 累计抢占额外成本

    # 状态历史
    preempt_count: int = 0
    retry_count: int = 0  # 重试次数（失败后重试）
    failure_reason: Optional[str] = None  # 失败原因（'hardware_failure' 或 'other'）

    # 资源分配
    allocated_machines: List[int] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"Job(id={self.job_id}, priority={self.priority}, "
                f"status={self.status}, resources={self.resource_num})")

    def to_lifecycle_record(self) -> JobLifecycleRecord:
        """转换为生命周期记录（用于最终指标）"""
        return JobLifecycleRecord(
            job_id=self.job_id,
            priority=self.priority,
            submit_time=self.submit_time,
            start_time=self.start_time,
            end_time=self.end_time,
            estimated_time=self.estimated_time,
            actual_time=self.actual_time,
            extra_time_due_to_preempt=self.extra_time_due_to_preempt,
            preempt_count=self.preempt_count,
            retry_count=self.retry_count,
            final_status=self.status,
            resource_num=self.resource_num,
            allocated_machines=self.allocated_machines.copy()
        )


# ============================================================================
# 调度器核心类
# ============================================================================

class Scheduler(ISchedulingStrategy):
    """
    调度器实现类，同时实现 ISchedulingStrategy 接口。

    维护作业队列、机器状态，处理所有事件并做出调度决策。
    严格遵循状态机转换规则，确保系统一致性。
    """

    def __init__(self, config: Any, strategy_name: str = "hrrn_preempt"):
        """
        初始化调度器。

        参数:
            config: 模拟器配置对象，需要包含 machine_count 等字段
            strategy_name: 调度策略名称（'fifo' 或 'hrrn_preempt'）
        """
        self.config = config
        self.strategy_name = strategy_name

        # 创建调度策略
        self.strategy = StrategyFactory.create_strategy(
            strategy_name,
            preemption_cost=getattr(config, 'preemption_cost', 10.0)
        )

        # 机器管理
        self.machines: List[_Machine] = []
        self._init_machines(config.machine_count)

        # 作业管理
        self.jobs: Dict[int, _Job] = {}  # job_id -> _Job
        self.pending_jobs: List[Dict[str, Any]] = []  # 等待队列（策略可见的视图）
        self.running_jobs: Dict[int, _Job] = {}  # 运行中作业
        self.completed_jobs: Dict[int, _Job] = {}  # 已完成/失败作业

        # 指标统计
        self.total_preemptions: int = 0
        self.total_failures: int = 0
        self.completed_normal_count: int = 0
        self.completed_high_count: int = 0

        # 下一个作业ID（从调度器内部生成，避免与模拟器冲突）
        self.next_internal_job_id = 1000000

        print(f"[Scheduler] 初始化完成，策略={strategy_name}，机器数={config.machine_count}")

    def _init_machines(self, machine_count: int):
        """初始化机器列表"""
        self.machines = [_Machine(machine_id=i) for i in range(machine_count)]

    def _get_machine_statuses(self) -> List[str]:
        """获取所有机器的状态列表（供策略使用）"""
        return [machine.status for machine in self.machines]

    def _allocate_machines_to_job(self, job: _Job, machine_ids: List[int]):
        """
        将机器分配给作业，更新双方状态。

        参数:
            job: 作业对象
            machine_ids: 分配的机器ID列表
        """
        job.allocated_machines = machine_ids.copy()
        for machine_id in machine_ids:
            machine = self.machines[machine_id]
            machine.status = MachineStatus.BUSY
            machine.current_job_id = job.job_id

    def _release_machines_from_job(self, job_id: int):
        """
        释放作业占用的机器。

        参数:
            job_id: 作业ID
        """
        # 查找作业（可能在 running_jobs 或已完成作业中）
        job = None
        if job_id in self.running_jobs:
            job = self.running_jobs[job_id]
        elif job_id in self.jobs:
            job = self.jobs[job_id]

        if not job:
            return

        for machine_id in job.allocated_machines:
            if 0 <= machine_id < len(self.machines):
                machine = self.machines[machine_id]
                # 只释放没有被故障影响的机器
                if machine.status == MachineStatus.BUSY and machine.current_job_id == job_id:
                    machine.status = MachineStatus.IDLE
                    machine.current_job_id = None

        job.allocated_machines.clear()

    def _create_job_from_event(self, event: Event) -> _Job:
        """从作业提交事件创建内部作业对象"""
        data = event.data
        return _Job(
            job_id=data["job_id"],
            priority=data["priority"],
            submit_time=data["submit_time"],
            estimated_time=data["estimated_time"],
            resource_num=data["resource_num"]
        )

    def _job_to_strategy_view(self, job: _Job) -> Dict[str, Any]:
        """将内部作业对象转换为策略使用的字典视图"""
        return {
            "job_id": job.job_id,
            "priority": job.priority,
            "submit_time": job.submit_time,
            "estimated_time": job.estimated_time,
            "resource_num": job.resource_num,
            "status": job.status,
            "start_time": job.start_time,
            "extra_time_due_to_preempt": job.extra_time_due_to_preempt,
            "preempt_count": job.preempt_count,
            "allocated_machines": job.allocated_machines.copy()
        }

    def _try_schedule_single_job(self, current_time: float) -> List[Event]:
        """
        尝试调度单个作业（从等待队列中选择并分配资源）。

        参数:
            current_time: 当前时间戳

        返回:
            生成的新事件列表（如 JobStartEvent, JobCompleteEvent）
        """
        events = []

        # 获取策略视图的等待作业列表
        pending_view = [self._job_to_strategy_view(self.jobs[j["job_id"]])
                       for j in self.pending_jobs]

        # 让策略选择作业
        selected_view = self.strategy.select_job(pending_view, current_time)
        if not selected_view:
            return events  # 没有作业可调度

        selected_job = self.jobs[selected_view["job_id"]]

        # 尝试分配机器
        machine_statuses = self._get_machine_statuses()
        allocated = self.strategy.allocate_machines(selected_view, machine_statuses)

        if allocated is None:
            # 分配失败，检查是否需要抢占（仅对高优作业）
            if selected_job.priority == "high":
                # 获取运行中普通作业的策略视图
                running_normal_views = []
                for job in self.running_jobs.values():
                    if job.priority == "normal" and job.status == JobStatus.RUNNING:
                        running_normal_views.append(self._job_to_strategy_view(job))

                # 检查是否需要抢占
                should_preempt, victim_view = self.strategy.should_preempt(
                    selected_view, running_normal_views, current_time
                )

                if should_preempt and victim_view:
                    # 执行抢占
                    victim_job = self.jobs[victim_view["job_id"]]
                    return self._execute_preemption(selected_job, victim_job, current_time)

            # 没有抢占或抢占失败，返回空事件列表
            return events

        # 分配成功，启动作业
        # 更新作业状态
        selected_job.status = JobStatus.RUNNING
        selected_job.start_time = current_time

        # 分配机器
        self._allocate_machines_to_job(selected_job, allocated)

        # 从等待队列移除
        self.pending_jobs = [j for j in self.pending_jobs
                            if j["job_id"] != selected_job.job_id]

        # 添加到运行中作业
        self.running_jobs[selected_job.job_id] = selected_job

        # 生成实际执行时间（±20% 误差）
        # 注意：使用独立随机种子确保可复现性
        seed = getattr(self.config, 'random_seed', None)
        if seed is not None:
            local_seed = seed + selected_job.job_id + int(selected_job.estimated_time * 1000)
        else:
            local_seed = None

        selected_job.actual_time = generate_actual_time(
            selected_job.estimated_time, local_seed
        )

        # 生成作业开始事件
        start_event = Event(
            timestamp=current_time,
            event_type=EventType.JOB_START,
            data={
                "job_id": selected_job.job_id,
                "start_time": current_time,
                "allocated_machines": allocated,
                "estimated_time": selected_job.estimated_time,
                "actual_time": selected_job.actual_time
            }
        )
        events.append(start_event)

        # 生成作业完成事件（基于实际执行时间 + 额外成本）
        completion_time = current_time + selected_job.actual_time + selected_job.extra_time_due_to_preempt
        complete_event = Event(
            timestamp=completion_time,
            event_type=EventType.JOB_COMPLETE,
            data={
                "job_id": selected_job.job_id,
                "expected_completion_time": completion_time
            }
        )
        events.append(complete_event)

        print(f"[Scheduler] 作业 {selected_job.job_id}({selected_job.priority}) 开始执行，"
              f"预计完成时间 {completion_time:.2f}")

        return events

    def _execute_preemption(self, high_job: _Job, victim_job: _Job,
                           current_time: float) -> List[Event]:
        """
        执行抢占操作。

        关键步骤：
        1. 更新受害者作业（extra_time +10，状态回退）
        2. 释放受害者的机器
        3. 立即将机器分配给高优作业
        4. 生成抢占事件和作业开始事件

        参数:
            high_job: 高优作业（抢占者）
            victim_job: 受害者作业（普通作业）
            current_time: 当前时间戳

        返回:
            生成的新事件列表
        """
        print(f"[Scheduler] 抢占: 高优作业 {high_job.job_id} 抢占普通作业 {victim_job.job_id}")

        events = []
        self.total_preemptions += 1

        # 1. 更新受害者作业（通过策略）
        victim_view = self._job_to_strategy_view(victim_job)
        updated_view = self.strategy.update_job_after_preempt(victim_view, current_time)

        # 应用更新到受害者作业
        victim_job.extra_time_due_to_preempt = updated_view["extra_time_due_to_preempt"]
        victim_job.preempt_count = updated_view["preempt_count"]
        victim_job.status = JobStatus.PREEMPTED

        # 2. 释放受害者的机器（但先记住是哪些机器）
        victim_machines = victim_job.allocated_machines.copy()
        self._release_machines_from_job(victim_job.job_id)

        # 3. 从运行中作业移除受害者
        if victim_job.job_id in self.running_jobs:
            del self.running_jobs[victim_job.job_id]

        # 4. 将受害者重新加入等待队列（状态回退为 PENDING）
        victim_job.status = JobStatus.PENDING
        victim_job.start_time = None  # 清除开始时间
        victim_job.allocated_machines = []  # 已释放

        # 重新计算HRRN得分并加入等待队列
        victim_view_updated = self._job_to_strategy_view(victim_job)
        self.pending_jobs.append(victim_view_updated)

        # 5. 立即将释放的机器分配给高优作业
        high_job.allocated_machines = victim_machines
        for machine_id in victim_machines:
            machine = self.machines[machine_id]
            machine.status = MachineStatus.BUSY
            machine.current_job_id = high_job.job_id

        # 6. 更新高优作业状态并开始执行
        high_job.status = JobStatus.RUNNING
        high_job.start_time = current_time
        self.running_jobs[high_job.job_id] = high_job

        # 从等待队列移除高优作业
        self.pending_jobs = [j for j in self.pending_jobs
                            if j["job_id"] != high_job.job_id]

        # 7. 为高优作业生成实际执行时间
        seed = getattr(self.config, 'random_seed', None)
        if seed is not None:
            local_seed = seed + high_job.job_id + int(high_job.estimated_time * 1000)
        else:
            local_seed = None

        high_job.actual_time = generate_actual_time(high_job.estimated_time, local_seed)

        # 8. 生成抢占事件
        preempt_event = Event(
            timestamp=current_time,
            event_type=EventType.JOB_PREEMPT,
            data={
                "victim_job_id": victim_job.job_id,
                "high_job_id": high_job.job_id,
                "victim_machines": victim_machines,
                "preemption_cost": self.strategy.preemption_cost
            }
        )
        events.append(preempt_event)

        # 9. 生成高优作业开始事件
        start_event = Event(
            timestamp=current_time,
            event_type=EventType.JOB_START,
            data={
                "job_id": high_job.job_id,
                "start_time": current_time,
                "allocated_machines": victim_machines,
                "estimated_time": high_job.estimated_time,
                "actual_time": high_job.actual_time
            }
        )
        events.append(start_event)

        # 10. 生成高优作业完成事件
        completion_time = current_time + high_job.actual_time + high_job.extra_time_due_to_preempt
        complete_event = Event(
            timestamp=completion_time,
            event_type=EventType.JOB_COMPLETE,
            data={
                "job_id": high_job.job_id,
                "expected_completion_time": completion_time
            }
        )
        events.append(complete_event)

        print(f"[Scheduler] 抢占完成: 受害者 {victim_job.job_id} 回队，"
              f"高优作业 {high_job.job_id} 开始执行")

        return events

    # ============================================================================
    # IScheduler 接口实现
    # ============================================================================

    def handle_job_submit(self, event: Event) -> List[Event]:
        """
        处理作业提交事件。

        步骤：
        1. 创建内部作业对象
        2. 加入等待队列
        3. 尝试调度
        """
        data = event.data
        job_id = data["job_id"]

        print(f"[Scheduler] 处理作业提交: job_id={job_id}, priority={data['priority']}, "
              f"resources={data['resource_num']}")

        # 1. 创建作业对象
        job = self._create_job_from_event(event)
        self.jobs[job_id] = job

        # 2. 加入等待队列（策略视图）
        job_view = self._job_to_strategy_view(job)
        self.pending_jobs.append(job_view)

        # 3. 尝试调度（可能触发抢占）
        events = self.try_schedule(event.timestamp)

        return events

    def handle_job_completion(self, event: Event) -> List[Event]:
        """
        处理作业完成事件（成功或失败）。

        注意：实际成功/失败在模拟器中决定，这里处理两种可能。
        """
        data = event.data
        job_id = data["job_id"]
        success = data.get("success", True)  # 默认为成功
        failure_reason = data.get("failure_reason")

        print(f"[Scheduler] 处理作业完成: job_id={job_id}, success={success}, "
              f"reason={failure_reason}")

        events = []

        # 查找作业
        if job_id not in self.jobs:
            print(f"[Scheduler] 警告: 作业 {job_id} 不存在")
            return events

        job = self.jobs[job_id]

        if success:
            # 作业成功完成
            job.status = JobStatus.COMPLETED
            job.end_time = event.timestamp

            # 更新完成统计
            if job.priority == "normal":
                self.completed_normal_count += 1
            else:
                self.completed_high_count += 1

            # 将作业移动到完成列表
            self.completed_jobs[job_id] = job
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            print(f"[Scheduler] 作业 {job_id} 成功完成")

        else:
            # 作业失败
            self.total_failures += 1
            job.status = JobStatus.FAILED
            job.end_time = event.timestamp
            job.failure_reason = failure_reason
            job.retry_count += 1

            # 检查是否为硬件故障
            is_hardware_failure = (failure_reason == "hardware_failure")

            if is_hardware_failure:
                # 硬件故障：相关机器下线维修
                for machine_id in job.allocated_machines:
                    machine = self.machines[machine_id]
                    machine.status = MachineStatus.DOWN
                    machine.current_job_id = None
                    machine.failure_time = event.timestamp
                    machine.repair_complete_time = event.timestamp + 1440.0  # 24小时

                    # 生成机器维修完成事件
                    repair_event = Event(
                        timestamp=machine.repair_complete_time,
                        event_type=EventType.MACHINE_REPAIR,
                        data={"machine_id": machine_id}
                    )
                    events.append(repair_event)

                print(f"[Scheduler] 作业 {job_id} 因硬件故障失败，"
                      f"{len(job.allocated_machines)} 台机器下线维修")
            else:
                # 非硬件故障：释放机器
                self._release_machines_from_job(job_id)
                print(f"[Scheduler] 作业 {job_id} 因其他原因失败，机器已释放")

            # 作业重试：状态回退为 PENDING，重新加入队列
            job.status = JobStatus.PENDING
            job.start_time = None
            job.allocated_machines = []

            # 重新加入等待队列
            job_view = self._job_to_strategy_view(job)
            self.pending_jobs.append(job_view)

            # 从运行中作业移除
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

        # 释放机器（对于成功或非硬件故障失败）
        if success or (not success and failure_reason != "hardware_failure"):
            self._release_machines_from_job(job_id)

        # 尝试调度新作业
        schedule_events = self.try_schedule(event.timestamp)
        events.extend(schedule_events)

        return events

    def handle_machine_failure(self, event: Event) -> List[Event]:
        """
        处理机器故障事件。

        步骤：
        1. 标记机器为 DOWN
        2. 如果有作业正在该机器上运行，作业失败
        3. 生成作业失败事件
        """
        data = event.data
        machine_id = data["machine_id"]

        print(f"[Scheduler] 处理机器故障: machine_id={machine_id}")

        events = []
        machine = self.machines[machine_id]

        # 1. 更新机器状态
        machine.status = MachineStatus.DOWN
        machine.failure_time = event.timestamp
        machine.repair_complete_time = event.timestamp + 1440.0  # 24小时维修

        # 2. 检查是否有作业正在该机器上运行
        if machine.current_job_id is not None:
            job_id = machine.current_job_id
            if job_id in self.jobs:
                job = self.jobs[job_id]

                # 作业因硬件故障失败
                failure_event = Event(
                    timestamp=event.timestamp,
                    event_type=EventType.JOB_FAIL,
                    data={
                        "job_id": job_id,
                        "failure_reason": "hardware_failure",
                        "failed_machines": [machine_id]
                    }
                )
                events.append(failure_event)

                print(f"[Scheduler] 机器 {machine_id} 故障导致作业 {job_id} 失败")

        # 3. 生成机器维修完成事件
        repair_event = Event(
            timestamp=machine.repair_complete_time,
            event_type=EventType.MACHINE_REPAIR,
            data={"machine_id": machine_id}
        )
        events.append(repair_event)

        return events

    def handle_machine_repair(self, event: Event) -> List[Event]:
        """
        处理机器维修完成事件。

        步骤：
        1. 恢复机器为 IDLE 状态
        2. 尝试调度等待中的作业
        """
        data = event.data
        machine_id = data["machine_id"]

        print(f"[Scheduler] 处理机器维修: machine_id={machine_id}")

        machine = self.machines[machine_id]
        machine.status = MachineStatus.IDLE
        machine.failure_time = None
        machine.repair_complete_time = None

        # 尝试调度等待中的作业
        events = self.try_schedule(event.timestamp)

        return events

    def try_schedule(self, current_time: float) -> List[Event]:
        """
        尝试调度等待队列中的作业。

        可能调度多个作业，直到没有足够资源或没有合适作业。
        """
        events = []

        # 循环尝试调度，直到没有作业可调度
        max_attempts = 100  # 防止无限循环
        for _ in range(max_attempts):
            new_events = self._try_schedule_single_job(current_time)
            if not new_events:
                break  # 没有作业被调度
            events.extend(new_events)

        return events

    def get_metrics_snapshot(self, current_time: float) -> MetricsSnapshot:
        """
        获取当前系统状态的指标快照。

        计算各种统计指标，用于时间序列分析和最终报告。
        """
        # 统计机器状态
        total_machines = len(self.machines)
        busy_machines = sum(1 for m in self.machines if m.status == MachineStatus.BUSY)
        idle_machines = sum(1 for m in self.machines if m.status == MachineStatus.IDLE)
        down_machines = sum(1 for m in self.machines if m.status == MachineStatus.DOWN)

        # 计算利用率（包含故障机器）
        utilization = (busy_machines + down_machines) / total_machines if total_machines > 0 else 0.0

        # 统计作业状态
        pending_normal = sum(1 for j in self.pending_jobs if j["priority"] == "normal")
        pending_high = sum(1 for j in self.pending_jobs if j["priority"] == "high")

        running_normal = sum(1 for j in self.running_jobs.values() if j.priority == "normal")
        running_high = sum(1 for j in self.running_jobs.values() if j.priority == "high")

        # 计算最长等待时间（仅普通作业）
        max_wait_time = 0.0
        for job_view in self.pending_jobs:
            if job_view["priority"] == "normal":
                wait_time = current_time - job_view["submit_time"] + job_view.get("extra_time_due_to_preempt", 0.0)
                max_wait_time = max(max_wait_time, wait_time)

        # 统计饥饿作业（等待超过阈值的普通作业）
        starving_threshold = getattr(self.config, 'starvation_threshold', 720.0)
        starving_jobs = 0
        for job_view in self.pending_jobs:
            if job_view["priority"] == "normal":
                wait_time = current_time - job_view["submit_time"] + job_view.get("extra_time_due_to_preempt", 0.0)
                if wait_time > starving_threshold:
                    starving_jobs += 1

        return MetricsSnapshot(
            timestamp=current_time,
            total_machines=total_machines,
            busy_machines=busy_machines,
            idle_machines=idle_machines,
            down_machines=down_machines,
            utilization=utilization,
            pending_normal_count=pending_normal,
            pending_high_count=pending_high,
            running_normal_count=running_normal,
            running_high_count=running_high,
            starving_jobs=starving_jobs,
            max_wait_time=max_wait_time,
            completed_normal=self.completed_normal_count,
            completed_high=self.completed_high_count,
            preempt_count=self.total_preemptions,
            failure_count=self.total_failures
        )

    def get_job_records(self) -> List[JobLifecycleRecord]:
        """
        获取所有作业的生命周期记录。

        包括等待中、运行中、已完成的所有作业。
        """
        records = []

        # 收集所有作业
        all_jobs = list(self.jobs.values())

        # 添加已完成作业（可能不在 self.jobs 中？这里设计是都在）
        for job in all_jobs:
            records.append(job.to_lifecycle_record())

        return records

    # ============================================================================
    # ISchedulingStrategy 接口实现（委托给内部策略）
    # ============================================================================

    def select_job(self, pending_jobs: List[Dict[str, Any]],
                   current_time: float) -> Optional[Dict[str, Any]]:
        return self.strategy.select_job(pending_jobs, current_time)

    def allocate_machines(self, job: Dict[str, Any],
                          machine_statuses: List[str]) -> Optional[List[int]]:
        return self.strategy.allocate_machines(job, machine_statuses)

    def should_preempt(self, high_job: Dict[str, Any],
                       running_normal_jobs: List[Dict[str, Any]],
                       current_time: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        return self.strategy.should_preempt(high_job, running_normal_jobs, current_time)

    def update_job_after_preempt(self, victim_job: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        return self.strategy.update_job_after_preempt(victim_job, current_time)

    def calculate_hrrn_score(self, job: Dict[str, Any],
                             current_time: float) -> float:
        return self.strategy.calculate_hrrn_score(job, current_time)


# ============================================================================
# 测试函数
# ============================================================================

def test_scheduler():
    """测试调度器基本功能"""
    print("=" * 60)
    print("测试调度器")
    print("=" * 60)

    # 创建模拟配置
    from dataclasses import make_dataclass
    Config = make_dataclass('Config', [
        ('machine_count', int, 10),
        ('preemption_cost', float, 10.0),
        ('starvation_threshold', float, 720.0),
        ('random_seed', int, 42)
    ])
    config = Config()

    # 测试 FIFO 调度器
    print("\n1. 测试 FIFO 调度器:")
    fifo_scheduler = Scheduler(config, "fifo")

    # 测试作业提交
    submit_event = Event(
        timestamp=100.0,
        event_type=EventType.JOB_SUBMIT,
        data={
            "job_id": 1,
            "priority": "normal",
            "submit_time": 100.0,
            "estimated_time": 60.0,
            "resource_num": 2
        }
    )

    events = fifo_scheduler.handle_job_submit(submit_event)
    print(f"  作业提交后生成事件数: {len(events)}")

    # 获取指标快照
    snapshot = fifo_scheduler.get_metrics_snapshot(100.0)
    print(f"  指标快照: pending_normal={snapshot.pending_normal_count}, "
          f"utilization={snapshot.utilization:.1%}")

    # 测试 HRRN+抢占调度器
    print("\n2. 测试 HRRN+抢占调度器:")
    hrrn_scheduler = Scheduler(config, "hrrn_preempt")

    # 提交一个普通作业
    normal_event = Event(
        timestamp=100.0,
        event_type=EventType.JOB_SUBMIT,
        data={
            "job_id": 101,
            "priority": "normal",
            "submit_time": 100.0,
            "estimated_time": 120.0,
            "resource_num": 2
        }
    )
    hrrn_scheduler.handle_job_submit(normal_event)

    # 提交一个高优作业（应该会调度成功，因为有足够机器）
    high_event = Event(
        timestamp=100.0,
        event_type=EventType.JOB_SUBMIT,
        data={
            "job_id": 102,
            "priority": "high",
            "submit_time": 100.0,
            "estimated_time": 60.0,
            "resource_num": 1
        }
    )
    events = hrrn_scheduler.handle_job_submit(high_event)
    print(f"  高优作业提交后生成事件数: {len(events)}")

    # 获取作业记录
    records = hrrn_scheduler.get_job_records()
    print(f"  作业记录数: {len(records)}")

    print("\n调度器测试完成！")


if __name__ == "__main__":
    test_scheduler()
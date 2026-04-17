"""
离散事件驱动的集群调度模拟器。

核心功能：
1. 基于优先队列的事件驱动时间推进
2. 生成作业提交事件（普通作业批量 + 高优作业泊松过程）
3. 模拟作业执行（包含 ±20% 时间误差和随机失败）
4. 生成机器故障事件
5. 调用调度器进行决策，并处理状态转换

注意：本模块不实现具体的调度策略，只提供模拟环境和事件驱动框架。
"""

import heapq
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from models import (
    Event, EventType, JobStatus, MachineStatus,
    EventLogEntry, MetricsSnapshot, JobLifecycleRecord,
    generate_actual_time, is_08_00_time
)


# ============================================================================
# 调度器接口定义
# ============================================================================

class IScheduler(ABC):
    """
    调度器抽象接口，具体调度策略需实现此接口。

    所有方法接收事件作为输入，返回可能触发的新事件列表。
    调度器负责维护作业队列、机器状态，并做出调度决策。
    """

    @abstractmethod
    def handle_job_submit(self, event: Event) -> List[Event]:
        """
        处理作业提交事件。

        参数:
            event: JobSubmitEvent，包含作业属性

        返回:
            可能触发的新事件列表，如 JobStartEvent（如果立即调度成功）
        """
        pass

    @abstractmethod
    def handle_job_completion(self, event: Event) -> List[Event]:
        """
        处理作业完成事件（成功或失败）。

        参数:
            event: JobCompleteEvent 或 JobFailEvent

        返回:
            可能触发的新事件列表，如调度新作业、机器故障事件等
        """
        pass

    @abstractmethod
    def handle_machine_failure(self, event: Event) -> List[Event]:
        """
        处理机器故障事件。

        参数:
            event: MachineFailureEvent

        返回:
            可能触发的新事件列表，如受影响作业的失败事件
        """
        pass

    @abstractmethod
    def handle_machine_repair(self, event: Event) -> List[Event]:
        """
        处理机器维修完成事件。

        参数:
            event: MachineRepairEvent

        返回:
            可能触发的新事件列表，如调度等待中的作业
        """
        pass

    @abstractmethod
    def try_schedule(self, current_time: float) -> List[Event]:
        """
        尝试调度等待队列中的作业。

        参数:
            current_time: 当前时间戳

        返回:
            可能触发的新事件列表，如 JobStartEvent
        """
        pass

    @abstractmethod
    def get_metrics_snapshot(self, current_time: float) -> MetricsSnapshot:
        """
        获取当前系统状态的指标快照。

        参数:
            current_time: 当前时间戳

        返回:
            包含所有系统指标的 MetricsSnapshot
        """
        pass

    @abstractmethod
    def get_job_records(self) -> List[JobLifecycleRecord]:
        """
        获取所有作业的生命周期记录。

        返回:
            所有作业的 JobLifecycleRecord 列表
        """
        pass


# ============================================================================
# 模拟器配置
# ============================================================================

@dataclass
class SimulatorConfig:
    """模拟器配置参数"""
    # 集群配置
    machine_count: int = 100          # 机器总数 M
    failure_prob_per_day: float = 0.001  # 每台机器每天的故障概率

    # 作业生成配置
    normal_batch_size: int = 500      # 每天08:00普通作业批量大小
    high_arrival_rate: float = 4.0    # 高优作业到达率（个/小时）

    # 作业属性配置
    resource_num_range: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    resource_num_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.15, 0.1, 0.05])
    estimated_time_range: Tuple[float, float] = (30.0, 180.0)  # 预估时间范围（分钟）

    # 故障与失败配置
    job_failure_prob: float = 0.05          # 作业失败概率
    hardware_failure_ratio: float = 0.3     # 失败中硬件故障的比例

    # 模拟配置
    duration_minutes: float = 10080.0       # 模拟总时长（7天，分钟）
    random_seed: Optional[int] = None       # 随机种子（None表示随机）
    sampling_interval: float = 10.0         # 指标采样间隔（分钟）

    # 调度策略配置
    scheduler_strategy: str = "hrrn_preempt"  # 调度策略名称
    preemption_enabled: bool = True         # 是否允许抢占
    preemption_cost: float = 10.0           # 每次抢占的额外成本（分钟）
    starvation_threshold: float = 720.0     # 饥饿作业判定阈值（分钟）

    def __post_init__(self):
        """配置后处理，验证参数有效性"""
        assert self.machine_count > 0, "机器数必须大于0"
        assert 0 <= self.failure_prob_per_day <= 1, "故障概率必须在[0,1]范围内"
        assert self.normal_batch_size > 0, "普通作业批量大小必须大于0"
        assert self.high_arrival_rate > 0, "高优作业到达率必须大于0"
        assert self.duration_minutes > 0, "模拟时长必须大于0"
        assert len(self.resource_num_range) == len(self.resource_num_weights), \
            "resource_num_range 和 weights 长度必须相同"


# ============================================================================
# 模拟器核心类
# ============================================================================

class Simulator:
    """
    离散事件驱动模拟器。

    核心职责：
    1. 管理事件优先队列，按时间顺序处理事件
    2. 生成作业提交和机器故障等外部事件
    3. 调用调度器处理事件，并将产生的新事件加入队列
    4. 收集日志和指标数据
    """

    def __init__(self, config: SimulatorConfig, scheduler: IScheduler):
        """
        初始化模拟器。

        参数:
            config: 模拟器配置
            scheduler: 调度器实例（需实现 IScheduler 接口）
        """
        self.config = config
        self.scheduler = scheduler

        # 随机数生成器（确保可复现性）
        self.random = random.Random(config.random_seed)

        # 事件队列（最小堆，按时间戳排序）
        self.event_queue: List[Event] = []

        # 状态变量
        self.current_time: float = 0.0
        self.is_running: bool = False
        self.next_high_job_id: int = 1
        self.next_normal_job_id: int = 10000  # 普通作业ID从10000开始，便于区分

        # 数据收集
        self.event_logs: List[EventLogEntry] = []
        self.metrics_snapshots: List[MetricsSnapshot] = []
        self.last_sample_time: float = 0.0

        # 下一个高优作业到达时间（泊松过程）
        self.next_high_arrival_time: float = self._generate_next_high_arrival()

        # DES陷阱防御状态
        self.last_batch_day = -1  # 上次生成普通作业批处理的天数
        self.last_failure_check_time = -1.0  # 上次检查机器故障的时间

    def _generate_next_high_arrival(self) -> float:
        """
        生成下一个高优作业的到达时间（泊松过程）。

        返回:
            下一个高优作业到达的时间戳（分钟）
        """
        # 泊松过程：到达间隔服从指数分布
        # λ = 4个/小时 = 4/60 个/分钟
        lambda_per_minute = self.config.high_arrival_rate / 60.0
        interval = self.random.expovariate(lambda_per_minute)
        return self.current_time + interval

    def _generate_job_attributes(self, priority: str) -> Dict[str, Any]:
        """
        生成作业的随机属性。

        参数:
            priority: 作业优先级（'high' 或 'normal'）

        返回:
            包含作业属性的字典
        """
        # 随机选择资源需求数量（加权随机）
        resource_num = self.random.choices(
            self.config.resource_num_range,
            weights=self.config.resource_num_weights,
            k=1
        )[0]

        # 随机生成预估执行时间（均匀分布）
        estimated_time = self.random.uniform(
            self.config.estimated_time_range[0],
            self.config.estimated_time_range[1]
        )

        return {
            "resource_num": resource_num,
            "estimated_time": estimated_time,
            "priority": priority,
            # 注意：actual_time 在作业开始时由 simulate_job_execution 生成
        }

    def generate_high_job_event(self) -> Event:
        """
        生成高优作业提交事件，并更新下一个到达时间。

        返回:
            JobSubmitEvent 事件
        """
        # 生成作业属性
        attributes = self._generate_job_attributes("high")
        job_id = self.next_high_job_id
        self.next_high_job_id += 1

        # 创建事件
        event = Event(
            timestamp=self.next_high_arrival_time,
            event_type=EventType.JOB_SUBMIT,
            data={
                "job_id": job_id,
                "submit_time": self.next_high_arrival_time,
                **attributes
            }
        )

        # 更新下一个到达时间
        self.next_high_arrival_time = self._generate_next_high_arrival()

        return event

    def generate_normal_batch_events(self) -> List[Event]:
        """
        如果当前时间是 08:00，生成普通作业批量提交事件。
        使用 last_batch_day 防御重复生成。

        返回:
            JobSubmitEvent 事件列表（可能为空）
        """
        events = []
        if not is_08_00_time(self.current_time):
            return events

        # 计算当前天数（整数除法，每天1440分钟）
        current_day = int(self.current_time // 1440)

        # 如果今天已经生成过批处理，则跳过
        if current_day <= self.last_batch_day:
            return events

        # 标记今天已生成批处理
        self.last_batch_day = current_day

        for _ in range(self.config.normal_batch_size):
            attributes = self._generate_job_attributes("normal")
            job_id = self.next_normal_job_id
            self.next_normal_job_id += 1

            event = Event(
                timestamp=self.current_time,
                event_type=EventType.JOB_SUBMIT,
                data={
                    "job_id": job_id,
                    "submit_time": self.current_time,
                    **attributes
                }
            )
            events.append(event)

        print(f"[{self.current_time:.2f}] 生成 {len(events)} 个普通作业（08:00批量）")
        return events

    def generate_machine_failure_events(self) -> List[Event]:
        """
        按固定的时间间隔（每分钟）生成机器故障事件。
        使用 last_failure_check_time 防御重复计算。

        返回:
            MachineFailureEvent 事件列表（可能为空）
        """
        events = []

        # 每分钟检查一次故障
        if self.last_failure_check_time < 0:
            # 第一次调用，初始化检查时间，不生成故障
            self.last_failure_check_time = self.current_time
            return events

        time_since_last_check = self.current_time - self.last_failure_check_time
        if time_since_last_check < 1.0:
            # 距离上次检查不足1分钟，不生成故障
            return events

        # 更新最后检查时间（向前取整到当前分钟）
        self.last_failure_check_time = self.current_time

        # 将每天的故障概率转换为每分钟概率
        minute_prob = self.config.failure_prob_per_day / (24 * 60)

        # 每台机器独立检查是否发生故障
        # 注意：实际实现中需要知道机器总数，这里简化处理
        # 在完整实现中，应该遍历所有机器检查
        # 这里生成一个随机数量的故障事件
        expected_failures = minute_prob * self.config.machine_count
        if self.random.random() < expected_failures:  # 简化：以概率生成一个故障
            # 随机选择一台机器
            machine_id = self.random.randint(0, self.config.machine_count - 1)
            event = Event(
                timestamp=self.current_time,
                event_type=EventType.MACHINE_FAILURE,
                data={
                    "machine_id": machine_id,
                    "failure_type": "hardware"  # 目前只模拟硬件故障
                }
            )
            events.append(event)

        return events

    def simulate_job_execution(self, job_attrs: Dict[str, Any]) -> Tuple[float, bool, bool]:
        """
        模拟作业执行，生成实际时间和成功/失败结果。

        参数:
            job_attrs: 作业属性字典，必须包含 estimated_time

        返回:
            (actual_time, success, is_hardware_failure) 元组:
            - actual_time: 实际执行时间（分钟），包含 ±20% 误差
            - success: 是否成功完成
            - is_hardware_failure: 如果失败，是否为硬件故障
        """
        estimated_time = job_attrs["estimated_time"]

        # 生成实际执行时间（±20% 误差）
        # 注意：使用独立随机种子确保可复现性
        seed = self.config.random_seed
        if seed is not None:
            # 基于作业ID和预估时间生成确定性种子
            job_id = job_attrs.get("job_id", 0)
            local_seed = seed + job_id + int(estimated_time * 1000)
        else:
            local_seed = None

        actual_time = generate_actual_time(estimated_time, local_seed)

        # 决定作业是否失败
        success = self.random.random() >= self.config.job_failure_prob

        # 如果失败，决定是否为硬件故障
        is_hardware_failure = False
        if not success:
            is_hardware_failure = self.random.random() < self.config.hardware_failure_ratio

        return actual_time, success, is_hardware_failure

    def _collect_metrics_snapshot(self):
        """收集当前时刻的指标快照"""
        snapshot = self.scheduler.get_metrics_snapshot(self.current_time)
        self.metrics_snapshots.append(snapshot)
        self.last_sample_time = self.current_time

    def _log_event(self, event: Event, extra_data: Dict[str, Any] = None):
        """
        记录事件日志。

        参数:
            event: 事件对象
            extra_data: 额外的日志数据
        """
        log_entry = EventLogEntry(
            timestamp=event.timestamp,
            event_type=event.event_type,
            job_id=event.data.get("job_id"),
            machine_id=event.data.get("machine_id"),
            job_priority=event.data.get("priority"),
            job_resource_num=event.data.get("resource_num"),
            extra_data=extra_data or {}
        )
        self.event_logs.append(log_entry)

    def _handle_event(self, event: Event) -> List[Event]:
        """
        处理单个事件，调用调度器的相应方法。

        参数:
            event: 待处理的事件

        返回:
            调度器返回的新事件列表
        """
        # 记录事件日志
        self._log_event(event)

        # 根据事件类型调用调度器的不同处理方法
        if event.event_type == EventType.JOB_SUBMIT:
            new_events = self.scheduler.handle_job_submit(event)

        elif event.event_type in [EventType.JOB_COMPLETE, EventType.JOB_FAIL]:
            new_events = self.scheduler.handle_job_completion(event)

        elif event.event_type == EventType.MACHINE_FAILURE:
            new_events = self.scheduler.handle_machine_failure(event)

        elif event.event_type == EventType.MACHINE_REPAIR:
            new_events = self.scheduler.handle_machine_repair(event)

        elif event.event_type == EventType.JOB_PREEMPT:
            # 抢占事件由调度器内部处理，这里只记录日志
            new_events = []

        elif event.event_type == EventType.JOB_START:
            # 作业开始事件，只记录日志，不产生新事件
            new_events = []

        elif event.event_type == EventType.SCHEDULING_TICK:
            new_events = self.scheduler.try_schedule(self.current_time)

        elif event.event_type == EventType.SIMULATION_END:
            # 模拟结束，不产生新事件
            new_events = []
            self.is_running = False

        else:
            raise ValueError(f"未知事件类型: {event.event_type}")

        return new_events

    def run(self) -> None:
        """
        运行模拟器主循环。

        从事件队列中按时间顺序处理事件，直到模拟结束或队列为空。
        """
        print(f"开始模拟，总时长: {self.config.duration_minutes} 分钟")
        print(f"随机种子: {self.config.random_seed}")
        print(f"调度策略: {self.config.scheduler_strategy}")

        # 初始化状态
        self.current_time = 0.0
        self.is_running = True

        # 收集初始指标快照
        self._collect_metrics_snapshot()

        # 插入初始事件
        # 1. 第一个高优作业到达事件
        first_high_event = self.generate_high_job_event()
        heapq.heappush(self.event_queue, first_high_event)

        # 2. 模拟结束事件
        end_event = Event(
            timestamp=self.config.duration_minutes,
            event_type=EventType.SIMULATION_END,
            data={"reason": "duration_reached"}
        )
        heapq.heappush(self.event_queue, end_event)

        # 主事件循环
        event_count = 0
        while self.is_running and self.event_queue:
            # 获取下一个事件
            event = heapq.heappop(self.event_queue)
            self.current_time = event.timestamp

            # 定期收集指标
            if self.current_time - self.last_sample_time >= self.config.sampling_interval:
                self._collect_metrics_snapshot()

            # 生成普通作业批量（如果是08:00）
            normal_events = self.generate_normal_batch_events()
            for e in normal_events:
                heapq.heappush(self.event_queue, e)

            # 生成机器故障事件
            failure_events = self.generate_machine_failure_events()
            for e in failure_events:
                heapq.heappush(self.event_queue, e)

            # 处理当前事件
            new_events = self._handle_event(event)

            # 将新事件加入队列
            for e in new_events:
                heapq.heappush(self.event_queue, e)

            # 生成下一个高优作业事件（如果到达时间已到）
            if self.current_time >= self.next_high_arrival_time:
                next_high_event = self.generate_high_job_event()
                heapq.heappush(self.event_queue, next_high_event)

            event_count += 1
            if event_count % 1000 == 0:
                print(f"[{self.current_time:.2f}] 已处理 {event_count} 个事件，队列长度: {len(self.event_queue)}")

        # 模拟结束
        print(f"模拟结束，总处理事件数: {event_count}")
        print(f"最终时间: {self.current_time:.2f}")
        print(f"事件日志记录数: {len(self.event_logs)}")
        print(f"指标快照数: {len(self.metrics_snapshots)}")

    def get_results(self) -> Tuple[List[EventLogEntry], List[MetricsSnapshot], List[JobLifecycleRecord]]:
        """
        获取模拟结果。

        返回:
            (事件日志列表, 指标快照列表, 作业记录列表)
        """
        job_records = self.scheduler.get_job_records()
        return self.event_logs, self.metrics_snapshots, job_records


# ============================================================================
# 占位调度器（用于测试）
# ============================================================================

class DummyScheduler(IScheduler):
    """
    占位调度器，用于测试模拟器框架。

    不实现实际调度逻辑，所有方法返回空列表，指标返回默认值。
    """

    def handle_job_submit(self, event: Event) -> List[Event]:
        print(f"[DummyScheduler] 处理作业提交: job_id={event.data.get('job_id')}")
        return []

    def handle_job_completion(self, event: Event) -> List[Event]:
        print(f"[DummyScheduler] 处理作业完成: job_id={event.data.get('job_id')}")
        return []

    def handle_machine_failure(self, event: Event) -> List[Event]:
        print(f"[DummyScheduler] 处理机器故障: machine_id={event.data.get('machine_id')}")
        return []

    def handle_machine_repair(self, event: Event) -> List[Event]:
        print(f"[DummyScheduler] 处理机器维修: machine_id={event.data.get('machine_id')}")
        return []

    def try_schedule(self, current_time: float) -> List[Event]:
        print(f"[DummyScheduler] 尝试调度 @ {current_time:.2f}")
        return []

    def get_metrics_snapshot(self, current_time: float) -> MetricsSnapshot:
        return MetricsSnapshot(
            timestamp=current_time,
            total_machines=100,
            busy_machines=0,
            idle_machines=100,
            down_machines=0,
            utilization=0.0,
            pending_normal_count=0,
            pending_high_count=0,
            running_normal_count=0,
            running_high_count=0,
            starving_jobs=0,
            max_wait_time=0.0,
            completed_normal=0,
            completed_high=0
        )

    def get_job_records(self) -> List[JobLifecycleRecord]:
        return []


# ============================================================================
# 测试函数
# ============================================================================

def test_simulator():
    """测试模拟器基本功能"""
    print("=" * 60)
    print("测试模拟器基本功能")
    print("=" * 60)

    # 创建测试配置（缩短模拟时间）
    config = SimulatorConfig(
        machine_count=10,           # 减少机器数，加速测试
        normal_batch_size=5,        # 减少批量大小
        duration_minutes=480,       # 只模拟8小时
        random_seed=42,             # 固定随机种子
        sampling_interval=60        # 每分钟采样一次
    )

    # 创建占位调度器
    scheduler = DummyScheduler()

    # 创建模拟器并运行
    simulator = Simulator(config, scheduler)
    simulator.run()

    # 获取结果
    event_logs, metrics_snapshots, job_records = simulator.get_results()

    print(f"\n测试结果统计:")
    print(f"- 事件日志数量: {len(event_logs)}")
    print(f"- 指标快照数量: {len(metrics_snapshots)}")
    print(f"- 作业记录数量: {len(job_records)}")

    # 显示前几个事件
    if event_logs:
        print(f"\n前5个事件日志:")
        for i, log in enumerate(event_logs[:5]):
            print(f"  {i+1}. t={log.timestamp:.2f}, type={log.event_type}, "
                  f"job={log.job_id}, machine={log.machine_id}")

    # 显示指标快照
    if metrics_snapshots:
        print(f"\n指标快照示例（最后一个）:")
        snapshot = metrics_snapshots[-1]
        print(f"  t={snapshot.timestamp:.2f}, util={snapshot.utilization:.1%}, "
              f"pending=[N:{snapshot.pending_normal_count}, H:{snapshot.pending_high_count}]")

    print("\n模拟器测试完成！")


if __name__ == "__main__":
    test_simulator()
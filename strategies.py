"""
调度策略实现模块。

包含：
1. FIFO（先来先服务）基线策略
2. HRRN+抢占核心策略（响应比高者优先 + 最小沉没成本抢占）
3. First Fit 机器分配算法
"""

from typing import List, Optional, Dict, Tuple, Any
from abc import ABC, abstractmethod
import heapq

from models import JobStatus, EventType


# ============================================================================
# 策略接口
# ============================================================================

class ISchedulingStrategy(ABC):
    """
    调度策略抽象接口。

    所有具体策略需实现此接口，定义如何选择作业、分配机器和处理抢占。
    """

    @abstractmethod
    def select_job(self, pending_jobs: List[Dict[str, Any]],
                   current_time: float) -> Optional[Dict[str, Any]]:
        """
        从等待队列中选择下一个要调度的作业。

        参数:
            pending_jobs: 等待作业列表，每个元素是包含作业属性的字典
            current_time: 当前时间戳

        返回:
            选择的作业字典，或 None（如果队列为空）
        """
        pass

    @abstractmethod
    def allocate_machines(self, job: Dict[str, Any],
                          machine_statuses: List[str]) -> Optional[List[int]]:
        """
        为作业分配机器（First Fit 算法）。

        参数:
            job: 作业属性字典，必须包含 resource_num
            machine_statuses: 机器状态列表，长度 = 机器总数，元素为 'idle'/'busy'/'down'

        返回:
            分配的机器ID列表（连续块），或 None（分配失败）
        """
        pass

    @abstractmethod
    def should_preempt(self, high_job: Dict[str, Any],
                       running_normal_jobs: List[Dict[str, Any]],
                       current_time: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        判断是否需要为高优作业触发抢占，并选择受害者。

        参数:
            high_job: 需要调度的高优作业
            running_normal_jobs: 正在运行的普通作业列表
            current_time: 当前时间戳

        返回:
            (should_preempt, victim_job) 元组:
            - should_preempt: 是否需要抢占
            - victim_job: 如果抢占，选择的受害者作业（None表示不抢占）
        """
        pass

    @abstractmethod
    def update_job_after_preempt(self, victim_job: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        """
        更新被抢占作业的状态和属性。

        参数:
            victim_job: 被抢占的作业
            current_time: 当前时间戳

        返回:
            更新后的作业字典（extra_time_due_to_preempt +10）
        """
        pass

    @abstractmethod
    def calculate_hrrn_score(self, job: Dict[str, Any],
                             current_time: float) -> float:
        """
        计算作业的 HRRN（响应比）得分。

        HRRN 公式：R = (等待时间 + 预估时间) / 预估时间
        等待时间 = current_time - submit_time + extra_time_due_to_preempt

        参数:
            job: 作业属性字典
            current_time: 当前时间戳

        返回:
            HRRN 得分（越高表示越应该优先调度）
        """
        pass


# ============================================================================
# First Fit 机器分配（策略共享）
# ============================================================================

def allocate_machines_first_fit(job: Dict[str, Any],
                                machine_statuses: List[str]) -> Optional[List[int]]:
    """
    First Fit 机器分配算法：查找第一个足够大的连续空闲块。

    参数:
        job: 作业属性字典，必须包含 resource_num
        machine_statuses: 机器状态列表

    返回:
        分配的机器ID列表（连续块），或 None（分配失败）
    """
    required = job["resource_num"]
    total_machines = len(machine_statuses)

    consecutive_idle = 0
    start_idx = -1

    for i in range(total_machines):
        if machine_statuses[i] == "idle":
            if consecutive_idle == 0:
                start_idx = i  # 记录连续空闲块的起始位置
            consecutive_idle += 1

            if consecutive_idle == required:
                # 找到足够大的连续空闲块
                return list(range(start_idx, start_idx + required))
        else:
            # 遇到非空闲机器，重置计数器
            consecutive_idle = 0

    # 没有找到足够大的连续空闲块
    return None


# ============================================================================
# FIFO 策略（基线）
# ============================================================================

class FIFOStrategy(ISchedulingStrategy):
    """
    先来先服务（FIFO）基线策略。

    特点：
    1. 按提交时间顺序选择作业（最早提交的优先）
    2. 不允许抢占
    3. 简单但公平性差，高优作业延迟高
    """

    def __init__(self):
        self.name = "FIFO"
        self.preemption_enabled = False  # FIFO 不允许抢占

    def select_job(self, pending_jobs: List[Dict[str, Any]],
                   current_time: float) -> Optional[Dict[str, Any]]:
        """
        FIFO 选择：按 submit_time 升序排序，选择最早的作业。
        """
        if not pending_jobs:
            return None

        # 按提交时间排序（最早的优先）
        sorted_jobs = sorted(pending_jobs, key=lambda j: j["submit_time"])
        return sorted_jobs[0]

    def allocate_machines(self, job: Dict[str, Any],
                          machine_statuses: List[str]) -> Optional[List[int]]:
        """
        使用共享的 First Fit 算法分配机器。
        """
        return allocate_machines_first_fit(job, machine_statuses)

    def should_preempt(self, high_job: Dict[str, Any],
                       running_normal_jobs: List[Dict[str, Any]],
                       current_time: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        FIFO 策略不允许抢占。
        """
        return False, None

    def update_job_after_preempt(self, victim_job: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        """
        FIFO 策略不发生抢占，此方法不应被调用。
        """
        raise NotImplementedError("FIFO策略不支持抢占")

    def calculate_hrrn_score(self, job: Dict[str, Any],
                             current_time: float) -> float:
        """
        FIFO 策略不使用 HRRN，返回 0。
        """
        return 0.0

    def __repr__(self) -> str:
        return f"FIFOStrategy(name={self.name})"


# ============================================================================
# HRRN + 抢占策略（核心）
# ============================================================================

class HRRNPreemptStrategy(ISchedulingStrategy):
    """
    HRRN（响应比高者优先）+ 抢占核心策略。

    特点：
    1. 普通作业按 HRRN 响应比排序（防止饥饿）
    2. 高优作业绝对优先（FCFS 在高优作业内部）
    3. 允许抢占：高优作业无机器时，抢占最小沉没成本的普通作业
    4. 抢占成本：每次抢占增加 10 分钟额外执行时间
    """

    def __init__(self, preemption_cost: float = 10.0):
        """
        初始化 HRRN+抢占策略。

        参数:
            preemption_cost: 每次抢占的额外成本（分钟）
        """
        self.name = "HRRN_Preempt"
        self.preemption_enabled = True
        self.preemption_cost = preemption_cost

    def select_job(self, pending_jobs: List[Dict[str, Any]],
                   current_time: float) -> Optional[Dict[str, Any]]:
        """
        选择策略：高优作业绝对优先，普通作业按 HRRN 排序。

        规则：
        1. 优先选择高优作业（按提交时间 FIFO）
        2. 如果没有高优作业，选择 HRRN 得分最高的普通作业
        """
        if not pending_jobs:
            return None

        # 分离高优和普通作业
        high_jobs = [j for j in pending_jobs if j["priority"] == "high"]
        normal_jobs = [j for j in pending_jobs if j["priority"] == "normal"]

        # 规则1：优先调度高优作业（按提交时间 FIFO）
        if high_jobs:
            # 高优作业内部按 FIFO 排序
            return min(high_jobs, key=lambda j: j["submit_time"])

        # 规则2：普通作业按 HRRN 排序
        if normal_jobs:
            # 计算每个作业的 HRRN 得分
            scored_jobs = []
            for job in normal_jobs:
                score = self.calculate_hrrn_score(job, current_time)
                scored_jobs.append((score, job))

            # 选择 HRRN 得分最高的作业
            scored_jobs.sort(reverse=True, key=lambda x: x[0])
            return scored_jobs[0][1]

        return None

    def allocate_machines(self, job: Dict[str, Any],
                          machine_statuses: List[str]) -> Optional[List[int]]:
        """
        使用共享的 First Fit 算法分配机器。
        """
        return allocate_machines_first_fit(job, machine_statuses)

    def should_preempt(self, high_job: Dict[str, Any],
                       running_normal_jobs: List[Dict[str, Any]],
                       current_time: float) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        抢占决策：高优作业需要机器时，选择最小沉没成本的普通作业。

        沉没成本 = 已执行时间 = current_time - start_time

        参数:
            high_job: 需要调度的高优作业
            running_normal_jobs: 正在运行的普通作业列表

        返回:
            (True, victim_job) 如果需要抢占，否则 (False, None)
        """
        if not running_normal_jobs:
            return False, None

        # 计算每个运行中普通作业的沉没成本
        victims = []
        for job in running_normal_jobs:
            if job["priority"] == "normal" and job["status"] == JobStatus.RUNNING:
                start_time = job.get("start_time")
                if start_time is not None:
                    sunk_cost = current_time - start_time
                    victims.append((sunk_cost, job))

        if not victims:
            return False, None

        # 选择沉没成本最小的作业作为受害者
        victims.sort(key=lambda x: x[0])  # 按沉没成本升序排序
        return True, victims[0][1]

    def update_job_after_preempt(self, victim_job: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        """
        更新被抢占作业：增加额外成本，重置状态。

        关键操作：
        1. extra_time_due_to_preempt += preemption_cost
        2. preempt_count += 1
        3. 状态回退为 PENDING
        4. 清除 start_time（下次重新开始）
        """
        # 创建副本以避免修改原始数据
        updated_job = victim_job.copy()

        # 增加抢占额外成本
        current_extra = updated_job.get("extra_time_due_to_preempt", 0.0)
        updated_job["extra_time_due_to_preempt"] = current_extra + self.preemption_cost

        # 增加抢占计数
        current_preempts = updated_job.get("preempt_count", 0)
        updated_job["preempt_count"] = current_preempts + 1

        # 重置状态
        updated_job["status"] = JobStatus.PREEMPTED
        # 注意：start_time 在调度器释放机器后会被清除

        return updated_job

    def calculate_hrrn_score(self, job: Dict[str, Any],
                             current_time: float) -> float:
        """
        计算 HRRN 响应比：R = (等待时间 + 预估时间) / 预估时间

        等待时间 = (current_time - submit_time) + extra_time_due_to_preempt

        参数:
            job: 作业属性字典，必须包含 submit_time, estimated_time
            current_time: 当前时间戳

        返回:
            HRRN 得分（越高表示越应该优先调度）
        """
        submit_time = job["submit_time"]
        estimated_time = job["estimated_time"]
        extra_time = job.get("extra_time_due_to_preempt", 0.0)

        # 等待时间 = 当前时间 - 提交时间 + 抢占额外成本
        wait_time = (current_time - submit_time) + extra_time

        # HRRN 公式
        return (wait_time + estimated_time) / estimated_time

    def __repr__(self) -> str:
        return f"HRRNPreemptStrategy(name={self.name}, preemption_cost={self.preemption_cost})"


# ============================================================================
# 策略工厂
# ============================================================================

class StrategyFactory:
    """
    策略工厂，根据名称创建对应的调度策略实例。
    """

    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> ISchedulingStrategy:
        """
        创建调度策略实例。

        参数:
            strategy_name: 策略名称，支持 'fifo' 或 'hrrn_preempt'
            **kwargs: 传递给策略构造函数的额外参数

        返回:
            调度策略实例

        异常:
            ValueError: 如果策略名称不支持
        """
        strategy_name = strategy_name.lower()

        if strategy_name == "fifo":
            return FIFOStrategy()

        elif strategy_name == "hrrn_preempt" or strategy_name == "hrrn":
            preemption_cost = kwargs.get("preemption_cost", 10.0)
            return HRRNPreemptStrategy(preemption_cost=preemption_cost)

        else:
            raise ValueError(f"不支持的策略名称: {strategy_name}. 支持: 'fifo', 'hrrn_preempt'")


# ============================================================================
# 测试函数
# ============================================================================

def test_strategies():
    """测试策略实现"""
    print("=" * 60)
    print("测试调度策略")
    print("=" * 60)

    # 创建测试作业
    test_jobs = [
        {
            "job_id": 1,
            "priority": "normal",
            "submit_time": 100.0,
            "estimated_time": 60.0,
            "resource_num": 2,
            "status": JobStatus.PENDING,
            "extra_time_due_to_preempt": 0.0
        },
        {
            "job_id": 2,
            "priority": "high",
            "submit_time": 110.0,
            "estimated_time": 30.0,
            "resource_num": 1,
            "status": JobStatus.PENDING,
            "extra_time_due_to_preempt": 0.0
        },
        {
            "job_id": 3,
            "priority": "normal",
            "submit_time": 90.0,  # 最早提交
            "estimated_time": 120.0,
            "resource_num": 3,
            "status": JobStatus.PENDING,
            "extra_time_due_to_preempt": 20.0  # 已被抢占过
        }
    ]

    # 测试机器状态
    machine_statuses = ["idle", "idle", "busy", "idle", "idle", "down"]

    # 测试 FIFO 策略
    print("\n1. 测试 FIFO 策略:")
    fifo = StrategyFactory.create_strategy("fifo")
    selected = fifo.select_job(test_jobs, current_time=150.0)
    print(f"  选择的作业: job_id={selected['job_id'] if selected else None}")

    # 测试机器分配
    allocation = fifo.allocate_machines(test_jobs[0], machine_statuses)
    print(f"  机器分配结果: {allocation}")

    # 测试 HRRN 策略
    print("\n2. 测试 HRRN+抢占策略:")
    hrrn = StrategyFactory.create_strategy("hrrn_preempt", preemption_cost=10.0)

    # 测试作业选择
    selected = hrrn.select_job(test_jobs, current_time=150.0)
    print(f"  选择的作业: job_id={selected['job_id'] if selected else None}")

    # 测试 HRRN 计算
    for job in test_jobs:
        if job["priority"] == "normal":
            score = hrrn.calculate_hrrn_score(job, current_time=150.0)
            print(f"  作业 {job['job_id']} HRRN 得分: {score:.3f}")

    # 测试抢占决策
    running_jobs = [
        {
            "job_id": 101,
            "priority": "normal",
            "start_time": 140.0,
            "estimated_time": 60.0,
            "status": JobStatus.RUNNING
        },
        {
            "job_id": 102,
            "priority": "normal",
            "start_time": 130.0,  # 更早开始，沉没成本更大
            "estimated_time": 90.0,
            "status": JobStatus.RUNNING
        }
    ]

    high_job = {
        "job_id": 201,
        "priority": "high",
        "resource_num": 2
    }

    should_preempt, victim = hrrn.should_preempt(
        high_job, running_jobs, current_time=150.0
    )
    print(f"  抢占决策: should_preempt={should_preempt}")
    if victim:
        print(f"  受害者: job_id={victim['job_id']}, 沉没成本={150.0 - victim['start_time']:.1f}")

    # 测试抢占后更新
    if victim:
        updated = hrrn.update_job_after_preempt(victim, current_time=150.0)
        print(f"  更新后作业: extra_time={updated['extra_time_due_to_preempt']}, "
              f"preempt_count={updated.get('preempt_count', 0)}")

    # 测试策略工厂
    print("\n3. 测试策略工厂:")
    strategies = ["fifo", "hrrn_preempt"]
    for name in strategies:
        strategy = StrategyFactory.create_strategy(name)
        print(f"  {name}: {strategy}")

    print("\n策略测试完成！")


if __name__ == "__main__":
    test_strategies()
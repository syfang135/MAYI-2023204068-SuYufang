"""
集成测试与场景执行器。

包含5个验收场景：
1. 基线对比（FIFO vs HRRN+抢占）
2. 波峰场景（08:00瞬间1000个批量）
3. 故障场景（注入宕机事件）
4. 抢占场景（高频插入high作业）
5. 可复现性验证（相同random_seed跑两次）

支持两种规模：mini（快速验证）和 full（完整评估）。
"""

import os
import sys
import time
import json
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import asdict

from simulator import Simulator, SimulatorConfig, DummyScheduler
from scheduler import Scheduler
from models import EventLogEntry, MetricsSnapshot, JobLifecycleRecord, FinalMetricsReport
from visualization import generate_all_plots, calculate_final_metrics, ensure_results_dir


# ============================================================================
# 配置工厂
# ============================================================================

def create_config(scale: str = "mini", **kwargs) -> SimulatorConfig:
    """
    创建模拟器配置。

    参数:
        scale: "mini"（快速测试）或 "full"（完整评估）
        **kwargs: 覆盖默认配置

    返回:
        SimulatorConfig 对象
    """
    if scale == "mini":
        # 迷你规模：快速验证
        base_config = {
            "machine_count": 10,                # 机器总数
            "failure_prob_per_day": 0.001,      # 故障概率
            "normal_batch_size": 20,            # 普通作业批量大小
            "high_arrival_rate": 2.0,           # 高优作业到达率（个/小时）
            "duration_minutes": 480,            # 模拟时长（8小时）
            "random_seed": 42,                  # 随机种子
            "sampling_interval": 30,            # 采样间隔（30分钟）
            "preemption_enabled": True,         # 是否允许抢占
            "preemption_cost": 10.0,            # 抢占额外成本
            "starvation_threshold": 180.0,      # 饥饿阈值（3小时）
        }
    else:  # scale == "full"
        # 完整规模：生产评估
        base_config = {
            "machine_count": 100,               # 机器总数
            "failure_prob_per_day": 0.001,      # 故障概率
            "normal_batch_size": 500,           # 普通作业批量大小
            "high_arrival_rate": 4.0,           # 高优作业到达率（个/小时）
            "duration_minutes": 10080,          # 模拟时长（7天）
            "random_seed": None,                # 随机种子（None表示随机）
            "sampling_interval": 10,            # 采样间隔（10分钟）
            "preemption_enabled": True,         # 是否允许抢占
            "preemption_cost": 10.0,            # 抢占额外成本
            "starvation_threshold": 720.0,      # 饥饿阈值（12小时）
        }

    # 应用自定义参数
    base_config.update(kwargs)

    return SimulatorConfig(**base_config)


def create_scheduler(config: SimulatorConfig, strategy_name: str) -> Scheduler:
    """
    创建调度器实例。

    参数:
        config: 模拟器配置
        strategy_name: 调度策略名称

    返回:
        Scheduler 实例
    """
    return Scheduler(config, strategy_name)


# ============================================================================
# 场景定义
# ============================================================================

def scenario_baseline_comparison(scale: str = "mini") -> Dict[str, Tuple[List[EventLogEntry],
                                                                        List[MetricsSnapshot],
                                                                        List[JobLifecycleRecord]]]:
    """
    场景1: 基线对比。

    对比 FIFO 和 HRRN+抢占 两种策略的性能差异。

    参数:
        scale: "mini" 或 "full"

    返回:
        策略名称到(事件日志, 指标快照, 作业记录)的映射
    """
    print("\n" + "="*60)
    print("场景1: 基线对比 (FIFO vs HRRN+抢占)")
    print("="*60)

    results = {}

    # 测试两种策略
    strategies = ["fifo", "hrrn_preempt"]

    for strategy in strategies:
        print(f"\n运行 {strategy.upper()} 策略...")
        start_time = time.time()

        # 创建配置和调度器
        config = create_config(scale, scheduler_strategy=strategy)
        scheduler = create_scheduler(config, strategy)

        # 创建模拟器并运行
        simulator = Simulator(config, scheduler)
        simulator.run()

        # 获取结果
        event_logs, metrics_snapshots, job_records = simulator.get_results()
        results[strategy] = (event_logs, metrics_snapshots, job_records)

        elapsed = time.time() - start_time
        print(f"  {strategy.upper()} 策略运行完成，耗时 {elapsed:.1f} 秒")
        print(f"  事件日志: {len(event_logs)} 条")
        print(f"  指标快照: {len(metrics_snapshots)} 条")
        print(f"  作业记录: {len(job_records)} 条")

    return results


def scenario_peak_load(scale: str = "mini") -> Dict[str, Tuple[List[EventLogEntry],
                                                              List[MetricsSnapshot],
                                                              List[JobLifecycleRecord]]]:
    """
    场景2: 波峰场景。

    测试 08:00 瞬间到达 1000 个批量作业时系统的消化能力。
    使用 HRRN+抢占策略。

    参数:
        scale: "mini" 或 "full"

    返回:
        策略名称到(事件日志, 指标快照, 作业记录)的映射
    """
    print("\n" + "="*60)
    print("场景2: 波峰场景 (08:00瞬间1000个批量)")
    print("="*60)

    # 调整配置：增加批量大小，缩短模拟时间以观察队列消化
    if scale == "mini":
        batch_size = 50
        duration = 960  # 16小时（包含两个08:00）
    else:
        batch_size = 1000
        duration = 2880  # 48小时（包含三个08:00）

    config = create_config(
        scale,
        scheduler_strategy="hrrn_preempt",
        normal_batch_size=batch_size,
        duration_minutes=duration,
        high_arrival_rate=2.0,  # 降低高优作业到达率，专注于普通作业消化
    )

    print(f"配置: 机器数={config.machine_count}, 批量大小={config.normal_batch_size}")
    print(f"      模拟时长={config.duration_minutes/60:.1f}小时")

    start_time = time.time()

    scheduler = create_scheduler(config, "hrrn_preempt")
    simulator = Simulator(config, scheduler)
    simulator.run()

    event_logs, metrics_snapshots, job_records = simulator.get_results()

    elapsed = time.time() - start_time
    print(f"运行完成，耗时 {elapsed:.1f} 秒")
    print(f"事件日志: {len(event_logs)} 条")
    print(f"指标快照: {len(metrics_snapshots)} 条")
    print(f"作业记录: {len(job_records)} 条")

    # 分析波峰消化情况
    if metrics_snapshots:
        peak_queue = max(snapshot.pending_normal_count + snapshot.pending_high_count
                        for snapshot in metrics_snapshots)
        final_queue = metrics_snapshots[-1].pending_normal_count + metrics_snapshots[-1].pending_high_count

        print(f"波峰分析: 最大队列长度={peak_queue}, 最终队列长度={final_queue}")
        if final_queue == 0:
            print("  ✅ 队列完全消化")
        elif final_queue < peak_queue * 0.1:
            print(f"  ✅ 队列消化良好 ({final_queue}/{peak_queue})")
        else:
            print(f"  ⚠️  队列消化较慢 ({final_queue}/{peak_queue})")

    return {"hrrn_preempt_peak": (event_logs, metrics_snapshots, job_records)}


def scenario_failure_injection(scale: str = "mini") -> Dict[str, Tuple[List[EventLogEntry],
                                                                      List[MetricsSnapshot],
                                                                      List[JobLifecycleRecord]]]:
    """
    场景3: 故障场景。

    注入宕机事件，观察系统的恢复能力。
    增加故障概率，观察故障恢复曲线。

    参数:
        scale: "mini" 或 "full"

    返回:
        策略名称到(事件日志, 指标快照, 作业记录)的映射
    """
    print("\n" + "="*60)
    print("场景3: 故障场景 (注入宕机事件)")
    print("="*60)

    # 增加故障概率，缩短维修时间以便观察
    if scale == "mini":
        failure_prob = 0.01  # 1% 每天
        duration = 1440  # 24小时
    else:
        failure_prob = 0.005  # 0.5% 每天
        duration = 4320  # 72小时

    config = create_config(
        scale,
        scheduler_strategy="hrrn_preempt",
        failure_prob_per_day=failure_prob,
        duration_minutes=duration,
        # 注：实际维修时间在代码中固定为24小时，这里无法配置
    )

    print(f"配置: 故障概率={config.failure_prob_per_day*100:.1f}%/天")
    print(f"      模拟时长={config.duration_minutes/24/60:.1f}天")

    start_time = time.time()

    scheduler = create_scheduler(config, "hrrn_preempt")
    simulator = Simulator(config, scheduler)
    simulator.run()

    event_logs, metrics_snapshots, job_records = simulator.get_results()

    elapsed = time.time() - start_time
    print(f"运行完成，耗时 {elapsed:.1f} 秒")

    # 统计故障事件
    failure_events = [e for e in event_logs if e.event_type == "machine_failure"]
    repair_events = [e for e in event_logs if e.event_type == "machine_repair"]

    print(f"故障统计: 故障事件={len(failure_events)} 次, 维修事件={len(repair_events)} 次")

    if failure_events:
        # 计算平均故障间隔
        failure_times = sorted([e.timestamp for e in failure_events])
        if len(failure_times) > 1:
            intervals = [failure_times[i+1] - failure_times[i] for i in range(len(failure_times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            print(f"故障分析: 平均故障间隔={avg_interval/60:.1f} 小时")

    return {"hrrn_preempt_failure": (event_logs, metrics_snapshots, job_records)}


def scenario_preemption_intensive(scale: str = "mini") -> Dict[str, Tuple[List[EventLogEntry],
                                                                        List[MetricsSnapshot],
                                                                        List[JobLifecycleRecord]]]:
    """
    场景4: 抢占场景。

    高频插入高优作业，观察抢占行为对普通作业的影响。
    增加高优作业到达率，减少机器数以制造资源竞争。

    参数:
        scale: "mini" 或 "full"

    返回:
        策略名称到(事件日志, 指标快照, 作业记录)的映射
    """
    print("\n" + "="*60)
    print("场景4: 抢占场景 (高频插入高优作业)")
    print("="*60)

    # 调整配置：增加高优作业到达率，减少可用机器数
    if scale == "mini":
        machine_count = 5
        high_arrival_rate = 10.0  # 10个/小时
        duration = 480  # 8小时
    else:
        machine_count = 50
        high_arrival_rate = 20.0  # 20个/小时
        duration = 1440  # 24小时

    config = create_config(
        scale,
        scheduler_strategy="hrrn_preempt",
        machine_count=machine_count,
        high_arrival_rate=high_arrival_rate,
        duration_minutes=duration,
        preemption_enabled=True,
    )

    print(f"配置: 机器数={config.machine_count}")
    print(f"      高优作业到达率={config.high_arrival_rate} 个/小时")
    print(f"      模拟时长={config.duration_minutes/60:.1f} 小时")

    start_time = time.time()

    scheduler = create_scheduler(config, "hrrn_preempt")
    simulator = Simulator(config, scheduler)
    simulator.run()

    event_logs, metrics_snapshots, job_records = simulator.get_results()

    elapsed = time.time() - start_time
    print(f"运行完成，耗时 {elapsed:.1f} 秒")

    # 统计抢占事件
    preempt_events = [e for e in event_logs if e.event_type == "job_preempt"]
    print(f"抢占统计: 抢占事件={len(preempt_events)} 次")

    # 分析被抢占的作业
    if job_records:
        preempted_jobs = [j for j in job_records if j.preempt_count > 0]
        if preempted_jobs:
            avg_preempts = sum(j.preempt_count for j in preempted_jobs) / len(preempted_jobs)
            max_preempts = max(j.preempt_count for j in preempted_jobs)
            total_extra_time = sum(j.extra_time_due_to_preempt for j in preempted_jobs)

            print(f"作业分析: 被抢占作业数={len(preempted_jobs)}")
            print(f"         平均抢占次数={avg_preempts:.1f} 次/作业")
            print(f"         最大抢占次数={max_preempts} 次")
            print(f"         总额外成本={total_extra_time:.1f} 分钟")

    return {"hrrn_preempt_intensive": (event_logs, metrics_snapshots, job_records)}


def scenario_reproducibility(scale: str = "mini") -> bool:
    """
    场景5: 可复现性验证。

    使用相同的 random_seed 运行两次模拟，断言指标完全一致。

    参数:
        scale: "mini" 或 "full"

    返回:
        验证是否通过 (True/False)
    """
    print("\n" + "="*60)
    print("场景5: 可复现性验证 (相同random_seed跑两次)")
    print("="*60)

    # 使用固定的随机种子
    random_seed = 12345

    # 第一次运行
    print("第一次运行...")
    config1 = create_config(scale, random_seed=random_seed, scheduler_strategy="hrrn_preempt")
    scheduler1 = create_scheduler(config1, "hrrn_preempt")
    simulator1 = Simulator(config1, scheduler1)
    simulator1.run()
    _, metrics1, jobs1 = simulator1.get_results()

    # 第二次运行（相同配置）
    print("第二次运行...")
    config2 = create_config(scale, random_seed=random_seed, scheduler_strategy="hrrn_preempt")
    scheduler2 = create_scheduler(config2, "hrrn_preempt")
    simulator2 = Simulator(config2, scheduler2)
    simulator2.run()
    _, metrics2, jobs2 = simulator2.get_results()

    # 比较指标快照
    metrics_match = True
    if len(metrics1) != len(metrics2):
        print(f"❌ 指标快照数量不一致: {len(metrics1)} vs {len(metrics2)}")
        metrics_match = False
    else:
        for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
            if m1.timestamp != m2.timestamp:
                print(f"❌ 第{i}个快照时间戳不一致: {m1.timestamp} vs {m2.timestamp}")
                metrics_match = False
            if m1.utilization != m2.utilization:
                print(f"❌ 第{i}个快照利用率不一致: {m1.utilization} vs {m2.utilization}")
                metrics_match = False

    # 比较作业记录
    jobs_match = True
    if len(jobs1) != len(jobs2):
        print(f"❌ 作业记录数量不一致: {len(jobs1)} vs {len(jobs2)}")
        jobs_match = False
    else:
        # 按job_id排序后比较
        jobs1_sorted = sorted(jobs1, key=lambda j: j.job_id)
        jobs2_sorted = sorted(jobs2, key=lambda j: j.job_id)

        for j1, j2 in zip(jobs1_sorted, jobs2_sorted):
            if j1.job_id != j2.job_id:
                print(f"❌ 作业ID不一致: {j1.job_id} vs {j2.job_id}")
                jobs_match = False
            if j1.wait_time != j2.wait_time:
                print(f"❌ 作业{j1.job_id}等待时间不一致: {j1.wait_time} vs {j2.wait_time}")
                jobs_match = False

    # 验证结果
    if metrics_match and jobs_match:
        print("✅ 可复现性验证通过！两次运行结果完全一致。")
        return True
    else:
        print("❌ 可复现性验证失败！两次运行结果不一致。")
        return False


# ============================================================================
# 主执行器
# ============================================================================

def run_all_scenarios(scale: str = "mini", generate_plots: bool = True) -> Dict[str, Any]:
    """
    运行所有验收场景。

    参数:
        scale: "mini"（快速验证）或 "full"（完整评估）
        generate_plots: 是否生成可视化图表

    返回:
        包含所有场景结果的字典
    """
    print("\n" + "="*60)
    print(f"开始运行所有验收场景 (规模: {scale})")
    print("="*60)

    start_time = time.time()
    all_results = {}

    # 场景1: 基线对比
    baseline_results = scenario_baseline_comparison(scale)
    all_results["baseline"] = baseline_results

    # 场景2: 波峰场景
    peak_results = scenario_peak_load(scale)
    all_results["peak"] = peak_results

    # 场景3: 故障场景
    failure_results = scenario_failure_injection(scale)
    all_results["failure"] = failure_results

    # 场景4: 抢占场景
    preemption_results = scenario_preemption_intensive(scale)
    all_results["preemption"] = preemption_results

    # 场景5: 可复现性验证
    reproducibility_passed = scenario_reproducibility(scale)
    all_results["reproducibility"] = reproducibility_passed

    # 生成可视化图表
    if generate_plots:
        print("\n" + "="*60)
        print("生成可视化图表")
        print("="*60)

        # 合并所有策略的结果用于图表生成
        all_strategy_results = {}
        for scenario_name, scenario_result in all_results.items():
            if scenario_name == "reproducibility":
                continue  # 可复现性场景没有策略结果

            if isinstance(scenario_result, dict):
                for strategy_name, (event_logs, metrics_snapshots, job_records) in scenario_result.items():
                    # 为每个策略结果创建唯一名称
                    unique_name = f"{scenario_name}_{strategy_name}"
                    all_strategy_results[unique_name] = (event_logs, metrics_snapshots, job_records)

        if all_strategy_results:
            # 确定模拟时长（使用第一个配置）
            first_result = next(iter(all_strategy_results.values()))
            _, metrics_snapshots, _ = first_result
            simulation_duration = metrics_snapshots[-1].timestamp if metrics_snapshots else 0.0

            # 生成所有图表
            reports = generate_all_plots(
                all_strategy_results,
                simulation_duration=simulation_duration
            )

            # 保存报告到JSON文件
            if reports:
                results_dir = ensure_results_dir()
                report_file = os.path.join(results_dir, "final_reports.json")

                # 转换报告为可序列化格式
                serializable_reports = {}
                for strategy_name, report in reports.items():
                    # 使用asdict转换dataclass，然后处理特殊类型
                    report_dict = asdict(report)
                    # 转换时间序列为列表的列表（JSON可序列化）
                    report_dict["utilization_timeline"] = [
                        [float(t), float(u)] for t, u in report_dict["utilization_timeline"]
                    ]
                    report_dict["queue_length_timeline"] = [
                        [float(t), int(q)] for t, q in report_dict["queue_length_timeline"]
                    ]
                    serializable_reports[strategy_name] = report_dict

                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_reports, f, indent=2, ensure_ascii=False)

                print(f"最终报告已保存: {report_file}")

    # 输出总运行时间
    total_elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"所有场景运行完成！总耗时: {total_elapsed:.1f} 秒")
    print("="*60)

    return all_results


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="集群调度模拟器验收测试")
    parser.add_argument("--scale", choices=["mini", "full"], default="mini",
                       help="测试规模: mini（快速验证）或 full（完整评估）")
    parser.add_argument("--no-plots", action="store_true",
                       help="不生成可视化图表")
    parser.add_argument("--scenario", choices=["1", "2", "3", "4", "5", "all"],
                       default="all", help="运行特定场景 (1-5) 或 all")

    args = parser.parse_args()

    print("集群调度模拟器验收测试")
    print(f"测试规模: {args.scale}")
    print(f"生成图表: {not args.no_plots}")

    if args.scenario == "all":
        # 运行所有场景
        results = run_all_scenarios(
            scale=args.scale,
            generate_plots=not args.no_plots
        )

        # 输出摘要
        print("\n" + "="*60)
        print("测试摘要")
        print("="*60)

        for scenario_name, scenario_result in results.items():
            if scenario_name == "reproducibility":
                status = "✅ 通过" if scenario_result else "❌ 失败"
                print(f"场景5（可复现性）: {status}")
            else:
                print(f"{scenario_name}: 完成")

    else:
        # 运行单个场景
        scenario_funcs = {
            "1": scenario_baseline_comparison,
            "2": scenario_peak_load,
            "3": scenario_failure_injection,
            "4": scenario_preemption_intensive,
            "5": scenario_reproducibility,
        }

        if args.scenario in scenario_funcs:
            func = scenario_funcs[args.scenario]
            result = func(args.scale)

            # 如果是可复现性场景，输出结果
            if args.scenario == "5":
                if result:
                    print("✅ 可复现性验证通过")
                else:
                    print("❌ 可复现性验证失败")
        else:
            print(f"错误: 未知场景 {args.scenario}")
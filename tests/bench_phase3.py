"""
A.C.C.E.S.S. — Phase 3 Micro-Benchmarks

Measures per-operation latency for all Phase 3 components.
Reports mean, p50, p95, p99, and max across 10,000 iterations.

Targets (CI-safe thresholds):
    Reflection:        < 0.5ms  mean
    Trajectory update: < 0.2ms  mean
    Logging:           < 0.2ms  mean
    Full Phase 3 turn: < 1.0ms  mean (reflection + trajectory + logging combined)

Usage:
    python tests/bench_phase3.py
    python -m pytest tests/bench_phase3.py -v   # runs as assertions too

Output: formatted table to stdout + pass/fail assertions.
"""

import sys
import os
import time
import random
import statistics
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.reflection_engine import ReflectionEngine, ReflectionResult
from agent.trajectory import TrajectoryTracker
from agent.logger import StructuredLogger

ITERATIONS = 10_000

GOAL_SIGNALS = ["push_forward", "execute", "stabilize", "recover", "explore", None]
TRAJ_SIGNALS = ["progressing", "declining", "stable", "escalating", None]


# ─── Test fixtures ─────────────────────────────────────────────────────────────

@dataclass
class BenchState:
    primary_emotion: str = "frustration"
    intensity: float = 0.7
    is_positive: bool = False
    is_negative: bool = True
    is_high_arousal: bool = True
    label: str = "frustration"
    pad: object = None

    def __post_init__(self):
        @dataclass
        class P:
            valence: float = -0.3
            arousal: float = 0.7
            dominance: float = 0.4
        if self.pad is None:
            self.pad = P()


@dataclass
class BenchMod:
    tone: str = "grounding"
    pacing: str = "slow"
    verbosity: str = "concise"
    structure_bias: str = "conversational"
    emotional_validation: bool = True
    motivational_bias: float = 0.4
    cognitive_load_limit: float = 0.6
    active_strategies: tuple = ("burnout_shield",)


# ─── Benchmark runner ──────────────────────────────────────────────────────────

def _run_benchmark(label: str, fn, iterations: int = ITERATIONS) -> dict:
    """Run fn() `iterations` times, collect per-call timings."""
    # Warm up
    for _ in range(min(100, iterations // 10)):
        fn()

    timings_us = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        elapsed_us = (time.perf_counter() - t0) * 1_000_000  # microseconds
        timings_us.append(elapsed_us)

    timings_us.sort()
    mean = statistics.mean(timings_us)
    p50 = timings_us[int(len(timings_us) * 0.50)]
    p95 = timings_us[int(len(timings_us) * 0.95)]
    p99 = timings_us[int(len(timings_us) * 0.99)]
    mx = timings_us[-1]

    return {
        "label": label,
        "iterations": iterations,
        "mean_us": mean,
        "p50_us": p50,
        "p95_us": p95,
        "p99_us": p99,
        "max_us": mx,
        "mean_ms": mean / 1000,
    }


def _print_results(results: list[dict]) -> None:
    """Print formatted benchmark table."""
    print(f"\n{'='*80}")
    print(f"  A.C.C.E.S.S. Phase 3 — Micro-Benchmarks ({ITERATIONS:,} iterations)")
    print(f"{'='*80}")
    print(f"  {'Component':<30} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(
            f"  {r['label']:<30} "
            f"{r['mean_us']:>7.1f}μs "
            f"{r['p50_us']:>7.1f}μs "
            f"{r['p95_us']:>7.1f}μs "
            f"{r['p99_us']:>7.1f}μs "
            f"{r['max_us']:>7.1f}μs"
        )
    print(f"{'='*80}\n")


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

def bench_reflection_simple() -> dict:
    """Benchmark: ReflectionEngine.reflect() with typical inputs."""
    engine = ReflectionEngine()
    state = BenchState()
    mod = BenchMod()
    user_input = "I've been struggling with this for hours and I don't know what to do next."
    output = "Let me help you break this down step by step. First, let's identify the core issue."

    def fn():
        engine.reflect(user_input, output, state, mod, ())

    return _run_benchmark("reflection (simple)", fn)


def bench_reflection_with_tools() -> dict:
    """Benchmark: ReflectionEngine.reflect() with tool results."""
    engine = ReflectionEngine()
    state = BenchState()
    mod = BenchMod()

    @dataclass
    class FakeTR:
        tool_name: str = "calculator"
        success: bool = True
    tools = (FakeTR(), FakeTR(tool_name="echo"))

    def fn():
        engine.reflect("calc 2+2", "The result is 4.", state, mod, tools)

    return _run_benchmark("reflection (with tools)", fn)


def bench_reflection_long_output() -> dict:
    """Benchmark: ReflectionEngine.reflect() with long assistant output."""
    engine = ReflectionEngine()
    state = BenchState()
    mod = BenchMod()
    output = "x" * 5000

    def fn():
        engine.reflect("question", output, state, mod, ())

    return _run_benchmark("reflection (long output)", fn)


def bench_trajectory_update() -> dict:
    """Benchmark: TrajectoryTracker.update() steady state."""
    tracker = TrajectoryTracker(window_size=10)
    rng = random.Random(42)
    # Pre-fill to reach steady state
    for _ in range(100):
        tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))

    def fn():
        tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))

    return _run_benchmark("trajectory update", fn)


def bench_trajectory_state_access() -> dict:
    """Benchmark: TrajectoryTracker.state property access."""
    tracker = TrajectoryTracker(window_size=20)
    for _ in range(100):
        tracker.update("push_forward", "progressing")

    def fn():
        _ = tracker.state

    return _run_benchmark("trajectory .state", fn)


def bench_trajectory_to_dict() -> dict:
    """Benchmark: TrajectoryState.to_dict() serialization."""
    tracker = TrajectoryTracker(window_size=20)
    for _ in range(100):
        tracker.update("push_forward", "progressing")
    state = tracker.state

    def fn():
        state.to_dict()

    return _run_benchmark("trajectory to_dict", fn)


def bench_logger_log_event() -> dict:
    """Benchmark: StructuredLogger.log_event()."""
    log = StructuredLogger(max_events=50_000)  # avoid trim overhead during bench
    payload = {"turn_index": 1, "latency_ms": 42.5, "emotion": "neutral"}

    def fn():
        log.log_event("turn_completed", payload, session_id="s1", turn_index=1)

    return _run_benchmark("logger log_event", fn)


def bench_logger_with_sink() -> dict:
    """Benchmark: StructuredLogger.log_event() with sink callback."""
    sink_buffer = []
    log = StructuredLogger(sink=sink_buffer.append, max_events=50_000)
    payload = {"x": 1}

    def fn():
        log.log_event("test", payload)

    return _run_benchmark("logger + sink", fn)


def bench_logger_filter_by_type() -> dict:
    """Benchmark: get_logs_by_type after 1000 events."""
    log = StructuredLogger()
    for i in range(1000):
        log.log_event(f"type_{i % 5}", {"i": i})

    def fn():
        log.get_logs_by_type("type_2")

    return _run_benchmark("logger filter_by_type", fn, iterations=1000)


def bench_full_phase3_turn() -> dict:
    """Benchmark: Combined reflection + trajectory + logging (simulated turn)."""
    engine = ReflectionEngine()
    tracker = TrajectoryTracker(window_size=10)
    log = StructuredLogger(max_events=50_000)
    state = BenchState()
    mod = BenchMod()

    # Pre-fill
    for _ in range(50):
        tracker.update("push_forward", "progressing")

    def fn():
        # Step 1: Reflect
        result = engine.reflect("input", "output text here", state, mod, ())
        # Step 2: Update trajectory
        traj = tracker.update(result.goal_signal, result.trajectory_signal)
        # Step 3: Log
        log.log_event("turn_completed", {
            "importance": result.importance_score,
            "goal": result.goal_signal,
            "drift": traj.drift_score,
        })

    return _run_benchmark("full phase3 turn", fn)


# ═════════════════════════════════════════════════════════════════════════════
# PYTEST ASSERTIONS (CI-compatible)
# ═════════════════════════════════════════════════════════════════════════════

class TestBenchmarkThresholds:
    """Assertions that benchmark means stay within CI-safe budgets."""

    def test_reflection_under_500us(self):
        r = bench_reflection_simple()
        assert r["mean_ms"] < 0.5, f"Reflection mean {r['mean_ms']:.3f}ms exceeds 0.5ms"

    def test_trajectory_update_under_200us(self):
        r = bench_trajectory_update()
        assert r["mean_ms"] < 0.2, f"Trajectory update mean {r['mean_ms']:.3f}ms exceeds 0.2ms"

    def test_logging_under_200us(self):
        r = bench_logger_log_event()
        assert r["mean_ms"] < 0.2, f"Logging mean {r['mean_ms']:.3f}ms exceeds 0.2ms"

    def test_full_phase3_turn_under_1ms(self):
        r = bench_full_phase3_turn()
        assert r["mean_ms"] < 1.0, f"Full turn mean {r['mean_ms']:.3f}ms exceeds 1.0ms"


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def main():
    benchmarks = [
        bench_reflection_simple,
        bench_reflection_with_tools,
        bench_reflection_long_output,
        bench_trajectory_update,
        bench_trajectory_state_access,
        bench_trajectory_to_dict,
        bench_logger_log_event,
        bench_logger_with_sink,
        bench_logger_filter_by_type,
        bench_full_phase3_turn,
    ]

    results = []
    for bench_fn in benchmarks:
        r = bench_fn()
        results.append(r)

    _print_results(results)

    # Summary verdict
    thresholds = {
        "reflection (simple)": 0.5,
        "trajectory update": 0.2,
        "logger log_event": 0.2,
        "full phase3 turn": 1.0,
    }

    print("  Threshold checks:")
    all_pass = True
    for r in results:
        label = r["label"]
        if label in thresholds:
            budget = thresholds[label]
            status = "✅" if r["mean_ms"] < budget else "❌"
            if r["mean_ms"] >= budget:
                all_pass = False
            print(f"    {status} {label}: {r['mean_ms']:.3f}ms (budget: {budget}ms)")

    print()
    if all_pass:
        print("  All benchmarks within budget ✅")
    else:
        print("  ⚠️  Some benchmarks exceeded budget")
        sys.exit(1)


if __name__ == "__main__":
    main()
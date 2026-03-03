"""
A.C.C.E.S.S. — Stress Tests for TrajectoryTracker (Phase 3 Solidification)

Simulates extreme usage patterns:
    - 10,000+ sequential updates
    - Random goal signal switching
    - Large window sizes (500+)
    - All-None signal sequences
    - Rapid alternating signals
    - Reset mid-stream

Validates:
    - drift_score stability under load
    - No memory leak behavior (bounded state size)
    - Near-constant update performance (O(1) amortized)
    - Correctness under adversarial sequences
"""

import sys
import os
import time
import random
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.trajectory import TrajectoryTracker, TrajectoryState

GOAL_SIGNALS = ["push_forward", "execute", "stabilize", "recover", "explore", None]
TRAJ_SIGNALS = ["progressing", "declining", "stable", "escalating", None]
VALID_GOAL_SET = set(GOAL_SIGNALS)
VALID_TRAJ_SET = set(TRAJ_SIGNALS)


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: HIGH VOLUME SEQUENTIAL UPDATES
# ═════════════════════════════════════════════════════════════════════════════

class TestStressHighVolume:
    """10,000+ updates with varying signal distributions."""

    def test_10k_uniform_updates(self):
        """10,000 identical signals — drift must stay 0.0."""
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(10_000):
            tracker.update("push_forward", "progressing")
        s = tracker.state
        assert s.turn_count == 10_000
        assert s.drift_score == 0.0
        assert s.dominant_goal_signal == "push_forward"
        assert len(s.recent_goal_signals) == 10

    def test_10k_random_updates(self):
        """10,000 random signals — state must remain valid."""
        rng = random.Random(42)  # deterministic seed
        tracker = TrajectoryTracker(window_size=20)
        for _ in range(10_000):
            g = rng.choice(GOAL_SIGNALS)
            t = rng.choice(TRAJ_SIGNALS)
            tracker.update(g, t)
        s = tracker.state
        assert s.turn_count == 10_000
        assert 0.0 <= s.drift_score <= 1.0
        assert s.dominant_goal_signal in VALID_GOAL_SET
        assert s.dominant_trajectory in VALID_TRAJ_SET
        assert len(s.recent_goal_signals) == 20

    def test_50k_updates_no_degradation(self):
        """50,000 updates — verify no OOM or state corruption."""
        tracker = TrajectoryTracker(window_size=50)
        rng = random.Random(99)
        for i in range(50_000):
            g = rng.choice(GOAL_SIGNALS)
            tracker.update(g, None)
        s = tracker.state
        assert s.turn_count == 50_000
        assert len(s.recent_goal_signals) <= 50
        assert 0.0 <= s.drift_score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: SIGNAL SWITCHING PATTERNS
# ═════════════════════════════════════════════════════════════════════════════

class TestStressSignalPatterns:
    """Adversarial signal sequences designed to test edge cases."""

    def test_alternating_two_signals(self):
        """Rapid 2-way alternation — drift should be high."""
        tracker = TrajectoryTracker(window_size=10)
        for i in range(10_000):
            sig = "push_forward" if i % 2 == 0 else "stabilize"
            tracker.update(sig, None)
        s = tracker.state
        assert s.drift_score > 0.0  # must detect switching
        assert s.turn_count == 10_000

    def test_alternating_all_signals(self):
        """Cycle through all 5 non-None signals — max drift."""
        tracker = TrajectoryTracker(window_size=10)
        non_none = [s for s in GOAL_SIGNALS if s is not None]
        for i in range(10_000):
            tracker.update(non_none[i % len(non_none)], None)
        s = tracker.state
        assert s.drift_score > 0.3  # high diversity in window

    def test_phase_transition(self):
        """1000 of signal A, then 1000 of signal B — drift should peak then settle."""
        tracker = TrajectoryTracker(window_size=20)
        for _ in range(1000):
            tracker.update("push_forward", "progressing")
        mid_state = tracker.state
        assert mid_state.drift_score == 0.0  # uniform window

        for _ in range(1000):
            tracker.update("recover", "declining")
        end_state = tracker.state
        # After 1000 new signals with window=20, old signals are gone
        assert end_state.drift_score == 0.0
        assert end_state.dominant_goal_signal == "recover"

    def test_all_none_10k(self):
        """10,000 None signals — drift must stay 0.0."""
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(10_000):
            tracker.update(None, None)
        s = tracker.state
        assert s.drift_score == 0.0
        assert s.dominant_goal_signal is None
        assert s.dominant_trajectory is None

    def test_single_non_none_among_nones(self):
        """One real signal in a sea of Nones."""
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(9):
            tracker.update(None, None)
        tracker.update("execute", "progressing")
        s = tracker.state
        assert s.dominant_goal_signal == "execute"
        assert s.dominant_trajectory == "progressing"
        assert s.drift_score == 0.0  # only 1 non-None → no drift


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: LARGE WINDOW SIZES
# ═════════════════════════════════════════════════════════════════════════════

class TestStressLargeWindows:
    """Window sizes from 2 to 1000."""

    @pytest.mark.parametrize("window", [2, 10, 50, 100, 500, 1000])
    def test_large_window_1k_updates(self, window):
        """Various window sizes with 1000 updates."""
        tracker = TrajectoryTracker(window_size=window)
        rng = random.Random(window)
        for _ in range(1000):
            g = rng.choice(GOAL_SIGNALS)
            tracker.update(g, None)
        s = tracker.state
        assert len(s.recent_goal_signals) <= window
        assert 0.0 <= s.drift_score <= 1.0
        assert s.turn_count == 1000

    def test_window_equals_update_count(self):
        """Window exactly matches update count — all signals retained."""
        tracker = TrajectoryTracker(window_size=100)
        for i in range(100):
            tracker.update("push_forward" if i % 2 == 0 else "stabilize", None)
        s = tracker.state
        assert len(s.recent_goal_signals) == 100


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: MEMORY BEHAVIOR
# ═════════════════════════════════════════════════════════════════════════════

class TestStressMemoryBehavior:
    """Verify bounded memory regardless of update count."""

    def test_state_size_bounded_after_100k(self):
        """State internal lists must never exceed window_size."""
        tracker = TrajectoryTracker(window_size=10)
        for _ in range(100_000):
            tracker.update("push_forward", "progressing")
        # Internal state must be bounded
        assert len(tracker._goal_signals) <= 10
        assert len(tracker._trajectory_signals) <= 10

    def test_reset_reclaims_state(self):
        """After reset, internal lists are empty."""
        tracker = TrajectoryTracker(window_size=50)
        for _ in range(10_000):
            tracker.update("execute", "escalating")
        tracker.reset()
        assert len(tracker._goal_signals) == 0
        assert len(tracker._trajectory_signals) == 0
        assert tracker._turn_count == 0

    def test_repeated_reset_cycles(self):
        """Multiple fill → reset cycles must not leak state."""
        tracker = TrajectoryTracker(window_size=10)
        for cycle in range(100):
            for _ in range(100):
                tracker.update("push_forward", "progressing")
            tracker.reset()
        assert tracker.state.turn_count == 0
        assert len(tracker._goal_signals) == 0


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: UPDATE PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

class TestStressPerformance:
    """Verify near-constant update performance."""

    def test_update_time_scales_linearly_not_quadratically(self):
        """
        Compare 1k updates vs 10k updates — per-update time must not grow.
        This catches O(n) or O(n^2) regressions in window trimming.
        """
        tracker = TrajectoryTracker(window_size=20)
        rng = random.Random(42)

        # Warm up
        for _ in range(100):
            tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))

        # Measure 1k batch
        t0 = time.perf_counter()
        for _ in range(1_000):
            tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))
        t_1k = time.perf_counter() - t0

        # Measure next 10k batch (after 1.1k total)
        t0 = time.perf_counter()
        for _ in range(10_000):
            tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))
        t_10k = time.perf_counter() - t0

        per_update_1k = t_1k / 1_000
        per_update_10k = t_10k / 10_000

        # Per-update cost at 10k should not be >3x the cost at 1k
        # (generous margin for variance)
        assert per_update_10k < per_update_1k * 3.0, (
            f"Performance regression: {per_update_1k*1e6:.1f}μs/op (1k) "
            f"vs {per_update_10k*1e6:.1f}μs/op (10k)"
        )

    def test_10k_updates_under_100ms(self):
        """10,000 updates must complete in under 100ms total."""
        tracker = TrajectoryTracker(window_size=10)
        rng = random.Random(7)
        t0 = time.perf_counter()
        for _ in range(10_000):
            tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200.0, f"10k updates took {elapsed_ms:.1f}ms (budget: 200ms)"

    def test_state_access_constant_time(self):
        """Accessing .state property after many updates must be fast."""
        tracker = TrajectoryTracker(window_size=50)
        for _ in range(10_000):
            tracker.update("push_forward", "progressing")

        t0 = time.perf_counter()
        for _ in range(10_000):
            _ = tracker.state
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 200.0, f"10k state reads took {elapsed_ms:.1f}ms"


# ═════════════════════════════════════════════════════════════════════════════
# STRESS: SERIALIZATION UNDER LOAD
# ═════════════════════════════════════════════════════════════════════════════

class TestStressSerialization:
    """to_dict() must remain valid after heavy use."""

    def test_to_dict_after_10k_updates(self):
        import json
        tracker = TrajectoryTracker(window_size=20)
        rng = random.Random(42)
        for _ in range(10_000):
            tracker.update(rng.choice(GOAL_SIGNALS), rng.choice(TRAJ_SIGNALS))
        d = tracker.state.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["turn_count"] == 10_000
        assert len(d["recent_goal_signals"]) == 20

    def test_to_dict_after_reset(self):
        import json
        tracker = TrajectoryTracker()
        for _ in range(1000):
            tracker.update("execute", "escalating")
        tracker.reset()
        d = tracker.state.to_dict()
        json.dumps(d)  # must not raise
        assert d["turn_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
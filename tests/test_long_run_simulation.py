"""
A.C.C.E.S.S. -- Phase 5.1 Long-Run Simulation Suite
"""

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.adaptive_meta import (
    AdaptiveMetaController,
    AdaptiveMetaConfig,
    AdaptedContext,
)


@dataclass
class SimTurnRecord:
    turn: int
    raw_coherence: float
    adapted_threshold: float
    adapted_alpha: float
    fatigue_level: float
    stability_of_stability: float
    adjustments_active: bool
    stability_gate_open: bool
    dampening_factor: float
    circuit_breaker_active: bool
    coherence_mean: float


def run_simulation(coherence_sequence, config=None, risk_flags_fn=None):
    ctrl = AdaptiveMetaController(config=config)
    history = []
    records = []
    for i, coh in enumerate(coherence_sequence):
        history.append(coh)
        recent_history = history[-50:]
        flags = risk_flags_fn(i, coh) if risk_flags_fn else ()
        if not records:
            adj_triggered = False
        else:
            prev = records[-1]
            adj_triggered = prev.adjustments_active and not prev.circuit_breaker_active
        ctx = ctrl.adapt(
            raw_coherence=coh,
            raw_risk_flags=flags,
            coherence_history=recent_history,
            adjustment_triggered_this_turn=adj_triggered,
        )
        records.append(SimTurnRecord(
            turn=i, raw_coherence=coh,
            adapted_threshold=ctx.adapted_threshold,
            adapted_alpha=ctx.adapted_alpha,
            fatigue_level=ctx.fatigue_level,
            stability_of_stability=ctx.stability_of_stability,
            adjustments_active=ctx.adjustments_active,
            stability_gate_open=ctx.stability_gate_open,
            dampening_factor=ctx.dampening_factor,
            circuit_breaker_active=ctx.circuit_breaker_active,
            coherence_mean=ctrl.state.coherence_mean,
        ))
    return records


def assert_bounded_all(records):
    for r in records:
        assert 0.30 <= r.adapted_threshold <= 0.85
        assert 0.10 <= r.adapted_alpha <= 0.50
        assert 0.0 <= r.fatigue_level <= 1.0
        assert 0.0 <= r.stability_of_stability <= 1.0
        assert 0.0 < r.dampening_factor <= 1.0
        for attr in ("adapted_threshold", "adapted_alpha", "fatigue_level",
                      "stability_of_stability", "dampening_factor", "coherence_mean"):
            v = getattr(r, attr)
            assert not math.isnan(v)
            assert not math.isinf(v)


def mean_of(records, attr, start, end):
    vals = [getattr(records[i], attr) for i in range(start, min(end, len(records)))]
    return sum(vals) / len(vals) if vals else 0.0


def lcg_sequence(n, seed=42):
    m = 2**32
    a = 1664525
    c = 1013904223
    state = seed
    result = []
    for _ in range(n):
        state = (a * state + c) % m
        result.append(state / m)
    return result


class TestSimulateRegimeShift:

    def _build_sequence(self):
        seq = []
        for i in range(200):
            seq.append(0.85 + 0.02 * math.sin(i * 0.1))
        for i in range(200):
            seq.append(0.40 + 0.03 * math.sin(i * 0.15))
        for i in range(200):
            seq.append(0.80 + 0.02 * math.sin(i * 0.1))
        return seq

    def test_boundedness(self):
        assert_bounded_all(run_simulation(self._build_sequence()))

    def test_no_permanent_gate_lock(self):
        records = run_simulation(self._build_sequence())
        assert any(r.stability_gate_open for r in records[450:])

    def test_fatigue_returns_to_baseline(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "fatigue_level", 550, 600) < 0.5

    def test_threshold_adapts_to_degradation(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "adapted_threshold", 350, 400) < mean_of(records, "adapted_threshold", 150, 200)

    def test_convergence_no_divergence(self):
        records = run_simulation(self._build_sequence())
        for r in records:
            assert 0.0 <= r.coherence_mean <= 1.0
        assert mean_of(records, "coherence_mean", 550, 600) > mean_of(records, "coherence_mean", 350, 400)

    def test_adjustments_activate_during_transition(self):
        records = run_simulation(self._build_sequence())
        assert any(r.adjustments_active for r in records[200:240])

    def test_alpha_remains_bounded(self):
        records = run_simulation(self._build_sequence())
        assert all(0.10 <= r.adapted_alpha <= 0.50 for r in records)

    def test_no_slow_drift_after_recovery(self):
        records = run_simulation(self._build_sequence())
        stable_mean = mean_of(records, "adapted_threshold", 0, 150)
        recovery_mean = mean_of(records, "adapted_threshold", 500, 600)
        assert abs(recovery_mean - stable_mean) / stable_mean < 0.10
        ths = [records[i].adapted_threshold for i in range(500, 600)]
        n = len(ths)
        x_mean = (n - 1) / 2.0
        y_mean = sum(ths) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(ths))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-10 else 0.0
        assert slope < 0.001

    def test_fatigue_converges_monotonically_post_recovery(self):
        records = run_simulation(self._build_sequence())
        fat_last150 = [records[i].fatigue_level for i in range(450, 600)]
        max_fat = max(fat_last150)
        final_fat = fat_last150[-1]
        assert final_fat <= max_fat
        fat_last100 = [records[i].fatigue_level for i in range(500, 600)]
        n = len(fat_last100)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fat_last100) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(fat_last100))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-10 else 0.0
        assert slope <= 0.0 + 1e-9
        deltas = [abs(fat_last100[i] - fat_last100[i - 1]) for i in range(1, n)]
        assert max(deltas) < 0.20

    def test_stability_of_stability_recovers(self):
        records = run_simulation(self._build_sequence())
        degraded_s2 = mean_of(records, "stability_of_stability", 250, 350)
        recovery_s2 = mean_of(records, "stability_of_stability", 500, 600)
        assert recovery_s2 > degraded_s2
        assert recovery_s2 >= 0.6

    def test_no_hidden_explosive_trend(self):
        records = run_simulation(self._build_sequence())
        for i in range(20, len(records)):
            prev, curr = records[i - 1], records[i]
            assert abs(curr.adapted_threshold - prev.adapted_threshold) < 0.20
            assert abs(curr.adapted_alpha - prev.adapted_alpha) < 0.20
            assert abs(curr.fatigue_level - prev.fatigue_level) < 0.20


class TestSimulateOscillation:

    def _build_sequence(self):
        seq = []
        for i in range(200):
            seq.append(0.85 if i % 2 == 0 else 0.20)
        for _ in range(30):
            seq.append(0.20)
        for _ in range(200):
            seq.append(0.75)
        return seq

    def test_boundedness(self):
        assert_bounded_all(run_simulation(self._build_sequence()))

    def test_fatigue_accumulates(self):
        records = run_simulation(self._build_sequence())
        assert max(r.fatigue_level for r in records[:230]) > 0.1

    def test_circuit_breaker_activates(self):
        records = run_simulation(self._build_sequence())
        assert any(r.circuit_breaker_active for r in records[200:230])

    def test_system_recovers(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "fatigue_level", 380, 430) < max(r.fatigue_level for r in records[:230])

    def test_stability_decreases_under_oscillation(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "stability_of_stability", 50, 200) < 0.3

    def test_no_runaway_growth(self):
        records = run_simulation(self._build_sequence())
        for r in records:
            assert r.fatigue_level <= 1.0
            assert r.adapted_threshold <= 0.85
            assert r.adapted_alpha <= 0.50
            assert 0.0 <= r.coherence_mean <= 1.0

    def test_alpha_bounded(self):
        records = run_simulation(self._build_sequence())
        assert all(0.10 <= r.adapted_alpha <= 0.50 for r in records)

    def test_no_exponential_fatigue_growth(self):
        records = run_simulation(self._build_sequence())
        for r in records[:200]:
            assert r.fatigue_level <= 1.0
        consec_explosive = 0
        max_consec = 0
        for i in range(1, 200):
            prev_f = records[i - 1].fatigue_level
            curr_f = records[i].fatigue_level
            if prev_f > 0.05 and curr_f / prev_f > 1.5:
                consec_explosive += 1
                max_consec = max(max_consec, consec_explosive)
            else:
                consec_explosive = 0
        assert max_consec <= 3

    def test_dampening_factor_engages(self):
        records = run_simulation(self._build_sequence())
        damp_osc = [records[i].dampening_factor for i in range(200)]
        assert any(d < 1.0 for d in damp_osc)
        mean_damp_osc = sum(damp_osc) / len(damp_osc)
        damp_rec = [records[i].dampening_factor for i in range(230, len(records))]
        mean_damp_rec = sum(damp_rec) / len(damp_rec)
        assert mean_damp_osc < mean_damp_rec

    def test_no_memory_pathology_after_recovery(self):
        records = run_simulation(self._build_sequence())
        peak_fat = max(r.fatigue_level for r in records[:230])
        assert peak_fat > 0
        mean_fat_post = mean_of(records, "fatigue_level", 230, 400)
        assert mean_fat_post < 0.50 * peak_fat
        fat_last100 = [records[i].fatigue_level for i in range(len(records) - 100, len(records))]
        n = len(fat_last100)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fat_last100) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(fat_last100))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-10 else 0.0
        assert slope <= 1e-9

    def test_circuit_breaker_not_stuck(self):
        records = run_simulation(self._build_sequence())
        assert any(r.circuit_breaker_active for r in records)
        assert any(not r.circuit_breaker_active for r in records[230:])
        assert not all(r.circuit_breaker_active for r in records[-100:])

    def test_threshold_and_alpha_do_not_oscillate_amplify(self):
        records = run_simulation(self._build_sequence())
        warmup = 10
        for i in range(warmup, 200):
            assert abs(records[i].adapted_threshold - records[i - 1].adapted_threshold) < 0.25
            assert abs(records[i].adapted_alpha - records[i - 1].adapted_alpha) < 0.25
        def _var(vals):
            if len(vals) < 2:
                return 0.0
            m = sum(vals) / len(vals)
            return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
        th_first50 = [records[i].adapted_threshold for i in range(50)]
        th_last50 = [records[i].adapted_threshold for i in range(150, 200)]
        assert _var(th_last50) <= _var(th_first50) + 1e-12


class TestSimulateExtremeVolatility:

    def _build_sequence(self):
        return lcg_sequence(1000, seed=42)

    def test_boundedness(self):
        assert_bounded_all(run_simulation(self._build_sequence()))

    def test_stability_drops(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "stability_of_stability", 100, 300) < 0.8

    def test_stability_gate_closes(self):
        records = run_simulation(self._build_sequence())
        assert not all(r.stability_gate_open for r in records[50:])

    def test_no_variable_diverges(self):
        records = run_simulation(self._build_sequence())
        for r in records:
            assert 0.0 <= r.coherence_mean <= 1.0
            assert 0.0 <= r.fatigue_level <= 1.0
            assert 0.30 <= r.adapted_threshold <= 0.85
            assert 0.10 <= r.adapted_alpha <= 0.50

    def test_no_nan_inf(self):
        records = run_simulation(self._build_sequence())
        for r in records:
            for attr in ("adapted_threshold", "adapted_alpha", "fatigue_level",
                          "stability_of_stability", "dampening_factor", "coherence_mean"):
                v = getattr(r, attr)
                assert not math.isnan(v)
                assert not math.isinf(v)

    def test_coherence_mean_bounded(self):
        records = run_simulation(self._build_sequence())
        assert all(0.0 <= r.coherence_mean <= 1.0 for r in records)

    def test_deterministic_reproducibility(self):
        seq = self._build_sequence()
        a = run_simulation(seq)
        b = run_simulation(seq)
        for x, y in zip(a, b):
            assert x.adapted_threshold == y.adapted_threshold
            assert x.adapted_alpha == y.adapted_alpha
            assert x.fatigue_level == y.fatigue_level
            assert x.stability_of_stability == y.stability_of_stability
            assert x.adjustments_active == y.adjustments_active

    def test_no_long_term_threshold_drift(self):
        records = run_simulation(self._build_sequence())
        m1 = mean_of(records, "adapted_threshold", 0, 200)
        m2 = mean_of(records, "adapted_threshold", 400, 600)
        m3 = mean_of(records, "adapted_threshold", 800, 1000)
        overall = (m1 + m2 + m3) / 3.0
        assert abs(m1 - overall) / overall < 0.10
        assert abs(m2 - overall) / overall < 0.10
        assert abs(m3 - overall) / overall < 0.10
        assert not (m1 < m2 < m3)
        assert not (m1 > m2 > m3)

    def test_fatigue_stationarity_under_noise(self):
        records = run_simulation(self._build_sequence())
        def _var(vals):
            n = len(vals)
            m = sum(vals) / n
            return sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        v1 = _var([records[i].fatigue_level for i in range(500)])
        v2 = _var([records[i].fatigue_level for i in range(500, 1000)])
        ratio = max(v1, v2) / min(v1, v2) if min(v1, v2) > 1e-12 else 1.0
        assert ratio < 2.0
        fat_tail = [records[i].fatigue_level for i in range(700, 1000)]
        n = len(fat_tail)
        x_mean = (n - 1) / 2.0
        y_mean = sum(fat_tail) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(fat_tail))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-10 else 0.0
        assert slope <= 0.001

    def test_stability_of_stability_entropy(self):
        records = run_simulation(self._build_sequence())
        s2 = [r.stability_of_stability for r in records]
        n = len(s2)
        s2_mean = sum(s2) / n
        s2_std = (sum((v - s2_mean) ** 2 for v in s2) / n) ** 0.5
        assert s2_mean > 0.2
        assert s2_std < 0.4
        zero_count = sum(1 for v in s2 if v < 0.01)
        assert zero_count < n * 0.05

    def test_no_hidden_bias_accumulation(self):
        records = run_simulation(self._build_sequence())
        cm = [r.coherence_mean for r in records]
        n = len(cm)
        x_mean = (n - 1) / 2.0
        y_mean = sum(cm) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(cm))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / den if abs(den) > 1e-10 else 0.0
        assert abs(slope) < 0.001
        assert 0.3 <= y_mean <= 0.7

    def test_alpha_and_threshold_covariance_stable(self):
        records = run_simulation(self._build_sequence())
        th = [r.adapted_threshold for r in records]
        al = [r.adapted_alpha for r in records]
        n = len(th)
        th_m = sum(th) / n
        al_m = sum(al) / n
        cov = sum((t - th_m) * (a - al_m) for t, a in zip(th, al)) / (n - 1)
        assert abs(cov) < 0.05


class TestSimulateRecoveryCycle:

    def _build_sequence(self):
        seq = []
        for i in range(300):
            seq.append(0.35 + 0.10 * math.sin(i * 0.3))
        for i in range(300):
            t = i / 299.0
            seq.append(0.35 + (0.85 - 0.35) * t)
        return seq

    def test_boundedness(self):
        assert_bounded_all(run_simulation(self._build_sequence()))

    def test_no_fatigue_lock(self):
        records = run_simulation(self._build_sequence())
        assert records[-1].fatigue_level < 0.8

    def test_no_stuck_adjustments(self):
        records = run_simulation(self._build_sequence())
        assert not all(r.adjustments_active for r in records[-100:])

    def test_threshold_adapts_smoothly_after_warmup(self):
        records = run_simulation(self._build_sequence())
        ths = [r.adapted_threshold for r in records]
        max_jump = max(abs(ths[i] - ths[i-1]) for i in range(10, len(ths)))
        assert max_jump < 0.15

    def test_system_stabilizes(self):
        records = run_simulation(self._build_sequence())
        assert mean_of(records, "coherence_mean", 550, 600) > 0.55

    def test_stability_increases_during_recovery(self):
        records = run_simulation(self._build_sequence())
        degraded_s2 = mean_of(records, "stability_of_stability", 200, 300)
        recovery_s2 = mean_of(records, "stability_of_stability", 500, 600)
        assert recovery_s2 > degraded_s2

    def test_alpha_tracks_stability(self):
        records = run_simulation(self._build_sequence())
        degraded_a = mean_of(records, "adapted_alpha", 200, 300)
        recovery_a = mean_of(records, "adapted_alpha", 500, 600)
        assert recovery_a >= degraded_a


class TestAdversarialStress:

    def test_mixed_noise_and_oscillation(self):
        seq = []
        for i in range(100):
            seq.append(0.9 if i % 2 == 0 else 0.2)
        seq.extend(lcg_sequence(300, seed=77))
        for _ in range(200):
            seq.append(0.8)
        for i in range(100):
            seq.append(0.2 if i % 5 == 0 else 0.8)
        records = run_simulation(seq)
        assert_bounded_all(records)
        for r in records:
            for attr in ("adapted_threshold", "adapted_alpha", "fatigue_level",
                          "stability_of_stability", "dampening_factor", "coherence_mean"):
                v = getattr(r, attr)
                assert not math.isnan(v)
                assert not math.isinf(v)
        m1 = mean_of(records, "adapted_threshold", 0, 100)
        m2 = mean_of(records, "adapted_threshold", 300, 500)
        m3 = mean_of(records, "adapted_threshold", 600, 700)
        assert not (m1 < m2 < m3)
        assert not (m1 > m2 > m3)
        assert records[-1].fatigue_level < 0.6

    def test_forced_adjustment_trigger_stability(self):
        seq = lcg_sequence(500, seed=42)
        ctrl = AdaptiveMetaController()
        history = []
        records = []
        for i, coh in enumerate(seq):
            history.append(coh)
            recent = history[-50:]
            forced = (i % 3 == 0)
            ctx = ctrl.adapt(
                raw_coherence=coh,
                raw_risk_flags=(),
                coherence_history=recent,
                adjustment_triggered_this_turn=forced,
            )
            records.append(SimTurnRecord(
                turn=i, raw_coherence=coh,
                adapted_threshold=ctx.adapted_threshold,
                adapted_alpha=ctx.adapted_alpha,
                fatigue_level=ctx.fatigue_level,
                stability_of_stability=ctx.stability_of_stability,
                adjustments_active=ctx.adjustments_active,
                stability_gate_open=ctx.stability_gate_open,
                dampening_factor=ctx.dampening_factor,
                circuit_breaker_active=ctx.circuit_breaker_active,
                coherence_mean=ctrl.state.coherence_mean,
            ))
        assert all(r.fatigue_level <= 1.0 for r in records)
        assert all(0.10 <= r.adapted_alpha <= 0.50 for r in records)
        s2_last100 = [records[i].stability_of_stability for i in range(400, 500)]
        assert not all(v < 0.1 for v in s2_last100)
        assert not all(r.circuit_breaker_active for r in records[-100:])

    def test_risk_flag_injection(self):
        seq = []
        for i in range(300):
            seq.append(0.75 + 0.03 * math.sin(i * 0.1))
        for i in range(300):
            seq.append(0.40 + 0.05 * math.sin(i * 0.2))

        def risk_fn(turn, coh):
            if turn < 400 and turn % 7 == 0:
                return ("dependency_rising", "goals_failing")
            return ()

        records = run_simulation(seq, risk_flags_fn=risk_fn)
        assert_bounded_all(records)
        mean_fat_post = mean_of(records, "fatigue_level", 450, 600)
        peak_fat = max(r.fatigue_level for r in records[:400])
        if peak_fat > 0:
            assert mean_fat_post <= peak_fat
        assert all(0.30 <= r.adapted_threshold <= 0.85 for r in records)

    def test_short_history_vs_long_history(self):
        seq = []
        for i in range(150):
            seq.append(0.85 + 0.01 * math.sin(i * 0.1))
        for i in range(100):
            seq.append(0.40 + 0.03 * math.sin(i * 0.2))
        for i in range(150):
            seq.append(0.80 + 0.02 * math.sin(i * 0.1))
        records_long = run_simulation(seq)
        ctrl = AdaptiveMetaController()
        hist = []
        records_short = []
        for i, coh in enumerate(seq):
            hist.append(coh)
            recent = hist[-10:]
            adj = False
            if records_short:
                prev = records_short[-1]
                adj = prev.adjustments_active and not prev.circuit_breaker_active
            ctx = ctrl.adapt(
                raw_coherence=coh,
                raw_risk_flags=(),
                coherence_history=recent,
                adjustment_triggered_this_turn=adj,
            )
            records_short.append(SimTurnRecord(
                turn=i, raw_coherence=coh,
                adapted_threshold=ctx.adapted_threshold,
                adapted_alpha=ctx.adapted_alpha,
                fatigue_level=ctx.fatigue_level,
                stability_of_stability=ctx.stability_of_stability,
                adjustments_active=ctx.adjustments_active,
                stability_gate_open=ctx.stability_gate_open,
                dampening_factor=ctx.dampening_factor,
                circuit_breaker_active=ctx.circuit_breaker_active,
                coherence_mean=ctrl.state.coherence_mean,
            ))
        assert_bounded_all(records_short)
        n = len(seq)
        th_long = mean_of(records_long, "adapted_threshold", 0, n)
        th_short = mean_of(records_short, "adapted_threshold", 0, n)
        denom = max(th_long, th_short)
        assert abs(th_long - th_short) / denom < 0.15 if denom > 1e-10 else True
        for r in records_short:
            for attr in ("adapted_threshold", "adapted_alpha", "fatigue_level",
                          "stability_of_stability", "dampening_factor", "coherence_mean"):
                v = getattr(r, attr)
                assert not math.isnan(v)
                assert not math.isinf(v)
"""
A.C.C.E.S.S. — Conversation Layer Tests

Test strategy:
- No real EmotionEngine instantiation (no embedder, no DB)
- All inputs are duck-typed minimal objects
- ConversationConfig overridden per-test to make assertions deterministic
- Each test validates exactly one logical rule

Coverage:
    TestBurnoutModulation          — risk_burnout strategy fires + overrides
    TestManiaModulation            — risk_mania strategy fires + overrides
    TestDecliningTrendModulation   — trend_declining strategy
    TestPositiveMomentumModulation — trend_improving strategy
    TestDominantPatternInfluence   — pattern_frustration + pattern_confidence
    TestPADMappingLogic            — PAD-derived grounding / challenging / supportive
    TestConfigOverride             — all thresholds configurable, no hard-coded values
    TestStrategyChainOrdering      — priority order + multiple strategies composing
    TestExtensibility              — custom strategy injection (open/closed check)
    TestResponseModulationContract — ResponseModulation immutability + to_dict shape
"""

import sys
import os
import pytest
from dataclasses import dataclass
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from conversation.conversation_modulation import (
    ConversationModulator,
    ToneStrategy,
    BurnoutRiskStrategy,
    ManiaRiskStrategy,
    DecliningTrendStrategy,
    ImprovingTrendStrategy,
    SustainedFrustrationStrategy,
    SustainedConfidenceStrategy,
    PADGroundingStrategy,
    PADChallengingStrategy,
    PADSupportiveStrategy,
)
from conversation.conversation_config import ConversationConfig
from conversation.models import ResponseModulation, ModulationBuilder


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — minimal duck-typed stubs (zero dependencies)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakePAD:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5


@dataclass
class FakeState:
    """Duck-typed minimal EmotionalState — no imports from emotion package."""
    primary_emotion: str = "neutral"
    intensity: float = 0.5
    confidence: float = 0.8
    pad: FakePAD = None
    is_positive: bool = False
    is_negative: bool = False

    def __post_init__(self):
        if self.pad is None:
            self.pad = FakePAD()


def _make_trend(
    label: str = "stable",
    valence_slope: float = 0.0,
    arousal_slope: float = 0.0,
    burnout_risk: bool = False,
    mania_risk: bool = False,
    data_points: int = 10,
) -> dict:
    return {
        "trend_label": label,
        "valence_slope": valence_slope,
        "arousal_slope": arousal_slope,
        "dominance_slope": 0.0,
        "burnout_risk": burnout_risk,
        "mania_risk": mania_risk,
        "data_points": data_points,
    }


def _make_frustration_state() -> FakeState:
    return FakeState(
        primary_emotion="frustration",
        intensity=0.8,
        pad=FakePAD(valence=-0.65, arousal=0.70, dominance=0.30),
        is_negative=True,
    )


def _make_confidence_state() -> FakeState:
    return FakeState(
        primary_emotion="confidence",
        intensity=0.75,
        pad=FakePAD(valence=0.75, arousal=0.60, dominance=0.85),
        is_positive=True,
    )


def _make_fatigue_state() -> FakeState:
    return FakeState(
        primary_emotion="fatigue",
        intensity=0.7,
        pad=FakePAD(valence=-0.40, arousal=0.10, dominance=0.20),
        is_negative=True,
    )


def _default_modulator() -> ConversationModulator:
    return ConversationModulator(config=ConversationConfig())


# ─────────────────────────────────────────────────────────────────────────────
# TestBurnoutModulation
# ─────────────────────────────────────────────────────────────────────────────

class TestBurnoutModulation:
    """Burnout risk flag must produce safe, validating, low-pressure modulation."""

    def test_burnout_sets_calm_tone(self):
        mod = _default_modulator()
        trend = _make_trend(burnout_risk=True, valence_slope=-0.15)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.tone == "calm"

    def test_burnout_sets_slow_pacing(self):
        mod = _default_modulator()
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.pacing == "slow"

    def test_burnout_enables_emotional_validation(self):
        mod = _default_modulator()
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.emotional_validation is True

    def test_burnout_reduces_cognitive_load(self):
        config = ConversationConfig(burnout_cognitive_load=0.25)
        mod = ConversationModulator(config=config)
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.cognitive_load_limit == pytest.approx(0.25)

    def test_burnout_sets_small_positive_motivational_bias(self):
        config = ConversationConfig(burnout_motivational_bias=0.1)
        mod = ConversationModulator(config=config)
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        # Small positive: between 0 and 0.3
        assert 0.0 <= result.motivational_bias <= 0.3

    def test_burnout_strategy_recorded_in_active_strategies(self):
        mod = _default_modulator()
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert "risk_burnout" in result.active_strategies

    def test_burnout_overrides_improving_trend(self):
        """Burnout (priority 30) must win over improving trend (priority 20)."""
        mod = _default_modulator()
        # Both burnout flag AND positive trend slope (contradictory — burnout wins)
        trend = _make_trend(burnout_risk=True, label="improving", valence_slope=0.1)
        result = mod.build_modulation(_make_confidence_state(), trend)
        # Burnout should override the energizing tone
        assert result.tone == "calm"
        assert result.pacing == "slow"


# ─────────────────────────────────────────────────────────────────────────────
# TestManiaModulation
# ─────────────────────────────────────────────────────────────────────────────

class TestManiaModulation:
    """Mania risk flag must ground and structure the response."""

    def test_mania_sets_grounding_tone(self):
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.tone == "grounding"

    def test_mania_sets_normal_pacing(self):
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.pacing == "normal"

    def test_mania_sets_concise_verbosity(self):
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.verbosity == "concise"

    def test_mania_sets_structured_bias(self):
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.structure_bias == "structured"

    def test_mania_applies_slightly_negative_motivational_bias(self):
        config = ConversationConfig(mania_motivational_bias=-0.25)
        mod = ConversationModulator(config=config)
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.motivational_bias == pytest.approx(-0.25)

    def test_mania_strategy_recorded(self):
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert "risk_mania" in result.active_strategies

    def test_mania_overrides_confidence_pattern(self):
        """Mania (priority 30) must win over confidence pattern (priority 10)."""
        mod = _default_modulator()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(_make_confidence_state(), trend, dominant_pattern="confidence")
        assert result.tone == "grounding"


# ─────────────────────────────────────────────────────────────────────────────
# TestDecliningTrendModulation
# ─────────────────────────────────────────────────────────────────────────────

class TestDecliningTrendModulation:
    """Declining valence trend must produce reassuring, validating response."""

    def test_declining_label_sets_reassuring_tone(self):
        mod = _default_modulator()
        trend = _make_trend(label="declining", valence_slope=-0.08)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.tone == "reassuring"

    def test_escalating_label_also_triggers_declining_strategy(self):
        """Escalating = declining valence + rising arousal — still reassuring."""
        mod = _default_modulator()
        trend = _make_trend(label="escalating", valence_slope=-0.06)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.tone == "reassuring"

    def test_steep_negative_slope_triggers_declining_without_label(self):
        """Slope < -0.03 triggers strategy regardless of label."""
        mod = _default_modulator()
        trend = _make_trend(label="stable", valence_slope=-0.05)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.tone == "reassuring"

    def test_declining_enables_emotional_validation(self):
        mod = _default_modulator()
        trend = _make_trend(label="declining", valence_slope=-0.06)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.emotional_validation is True

    def test_declining_applies_gentle_positive_bias(self):
        config = ConversationConfig(declining_motivational_bias=0.15)
        mod = ConversationModulator(config=config)
        trend = _make_trend(label="declining", valence_slope=-0.06)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.motivational_bias >= 0.0

    def test_declining_strategy_recorded(self):
        mod = _default_modulator()
        trend = _make_trend(label="declining", valence_slope=-0.08)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert "trend_declining" in result.active_strategies


# ─────────────────────────────────────────────────────────────────────────────
# TestPositiveMomentumModulation
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveMomentumModulation:
    """Improving trend must amplify momentum with energizing, fast tone."""

    def test_improving_sets_energizing_tone(self):
        mod = _default_modulator()
        trend = _make_trend(label="improving", valence_slope=0.10)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.tone == "energizing"

    def test_improving_sets_fast_pacing(self):
        mod = _default_modulator()
        trend = _make_trend(label="improving", valence_slope=0.10)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.pacing == "fast"

    def test_improving_sets_positive_motivational_bias(self):
        config = ConversationConfig(improving_motivational_bias=0.4)
        mod = ConversationModulator(config=config)
        trend = _make_trend(label="improving", valence_slope=0.08)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.motivational_bias >= 0.3

    def test_positive_slope_triggers_improving_without_label(self):
        """Slope > 0.03 triggers strategy even if label is 'stable'."""
        mod = _default_modulator()
        trend = _make_trend(label="stable", valence_slope=0.05)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.tone == "energizing"

    def test_improving_strategy_recorded(self):
        mod = _default_modulator()
        trend = _make_trend(label="improving", valence_slope=0.10)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert "trend_improving" in result.active_strategies


# ─────────────────────────────────────────────────────────────────────────────
# TestDominantPatternInfluence
# ─────────────────────────────────────────────────────────────────────────────

class TestDominantPatternInfluence:
    """Dominant emotional pattern modulates load, structure, and challenge level."""

    def test_frustration_pattern_reduces_cognitive_load(self):
        config = ConversationConfig(frustration_cognitive_load=0.5)
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=-0.1, arousal=0.4, dominance=0.5))
        result = mod.build_modulation(state, trend, dominant_pattern="frustration")
        assert result.cognitive_load_limit <= 0.5

    def test_frustration_pattern_increases_structure(self):
        mod = _default_modulator()
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=-0.1, arousal=0.4, dominance=0.5))
        result = mod.build_modulation(state, trend, dominant_pattern="frustration")
        assert result.structure_bias == "structured"

    def test_frustration_pattern_strategy_recorded(self):
        mod = _default_modulator()
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=-0.1, arousal=0.4, dominance=0.5))
        result = mod.build_modulation(state, trend, dominant_pattern="frustration")
        assert "pattern_frustration" in result.active_strategies

    def test_confidence_pattern_raises_motivational_bias(self):
        config = ConversationConfig(confidence_motivational_bias=0.55)
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        result = mod.build_modulation(_make_confidence_state(), trend, dominant_pattern="confidence")
        assert result.motivational_bias >= 0.5

    def test_confidence_pattern_disables_validation(self):
        config = ConversationConfig(confidence_emotional_validation=False)
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        result = mod.build_modulation(_make_confidence_state(), trend, dominant_pattern="confidence")
        assert result.emotional_validation is False

    def test_confidence_pattern_strategy_recorded(self):
        mod = _default_modulator()
        trend = _make_trend()
        result = mod.build_modulation(_make_confidence_state(), trend, dominant_pattern="confidence")
        assert "pattern_confidence" in result.active_strategies

    def test_no_pattern_does_not_affect_cognitive_load(self):
        """Without a dominant pattern, cognitive load stays at default."""
        config = ConversationConfig(default_cognitive_load_limit=1.0)
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=0.3, arousal=0.3, dominance=0.6))
        result = mod.build_modulation(state, trend, dominant_pattern=None)
        assert result.cognitive_load_limit == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TestPADMappingLogic
# ─────────────────────────────────────────────────────────────────────────────

class TestPADMappingLogic:
    """PAD coordinates should drive tone independently of trend/pattern flags."""

    def test_low_valence_high_arousal_grounding(self):
        """Agitated negative state → grounding tone (stable trend, no pattern)."""
        mod = _default_modulator()
        trend = _make_trend()  # no risk flags
        state = FakeState(pad=FakePAD(valence=-0.5, arousal=0.8, dominance=0.4))
        result = mod.build_modulation(state, trend, dominant_pattern=None)
        # PAD grounding strategy fires; check it was recorded
        assert "pad_grounding" in result.active_strategies

    def test_high_valence_high_dominance_challenging(self):
        """Empowered state → challenging tone should appear."""
        mod = _default_modulator()
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=0.75, arousal=0.6, dominance=0.85))
        result = mod.build_modulation(state, trend, dominant_pattern=None)
        assert "pad_challenging" in result.active_strategies

    def test_high_valence_high_dominance_raises_motivational_bias(self):
        config = ConversationConfig(pad_challenging_motivational_bias=0.5)
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=0.75, arousal=0.6, dominance=0.85))
        result = mod.build_modulation(state, trend)
        assert result.motivational_bias >= 0.4

    def test_low_dominance_supportive(self):
        """Low dominance → supportive tone and validation."""
        mod = _default_modulator()
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=0.0, arousal=0.4, dominance=0.2))
        result = mod.build_modulation(state, trend, dominant_pattern=None)
        assert "pad_supportive" in result.active_strategies
        assert result.emotional_validation is True

    def test_neutral_pad_no_pad_strategy_fires(self):
        """Neutral PAD — no grounding, no challenging, no supportive."""
        mod = _default_modulator()
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=0.0, arousal=0.5, dominance=0.5))
        result = mod.build_modulation(state, trend)
        pad_strategies = {"pad_grounding", "pad_challenging", "pad_supportive"}
        fired = set(result.active_strategies)
        assert fired.isdisjoint(pad_strategies)

    def test_grounding_pacing_is_slow(self):
        config = ConversationConfig(pad_grounding_pacing="slow")
        mod = ConversationModulator(config=config)
        trend = _make_trend()
        state = FakeState(pad=FakePAD(valence=-0.5, arousal=0.8, dominance=0.4))
        result = mod.build_modulation(state, trend)
        assert result.pacing == "slow"


# ─────────────────────────────────────────────────────────────────────────────
# TestConfigOverride
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigOverride:
    """All thresholds must be configurable. No magic numbers in module code."""

    def test_burnout_cognitive_load_configurable(self):
        for load in [0.1, 0.3, 0.6]:
            config = ConversationConfig(burnout_cognitive_load=load)
            mod = ConversationModulator(config=config)
            result = mod.build_modulation(
                _make_frustration_state(), _make_trend(burnout_risk=True)
            )
            assert result.cognitive_load_limit == pytest.approx(load)

    def test_mania_motivational_bias_configurable(self):
        for bias in [-0.5, -0.2, 0.0]:
            config = ConversationConfig(mania_motivational_bias=bias)
            mod = ConversationModulator(config=config)
            result = mod.build_modulation(
                _make_confidence_state(), _make_trend(mania_risk=True)
            )
            assert result.motivational_bias == pytest.approx(bias)

    def test_improving_pacing_configurable(self):
        config = ConversationConfig(improving_pacing="normal")
        mod = ConversationModulator(config=config)
        trend = _make_trend(label="improving", valence_slope=0.1)
        result = mod.build_modulation(_make_confidence_state(), trend)
        assert result.pacing == "normal"

    def test_declining_motivational_bias_configurable(self):
        config = ConversationConfig(declining_motivational_bias=0.3)
        mod = ConversationModulator(config=config)
        trend = _make_trend(label="declining", valence_slope=-0.06)
        result = mod.build_modulation(_make_frustration_state(), trend)
        assert result.motivational_bias >= 0.3

    def test_frustration_cognitive_load_configurable(self):
        config = ConversationConfig(frustration_cognitive_load=0.3)
        mod = ConversationModulator(config=config)
        state = FakeState(pad=FakePAD(valence=-0.1, arousal=0.4, dominance=0.5))
        result = mod.build_modulation(state, _make_trend(), dominant_pattern="frustration")
        assert result.cognitive_load_limit <= 0.3

    def test_default_tone_configurable(self):
        config = ConversationConfig(default_tone="supportive")
        mod = ConversationModulator(config=config)
        state = FakeState(pad=FakePAD(valence=0.0, arousal=0.5, dominance=0.5))
        result = mod.build_modulation(state, _make_trend())
        # No other strategy fires → default applies
        assert result.tone == "supportive"


# ─────────────────────────────────────────────────────────────────────────────
# TestStrategyChainOrdering
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyChainOrdering:
    """Higher priority strategies must override lower priority ones."""

    def test_risk_overrides_pad(self):
        """burnout (p=30) overrides pad_grounding (p=0)."""
        mod = _default_modulator()
        # Agitated negative state → pad_grounding would fire
        state = FakeState(pad=FakePAD(valence=-0.5, arousal=0.8, dominance=0.3))
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(state, trend)
        # burnout is last (priority=30), so calm wins
        assert result.tone == "calm"

    def test_risk_overrides_pattern(self):
        """mania (p=30) overrides pattern_confidence (p=10)."""
        mod = _default_modulator()
        state = _make_confidence_state()
        trend = _make_trend(mania_risk=True)
        result = mod.build_modulation(state, trend, dominant_pattern="confidence")
        assert result.tone == "grounding"
        assert "risk_mania" in result.active_strategies

    def test_trend_overrides_pad(self):
        """trend_declining (p=20) overrides pad_grounding (p=0)."""
        mod = _default_modulator()
        state = FakeState(pad=FakePAD(valence=-0.5, arousal=0.8, dominance=0.4))
        trend = _make_trend(label="declining", valence_slope=-0.08)
        result = mod.build_modulation(state, trend)
        # declining tone = reassuring wins over grounding from PAD
        assert result.tone == "reassuring"

    def test_multiple_strategies_compose_fields(self):
        """Pattern strategy sets structure; trend strategy sets tone. Both should apply."""
        mod = _default_modulator()
        state = FakeState(pad=FakePAD(valence=-0.2, arousal=0.5, dominance=0.5))
        trend = _make_trend(label="declining", valence_slope=-0.05)
        result = mod.build_modulation(state, trend, dominant_pattern="frustration")
        # Declining trend → tone=reassuring
        assert result.tone == "reassuring"
        # Frustration pattern → structure=structured
        assert result.structure_bias == "structured"
        # Both strategies recorded
        assert "trend_declining" in result.active_strategies
        assert "pattern_frustration" in result.active_strategies

    def test_motivational_bias_clamped_to_range(self):
        """motivational_bias must never exceed [-1.0, +1.0] even with stacked boosts."""
        config = ConversationConfig(
            confidence_motivational_bias=0.9,
            improving_motivational_bias=0.8,
            pad_challenging_motivational_bias=0.7,
        )
        mod = ConversationModulator(config=config)
        state = FakeState(pad=FakePAD(valence=0.75, arousal=0.6, dominance=0.85))
        trend = _make_trend(label="improving", valence_slope=0.10)
        result = mod.build_modulation(state, trend, dominant_pattern="confidence")
        assert -1.0 <= result.motivational_bias <= 1.0

    def test_cognitive_load_clamped_to_range(self):
        """cognitive_load_limit must always be in [0.0, 1.0]."""
        config = ConversationConfig(burnout_cognitive_load=-0.5)  # invalid input
        mod = ConversationModulator(config=config)
        result = mod.build_modulation(
            _make_frustration_state(), _make_trend(burnout_risk=True)
        )
        assert 0.0 <= result.cognitive_load_limit <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TestExtensibility
# ─────────────────────────────────────────────────────────────────────────────

class TestExtensibility:
    """New strategies should be injectable without modifying core code."""

    def test_custom_strategy_fires_when_matching(self):
        """Inject a custom strategy and verify it fires."""

        class FatigueNightMode(ToneStrategy):
            priority = 5
            name = "fatigue_night_mode"

            def matches(self, state, trend, dominant_pattern):
                return getattr(state, "primary_emotion", "") == "fatigue"

            def apply(self, builder, state, trend, dominant_pattern, config):
                builder.verbosity = "concise"
                builder.pacing = "slow"

        mod = ConversationModulator(extra_strategies=[FatigueNightMode()])
        state = _make_fatigue_state()
        result = mod.build_modulation(state, _make_trend())
        assert "fatigue_night_mode" in result.active_strategies
        assert result.verbosity == "concise"

    def test_custom_strategy_does_not_fire_when_not_matching(self):
        class NeverFires(ToneStrategy):
            priority = 99
            name = "never_fires"

            def matches(self, state, trend, dominant_pattern):
                return False

            def apply(self, builder, state, trend, dominant_pattern, config):
                builder.tone = "THIS_SHOULD_NOT_APPEAR"

        mod = ConversationModulator(extra_strategies=[NeverFires()])
        result = mod.build_modulation(_make_confidence_state(), _make_trend())
        assert "never_fires" not in result.active_strategies
        assert result.tone != "THIS_SHOULD_NOT_APPEAR"

    def test_custom_strategy_priority_respected(self):
        """Custom strategy at priority 5 must not override burnout at priority 30."""

        class AlwaysEnergizing(ToneStrategy):
            priority = 5
            name = "always_energizing"

            def matches(self, state, trend, dominant_pattern):
                return True

            def apply(self, builder, state, trend, dominant_pattern, config):
                builder.tone = "energizing"

        mod = ConversationModulator(extra_strategies=[AlwaysEnergizing()])
        trend = _make_trend(burnout_risk=True)
        result = mod.build_modulation(_make_frustration_state(), trend)
        # burnout (p=30) fires after AlwaysEnergizing (p=5) → calm wins
        assert result.tone == "calm"

    def test_bad_strategy_does_not_crash_modulator(self):
        """A strategy that raises an exception must be caught and skipped."""

        class CrashingStrategy(ToneStrategy):
            priority = 1
            name = "crashing"

            def matches(self, state, trend, dominant_pattern):
                return True

            def apply(self, builder, state, trend, dominant_pattern, config):
                raise RuntimeError("Intentional crash for testing")

        mod = ConversationModulator(extra_strategies=[CrashingStrategy()])
        # Should not raise
        result = mod.build_modulation(_make_confidence_state(), _make_trend())
        # Crashing strategy should not appear in active strategies
        assert "crashing" not in result.active_strategies

    def test_strategy_names_accessible(self):
        mod = _default_modulator()
        names = mod.strategy_names
        assert "risk_burnout" in names
        assert "risk_mania" in names
        assert "trend_declining" in names
        assert "trend_improving" in names


# ─────────────────────────────────────────────────────────────────────────────
# TestResponseModulationContract
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseModulationContract:
    """ResponseModulation must be immutable and serialize correctly."""

    def test_response_modulation_is_frozen(self):
        mod = _default_modulator()
        result = mod.build_modulation(_make_frustration_state(), _make_trend())
        with pytest.raises((AttributeError, TypeError)):
            result.tone = "something_else"  # type: ignore

    def test_to_dict_has_all_required_keys(self):
        mod = _default_modulator()
        result = mod.build_modulation(_make_confidence_state(), _make_trend())
        d = result.to_dict()
        required_keys = {
            "tone", "pacing", "verbosity", "structure_bias",
            "emotional_validation", "motivational_bias",
            "cognitive_load_limit", "active_strategies",
        }
        assert required_keys == set(d.keys())

    def test_to_dict_motivational_bias_rounded(self):
        mod = _default_modulator()
        result = mod.build_modulation(_make_confidence_state(), _make_trend())
        d = result.to_dict()
        # Should be rounded to 3 decimal places
        bias = d["motivational_bias"]
        assert isinstance(bias, float)
        assert round(bias, 3) == bias

    def test_active_strategies_is_list_in_dict(self):
        mod = _default_modulator()
        result = mod.build_modulation(_make_frustration_state(), _make_trend(burnout_risk=True))
        d = result.to_dict()
        assert isinstance(d["active_strategies"], list)

    def test_active_strategies_is_tuple_on_object(self):
        mod = _default_modulator()
        result = mod.build_modulation(_make_frustration_state(), _make_trend(burnout_risk=True))
        assert isinstance(result.active_strategies, tuple)

    def test_no_strategy_fires_on_neutral_input(self):
        """Completely neutral state → only defaults, no strategies fire."""
        mod = _default_modulator()
        state = FakeState(pad=FakePAD(valence=0.0, arousal=0.5, dominance=0.5))
        trend = _make_trend(label="stable", valence_slope=0.0)
        result = mod.build_modulation(state, trend, dominant_pattern=None)
        assert len(result.active_strategies) == 0

    def test_explain_returns_string(self):
        mod = _default_modulator()
        state = _make_frustration_state()
        trend = _make_trend(burnout_risk=True)
        explanation = mod.explain(state, trend, dominant_pattern="frustration")
        assert isinstance(explanation, str)
        assert "risk_burnout" in explanation
        assert "ConversationModulator" in explanation
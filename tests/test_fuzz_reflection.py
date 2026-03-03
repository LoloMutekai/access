"""
A.C.C.E.S.S. — Fuzz Tests for ReflectionEngine (Phase 3 Solidification)

Generates randomized and extreme inputs to verify crash resistance:
    - Extreme float ranges for intensity, valence, arousal
    - Extreme motivational_bias values [-100, 100]
    - assistant_output from 0 to 10,000 chars
    - Malformed tool_results (wrong types, missing attrs)
    - None states and modulations in all combinations
    - Unicode, empty strings, newlines in user_input

Guarantees:
    - No crashes on any input
    - importance_score always clamped [0.0, 1.0]
    - goal_signal always valid
    - trajectory_signal always valid
    - summary always ≤ 200 chars
"""

import sys
import os
import random
import string
from dataclasses import dataclass
from typing import Optional

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.reflection_engine import ReflectionEngine, ReflectionConfig, ReflectionResult

VALID_GOAL_SIGNALS = {"push_forward", "execute", "stabilize", "recover", "explore", None}
VALID_TRAJECTORY_SIGNALS = {"progressing", "declining", "stable", "escalating", None}


# ─── Fuzz data generators ─────────────────────────────────────────────────────

@dataclass
class FuzzState:
    primary_emotion: str = "neutral"
    intensity: float = 0.5
    is_positive: bool = False
    is_negative: bool = False
    is_high_arousal: bool = False
    label: str = "neutral"
    pad: object = None


@dataclass
class FuzzPAD:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5


@dataclass
class FuzzMod:
    tone: str = "neutral"
    pacing: str = "normal"
    verbosity: str = "normal"
    structure_bias: str = "conversational"
    emotional_validation: bool = False
    motivational_bias: float = 0.0
    cognitive_load_limit: float = 1.0
    active_strategies: tuple = ()


@dataclass
class FuzzToolResult:
    tool_name: str = "unknown"
    success: bool = True
    output: dict = None
    latency_ms: float = 0.0

    def __post_init__(self):
        if self.output is None:
            self.output = {}


def _random_string(length: int) -> str:
    return "".join(random.choices(string.printable, k=length))


ENGINE = ReflectionEngine()


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: EXTREME FLOAT RANGES
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzExtremeFloats:
    """Intensity, valence, arousal at extreme values."""

    @given(intensity=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_extreme_intensity_no_crash(self, intensity):
        state = FuzzState(intensity=intensity, pad=FuzzPAD())
        mod = FuzzMod()
        result = ENGINE.reflect("test", "output", state, mod, ())
        assert 0.0 <= result.importance_score <= 1.0
        assert result.goal_signal in VALID_GOAL_SIGNALS

    @given(
        valence=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        arousal=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        dominance=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_extreme_pad_values_no_crash(self, valence, arousal, dominance):
        state = FuzzState(pad=FuzzPAD(valence=valence, arousal=arousal, dominance=dominance))
        mod = FuzzMod()
        result = ENGINE.reflect("test", "out", state, mod, ())
        assert isinstance(result, ReflectionResult)
        assert 0.0 <= result.importance_score <= 1.0

    @given(bias=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_extreme_motivational_bias_no_crash(self, bias):
        state = FuzzState(pad=FuzzPAD())
        mod = FuzzMod(motivational_bias=bias)
        result = ENGINE.reflect("test", "output", state, mod, ())
        assert 0.0 <= result.importance_score <= 1.0
        assert result.goal_signal in VALID_GOAL_SIGNALS
        assert result.trajectory_signal in VALID_TRAJECTORY_SIGNALS


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: OUTPUT LENGTH EXTREMES
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzOutputLength:
    """assistant_output from 0 to 10,000 chars."""

    @given(length=st.integers(min_value=0, max_value=10_000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_variable_output_length_no_crash(self, length):
        output = "x" * length
        state = FuzzState(pad=FuzzPAD())
        mod = FuzzMod()
        result = ENGINE.reflect("input", output, state, mod, ())
        assert 0.0 <= result.importance_score <= 1.0
        assert len(result.summary) <= 200

    def test_empty_output(self):
        result = ENGINE.reflect("input", "", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)

    def test_10k_output(self):
        result = ENGINE.reflect("input", "a" * 10_000, FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert 0.0 <= result.importance_score <= 1.0
        assert len(result.summary) <= 200


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: MALFORMED TOOL RESULTS
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzMalformedTools:
    """Tool results with wrong types, missing attributes, etc."""

    def test_tool_results_as_list_of_ints(self):
        """tool_results containing non-objects."""
        result = ENGINE.reflect("test", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), (1, 2, 3))
        assert 0.0 <= result.importance_score <= 1.0

    def test_tool_results_with_none_elements(self):
        result = ENGINE.reflect("test", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), (None, None))
        assert isinstance(result, ReflectionResult)

    def test_tool_results_with_mixed_types(self):
        tools = (FuzzToolResult(), "string", 42, None, {"dict": True})
        result = ENGINE.reflect("test", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), tools)
        assert 0.0 <= result.importance_score <= 1.0

    def test_empty_tool_tuple(self):
        result = ENGINE.reflect("test", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)

    def test_large_tool_count(self):
        tools = tuple(FuzzToolResult(tool_name=f"t{i}") for i in range(100))
        result = ENGINE.reflect("test", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), tools)
        assert 0.0 <= result.importance_score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: NONE COMBINATIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzNoneCombinations:
    """All None/missing attribute combinations for state and modulation."""

    COMBOS = [
        (None, None),
        (None, FuzzMod()),
        (FuzzState(pad=FuzzPAD()), None),
        (FuzzState(pad=None), FuzzMod()),
        (FuzzState(pad=FuzzPAD()), FuzzMod()),
    ]

    @pytest.mark.parametrize("state,mod", COMBOS)
    def test_none_combo_no_crash(self, state, mod):
        result = ENGINE.reflect("test", "output", state, mod, ())
        assert isinstance(result, ReflectionResult)
        assert 0.0 <= result.importance_score <= 1.0
        assert result.goal_signal in VALID_GOAL_SIGNALS
        assert result.trajectory_signal in VALID_TRAJECTORY_SIGNALS

    def test_completely_bare_object(self):
        """Objects with no relevant attributes at all."""
        result = ENGINE.reflect("test", "out", object(), object(), ())
        assert isinstance(result, ReflectionResult)
        assert 0.0 <= result.importance_score <= 1.0


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: USER INPUT EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzUserInput:
    """Unicode, empty strings, control characters, newlines."""

    @given(user_input=st.text(min_size=0, max_size=5000))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_arbitrary_unicode_no_crash(self, user_input):
        result = ENGINE.reflect(user_input, "out", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)
        assert len(result.summary) <= 200

    def test_empty_user_input(self):
        result = ENGINE.reflect("", "output", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)

    def test_newlines_and_tabs(self):
        result = ENGINE.reflect("\n\n\t\t\r\n", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)

    def test_emoji_input(self):
        result = ENGINE.reflect("🔥💀🤖🧠❤️", "out", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert isinstance(result, ReflectionResult)

    def test_very_long_user_input(self):
        result = ENGINE.reflect("x" * 50_000, "out", FuzzState(pad=FuzzPAD()), FuzzMod(), ())
        assert len(result.summary) <= 200


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: EMOTION LABEL EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzEmotionLabels:
    """Unknown emotion labels, empty strings, numeric emotions."""

    EDGE_EMOTIONS = [
        "", "UNKNOWN", "123", "null", "None", "undefined",
        "fatigue", "drive", "doubt", "confidence", "flow",
        "frustration", "anxiety", "excitement", "neutral",
        "🔥", "a" * 500,
    ]

    @pytest.mark.parametrize("emotion", EDGE_EMOTIONS)
    def test_edge_emotion_label_no_crash(self, emotion):
        state = FuzzState(primary_emotion=emotion, label=emotion, pad=FuzzPAD())
        result = ENGINE.reflect("test", "out", state, FuzzMod(), ())
        assert isinstance(result, ReflectionResult)
        assert result.goal_signal in VALID_GOAL_SIGNALS
        assert result.trajectory_signal in VALID_TRAJECTORY_SIGNALS


# ═════════════════════════════════════════════════════════════════════════════
# FUZZ: COMBINED RANDOM BARRAGE
# ═════════════════════════════════════════════════════════════════════════════

class TestFuzzCombinedBarrage:
    """Full random barrage: all parameters randomized simultaneously."""

    @given(
        emotion=st.text(min_size=0, max_size=50),
        intensity=st.floats(allow_nan=False, allow_infinity=False),
        is_pos=st.booleans(),
        is_neg=st.booleans(),
        is_ha=st.booleans(),
        valence=st.floats(allow_nan=False, allow_infinity=False),
        tone=st.text(min_size=0, max_size=30),
        bias=st.floats(allow_nan=False, allow_infinity=False),
        validation=st.booleans(),
        user_input=st.text(min_size=0, max_size=1000),
        output_len=st.integers(min_value=0, max_value=5000),
        n_tools=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_full_random_barrage(
        self, emotion, intensity, is_pos, is_neg, is_ha,
        valence, tone, bias, validation, user_input, output_len, n_tools,
    ):
        """The engine must NEVER crash regardless of input combination."""
        state = FuzzState(
            primary_emotion=emotion,
            intensity=intensity,
            is_positive=is_pos,
            is_negative=is_neg,
            is_high_arousal=is_ha,
            label=emotion,
            pad=FuzzPAD(valence=valence),
        )
        mod = FuzzMod(
            tone=tone,
            motivational_bias=bias,
            emotional_validation=validation,
        )
        tools = tuple(range(n_tools))
        output = "x" * output_len

        result = ENGINE.reflect(user_input, output, state, mod, tools)

        # Hard invariants that must hold for ANY input
        assert isinstance(result, ReflectionResult)
        assert 0.0 <= result.importance_score <= 1.0
        assert len(result.summary) <= 200
        assert result.goal_signal in VALID_GOAL_SIGNALS
        assert result.trajectory_signal in VALID_TRAJECTORY_SIGNALS


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
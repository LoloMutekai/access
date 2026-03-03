"""
A.C.C.E.S.S. — Phase 4 Test Suite

Sections:
    1. TestRelationshipState     — clamping, engine, decay, dependency, serialization
    2. TestPersonalityState      — bounds, drift, homeostasis, serialization
    3. TestSelfModel             — voting, hysteresis, patterns, serialization
    4. TestGoalQueue             — add/pop/complete, ordering, decay, capacity, dedup
    5. TestPersistence           — roundtrip, corrupt, missing, atomic write
    6. TestIntegration           — 5-turn sequential simulation through all systems
    7. TestBackwardCompatibility — Phase 3 works without Phase 4 enabled
"""

import sys
import os
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, UTC, timedelta
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from agent.relationship_state import (
    RelationshipState, RelationshipEngine, RelationshipConfig,
    _clamp, _bounded, _drift_toward,
)
from agent.personality_state import (
    PersonalityTraits, TraitDriftEngine, PersonalityConfig, TRAIT_NAMES,
)
from agent.self_model import (
    SelfModel, SelfModelEngine, SelfModelConfig,
    INTERACTION_MODES, GUIDANCE_STYLES, GOAL_PATTERNS,
)
from agent.goal_queue import (
    InternalGoal, GoalQueue, GoalQueueConfig,
    VALID_ORIGINS, VALID_STATUSES,
)
from agent.persistence import IdentityStore, PersistenceConfig, SCHEMA_VERSION
from agent.agent_config import AgentConfig


# ─── Fake signal objects (duck-typed) ────────────────────────────────────────

@dataclass
class FakeReflection:
    goal_signal: Optional[str] = None
    importance_score: float = 0.5
    trajectory_signal: Optional[str] = None
    summary: str = "test"
    emotional_tags: list = None
    def __post_init__(self):
        if self.emotional_tags is None:
            self.emotional_tags = []

@dataclass
class FakeTrajectory:
    dominant_trajectory: Optional[str] = None
    dominant_goal_signal: Optional[str] = None
    drift_score: float = 0.0
    turn_count: int = 0

@dataclass
class FakeEmotion:
    primary_emotion: str = "neutral"
    intensity: float = 0.5
    is_positive: bool = False
    is_negative: bool = False
    is_high_arousal: bool = False

@dataclass
class FakeModulation:
    tone: str = "neutral"
    pacing: str = "normal"
    verbosity: str = "normal"
    structure_bias: str = "conversational"
    emotional_validation: bool = False
    motivational_bias: float = 0.0
    cognitive_load_limit: float = 0.5


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RELATIONSHIP STATE
# ═════════════════════════════════════════════════════════════════════════════

class TestRelationshipState:

    def test_default_values(self):
        s = RelationshipState()
        assert s.trust_level == 0.3
        assert s.respect_level == 0.3
        assert s.challenge_tolerance == 0.35
        assert s.dependency_risk == 0.0
        assert s.interaction_count == 0

    def test_clamping_high(self):
        s = RelationshipState(trust_level=5.0, respect_level=2.0, dependency_risk=99.0)
        assert s.trust_level == 1.0
        assert s.respect_level == 1.0
        assert s.dependency_risk == 1.0

    def test_clamping_low(self):
        s = RelationshipState(trust_level=-1.0, respect_level=-5.0)
        assert s.trust_level == 0.0
        assert s.respect_level == 0.0

    def test_frozen(self):
        s = RelationshipState()
        with pytest.raises(AttributeError):
            s.trust_level = 0.9

    def test_serialization_roundtrip(self):
        s = RelationshipState(trust_level=0.75, respect_level=0.6, interaction_count=42)
        d = s.to_dict()
        s2 = RelationshipState.from_dict(d)
        assert abs(s2.trust_level - 0.75) < 1e-6
        assert abs(s2.respect_level - 0.6) < 1e-6
        assert s2.interaction_count == 42

    def test_dependency_alarm(self):
        s = RelationshipState(dependency_risk=0.69)
        assert not s.dependency_alarm
        s2 = RelationshipState(dependency_risk=0.71)
        assert s2.dependency_alarm

    def test_trust_ratio(self):
        s = RelationshipState(positive_interactions=3, negative_interactions=1)
        assert abs(s.trust_ratio - 0.75) < 1e-6

    def test_trust_ratio_zero(self):
        s = RelationshipState()
        assert s.trust_ratio == 0.0


class TestRelationshipEngine:

    def test_positive_turn_builds_trust(self):
        engine = RelationshipEngine()
        s = RelationshipState()
        reflection = FakeReflection(goal_signal="push_forward")
        trajectory = FakeTrajectory(dominant_trajectory="progressing")
        emotion = FakeEmotion(is_positive=True)
        s2 = engine.update_from_turn(s, reflection, trajectory, emotion)
        assert s2.trust_level > s.trust_level
        assert s2.interaction_count == 1

    def test_negative_turn_erodes_trust(self):
        engine = RelationshipEngine()
        s = RelationshipState(trust_level=0.8)
        reflection = FakeReflection(goal_signal="stabilize")
        trajectory = FakeTrajectory(dominant_trajectory="declining")
        emotion = FakeEmotion(is_negative=True, primary_emotion="frustration")
        s2 = engine.update_from_turn(s, reflection, trajectory, emotion)
        assert s2.trust_level < s.trust_level

    def test_max_single_step_respected(self):
        engine = RelationshipEngine()
        s = RelationshipState(trust_level=0.5)
        # Extreme positive turn
        reflection = FakeReflection(goal_signal="execute")
        trajectory = FakeTrajectory(dominant_trajectory="escalating")
        emotion = FakeEmotion(is_positive=True)
        s2 = engine.update_from_turn(s, reflection, trajectory, emotion)
        assert abs(s2.trust_level - s.trust_level) <= 0.05 + 1e-9

    def test_all_values_remain_clamped_after_100_updates(self):
        engine = RelationshipEngine()
        s = RelationshipState()
        reflection = FakeReflection(goal_signal="push_forward")
        trajectory = FakeTrajectory(dominant_trajectory="progressing")
        emotion = FakeEmotion(is_positive=True)
        for _ in range(100):
            s = engine.update_from_turn(s, reflection, trajectory, emotion)
        assert 0.0 <= s.trust_level <= 1.0
        assert 0.0 <= s.respect_level <= 1.0
        assert 0.0 <= s.challenge_tolerance <= 1.0
        assert 0.0 <= s.dependency_risk <= 1.0
        assert s.interaction_count == 100

    def test_none_inputs_no_crash(self):
        engine = RelationshipEngine()
        s = RelationshipState()
        s2 = engine.update_from_turn(s, None, None, None)
        assert s2.interaction_count == 1
        assert 0.0 <= s2.trust_level <= 1.0

    def test_decay_over_time(self):
        engine = RelationshipEngine()
        s = RelationshipState(trust_level=0.8, dependency_risk=0.5)
        s2 = engine.decay_over_time(s, hours_elapsed=72)  # 3 days
        assert s2.trust_level < s.trust_level
        assert s2.dependency_risk < s.dependency_risk
        assert s2.interaction_count == s.interaction_count  # unchanged

    def test_decay_zero_hours_noop(self):
        engine = RelationshipEngine()
        s = RelationshipState(trust_level=0.8)
        s2 = engine.decay_over_time(s, hours_elapsed=0)
        assert s2.trust_level == s.trust_level

    def test_detect_dependency_risk_levels(self):
        engine = RelationshipEngine()
        assert engine.detect_dependency_risk(RelationshipState(dependency_risk=0.1))["risk_level"] == "safe"
        assert engine.detect_dependency_risk(RelationshipState(dependency_risk=0.25))["risk_level"] == "mild"
        assert engine.detect_dependency_risk(RelationshipState(dependency_risk=0.5))["risk_level"] == "elevated"
        assert engine.detect_dependency_risk(RelationshipState(dependency_risk=0.8))["risk_level"] == "critical"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PERSONALITY STATE
# ═════════════════════════════════════════════════════════════════════════════

class TestPersonalityState:

    def test_default_values(self):
        p = PersonalityTraits()
        assert p.assertiveness == 0.45
        assert p.warmth == 0.55
        assert p.emotional_stability == 0.55

    def test_clamping(self):
        p = PersonalityTraits(assertiveness=-1.0, warmth=5.0)
        assert p.assertiveness == 0.0
        assert p.warmth == 1.0

    def test_frozen(self):
        p = PersonalityTraits()
        with pytest.raises(AttributeError):
            p.warmth = 0.9

    def test_trait_vector(self):
        p = PersonalityTraits()
        assert len(p.trait_vector) == 5
        assert all(0.0 <= v <= 1.0 for v in p.trait_vector)

    def test_distance_self_is_zero(self):
        p = PersonalityTraits()
        assert p.distance_from(p) == 0.0

    def test_distance_symmetric(self):
        a = PersonalityTraits(assertiveness=0.2)
        b = PersonalityTraits(assertiveness=0.8)
        assert abs(a.distance_from(b) - b.distance_from(a)) < 1e-9

    def test_serialization_roundtrip(self):
        p = PersonalityTraits(warmth=0.8, total_drift_events=10)
        d = p.to_dict()
        p2 = PersonalityTraits.from_dict(d)
        assert abs(p2.warmth - 0.8) < 1e-6
        assert p2.total_drift_events == 10


class TestTraitDrift:

    def test_drift_bounded_per_update(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits()
        cap = 0.01
        reflection = FakeReflection(goal_signal="execute")
        trajectory = FakeTrajectory(dominant_trajectory="progressing")
        emotion = FakeEmotion(is_positive=True)
        modulation = FakeModulation(emotional_validation=True, structure_bias="structured")
        p2 = engine.compute_drift(p, reflection, trajectory, emotion, modulation)
        for name in TRAIT_NAMES:
            assert abs(getattr(p2, name) - getattr(p, name)) <= cap + 1e-9, \
                f"{name} drifted more than {cap}"

    def test_drift_events_increment(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits(total_drift_events=5)
        p2 = engine.compute_drift(p, None, None, None, None)
        assert p2.total_drift_events == 6

    def test_all_traits_clamped_after_1000_drifts(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits()
        reflection = FakeReflection(goal_signal="push_forward")
        trajectory = FakeTrajectory(dominant_trajectory="escalating")
        emotion = FakeEmotion(is_positive=True, primary_emotion="drive")
        modulation = FakeModulation(emotional_validation=True, structure_bias="structured")
        for _ in range(1000):
            p = engine.compute_drift(p, reflection, trajectory, emotion, modulation)
        for name in TRAIT_NAMES:
            v = getattr(p, name)
            assert 0.0 <= v <= 1.0, f"{name} = {v} out of bounds"

    def test_homeostasis_drifts_toward_baseline(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits(warmth=0.9, assertiveness=0.1)
        p2 = engine.apply_homeostasis(p, hours_elapsed=240)  # 10 days
        assert p2.warmth < p.warmth     # drifts toward 0.55
        assert p2.assertiveness > p.assertiveness  # drifts toward 0.45

    def test_homeostasis_zero_hours_noop(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits(warmth=0.9)
        p2 = engine.apply_homeostasis(p, hours_elapsed=0)
        assert p2.warmth == p.warmth

    def test_none_inputs_no_crash(self):
        engine = TraitDriftEngine()
        p = PersonalityTraits()
        p2 = engine.compute_drift(p, None, None, None, None)
        assert p2.total_drift_events == 1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SELF MODEL
# ═════════════════════════════════════════════════════════════════════════════

class TestSelfModel:

    def test_default_values(self):
        m = SelfModel()
        assert m.dominant_interaction_mode == "collaborating"
        assert m.preferred_guidance_style == "socratic"
        assert m.total_observations == 0

    def test_frozen(self):
        m = SelfModel()
        with pytest.raises(AttributeError):
            m.dominant_interaction_mode = "coaching"

    def test_serialization_roundtrip(self):
        m = SelfModel(
            dominant_interaction_mode="coaching",
            long_term_goal_pattern="achievement",
            recurring_user_patterns=("emotion:drive", "goal:execute"),
            total_observations=50,
        )
        d = m.to_dict()
        m2 = SelfModel.from_dict(d)
        assert m2.dominant_interaction_mode == "coaching"
        assert m2.long_term_goal_pattern == "achievement"
        assert m2.total_observations == 50
        assert "emotion:drive" in m2.recurring_user_patterns


class TestSelfModelEngine:

    def test_hysteresis_prevents_early_shift(self):
        """Mode should NOT shift before threshold (5 votes)."""
        engine = SelfModelEngine(SelfModelConfig(mode_shift_threshold=5))
        m = SelfModel()
        reflection = FakeReflection(goal_signal="push_forward")
        trajectory = FakeTrajectory(dominant_trajectory="progressing")
        emotion = FakeEmotion()
        modulation = FakeModulation()
        for _ in range(3):
            m = engine.observe_turn(m, reflection, trajectory, emotion, modulation)
        assert m.dominant_interaction_mode == "collaborating"  # unchanged

    def test_mode_shifts_after_threshold(self):
        """Mode SHOULD shift after sufficient votes."""
        engine = SelfModelEngine(SelfModelConfig(mode_shift_threshold=5))
        m = SelfModel()
        reflection = FakeReflection(goal_signal="push_forward")
        trajectory = FakeTrajectory(dominant_trajectory="progressing")
        emotion = FakeEmotion()
        modulation = FakeModulation()
        for _ in range(10):
            m = engine.observe_turn(m, reflection, trajectory, emotion, modulation)
        assert m.dominant_interaction_mode == "coaching"

    def test_pattern_accumulation(self):
        engine = SelfModelEngine(SelfModelConfig(recurring_tag_min_count=3))
        m = SelfModel()
        emotion = FakeEmotion(primary_emotion="drive")
        reflection = FakeReflection(goal_signal="execute")
        for _ in range(5):
            m = engine.observe_turn(m, reflection, None, emotion, None)
        assert "emotion:drive" in m.recurring_user_patterns
        assert "goal:execute" in m.recurring_user_patterns

    def test_observation_count(self):
        engine = SelfModelEngine()
        m = SelfModel()
        for _ in range(7):
            m = engine.observe_turn(m, None, None, None, None)
        assert m.total_observations == 7

    def test_none_inputs_no_crash(self):
        engine = SelfModelEngine()
        m = SelfModel()
        m2 = engine.observe_turn(m, None, None, None, None)
        assert m2.total_observations == 1


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GOAL QUEUE
# ═════════════════════════════════════════════════════════════════════════════

class TestInternalGoal:

    def test_default_values(self):
        g = InternalGoal()
        assert g.status == "active"
        assert 0.0 <= g.priority <= 1.0

    def test_clamping(self):
        g = InternalGoal(priority=-5.0)
        assert g.priority == 0.0
        g2 = InternalGoal(priority=10.0)
        assert g2.priority == 1.0

    def test_invalid_origin_corrected(self):
        g = InternalGoal(origin="invalid")
        assert g.origin == "reflection"

    def test_with_priority(self):
        g = InternalGoal(priority=0.5)
        g2 = g.with_priority(0.8)
        assert g2.priority == 0.8
        assert g2.id == g.id  # same identity

    def test_with_status(self):
        g = InternalGoal(status="active")
        g2 = g.with_status("completed")
        assert g2.status == "completed"
        assert g2.id == g.id

    def test_serialization_roundtrip(self):
        g = InternalGoal(description="test goal", priority=0.7, origin="user")
        d = g.to_dict()
        g2 = InternalGoal.from_dict(d)
        assert g2.description == "test goal"
        assert abs(g2.priority - 0.7) < 1e-6
        assert g2.origin == "user"


class TestGoalQueue:

    def test_add_and_list(self):
        q = GoalQueue()
        q.add_goal("Learn Rust", priority=0.8, origin="user")
        q.add_goal("Fix bug", priority=0.6)
        assert q.active_count == 2
        active = q.list_active()
        assert active[0].priority >= active[1].priority  # sorted desc

    def test_pop_highest(self):
        q = GoalQueue()
        q.add_goal("Low", priority=0.3)
        q.add_goal("High", priority=0.9)
        q.add_goal("Mid", priority=0.6)
        top = q.pop_highest_priority()
        assert top.description == "High"
        assert q.active_count == 2

    def test_pop_empty(self):
        q = GoalQueue()
        assert q.pop_highest_priority() is None

    def test_complete_goal(self):
        q = GoalQueue()
        g = q.add_goal("Test")
        assert q.complete_goal(g.id)
        assert q.active_count == 0
        assert not q.complete_goal("nonexistent")

    def test_deduplication(self):
        q = GoalQueue()
        g1 = q.add_goal("Same goal")
        g2 = q.add_goal("Same goal")
        assert g1 is not None
        assert g2 is None  # duplicate blocked
        assert q.active_count == 1

    def test_capacity_enforced(self):
        q = GoalQueue(GoalQueueConfig(max_goals=3))
        q.add_goal("A", priority=0.5)
        q.add_goal("B", priority=0.3)
        q.add_goal("C", priority=0.8)
        q.add_goal("D", priority=0.9)
        assert q.active_count == 3  # capacity = 3, lowest expired

    def test_decay_priorities(self):
        q = GoalQueue(GoalQueueConfig(priority_decay_rate=0.1, priority_floor=0.05))
        q.add_goal("Will expire", priority=0.08)
        q.add_goal("Will survive", priority=0.5)
        expired = q.decay_priorities()
        assert expired == 1
        assert q.active_count == 1

    def test_boost_priority(self):
        q = GoalQueue()
        g = q.add_goal("Test", priority=0.5)
        assert q.boost_priority(g.id, 0.2)
        active = q.list_active()
        assert active[0].priority > 0.5

    def test_serialization_roundtrip(self):
        q = GoalQueue()
        q.add_goal("A", priority=0.8, origin="user")
        q.add_goal("B", priority=0.3)
        d = q.to_dict()
        q2 = GoalQueue.from_dict(d)
        assert q2.total_count == 2
        active = q2.list_active()
        assert len(active) == 2


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PERSISTENCE
# ═════════════════════════════════════════════════════════════════════════════

class TestPersistence:

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        d = tmp_path / "identity"
        d.mkdir()
        return d

    @pytest.fixture
    def store(self, tmp_dir):
        return IdentityStore(PersistenceConfig(data_dir=tmp_dir))

    def test_relationship_roundtrip(self, store):
        s = RelationshipState(trust_level=0.8, interaction_count=42)
        assert store.save_relationship_state(s)
        loaded = store.load_relationship_state()
        assert abs(loaded.trust_level - 0.8) < 1e-6
        assert loaded.interaction_count == 42

    def test_personality_roundtrip(self, store):
        p = PersonalityTraits(warmth=0.9, total_drift_events=100)
        assert store.save_personality_state(p)
        loaded = store.load_personality_state()
        assert abs(loaded.warmth - 0.9) < 1e-6
        assert loaded.total_drift_events == 100

    def test_self_model_roundtrip(self, store):
        m = SelfModel(dominant_interaction_mode="coaching", total_observations=50)
        assert store.save_self_model(m)
        loaded = store.load_self_model()
        assert loaded.dominant_interaction_mode == "coaching"
        assert loaded.total_observations == 50

    def test_goal_queue_roundtrip(self, store):
        q = GoalQueue()
        q.add_goal("Test goal", priority=0.8)
        assert store.save_goal_queue(q)
        loaded = store.load_goal_queue()
        assert loaded.active_count == 1

    def test_save_all(self, store):
        ok = store.save_all(
            RelationshipState(), PersonalityTraits(), SelfModel(), GoalQueue()
        )
        assert ok

    def test_missing_file_returns_defaults(self, store):
        r = store.load_relationship_state()
        assert r.trust_level == 0.3  # default
        p = store.load_personality_state()
        assert p.warmth == 0.55

    def test_corrupt_json_returns_defaults(self, store):
        path = store._path("relationship")
        path.write_text("{{{invalid json", encoding="utf-8")
        r = store.load_relationship_state()
        assert r.trust_level == 0.3

    def test_backup_created(self, store):
        s = RelationshipState(trust_level=0.5)
        store.save_relationship_state(s)
        s2 = RelationshipState(trust_level=0.9)
        store.save_relationship_state(s2)
        backup = store._path("relationship").with_suffix(".json.bak")
        assert backup.exists()

    def test_schema_version_in_file(self, store):
        store.save_relationship_state(RelationshipState())
        with open(store._path("relationship")) as f:
            envelope = json.load(f)
        assert envelope["schema_version"] == SCHEMA_VERSION

    def test_exists_check(self, store):
        assert not store.exists("relationship")
        store.save_relationship_state(RelationshipState())
        assert store.exists("relationship")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTEGRATION (5-turn sequential simulation)
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_5_turn_simulation(self, tmp_path):
        """
        Simulate 5 turns through the full Phase 4 pipeline:
            Turn 1: positive push_forward
            Turn 2: positive execute
            Turn 3: negative stabilize (frustration)
            Turn 4: negative recover (fatigue)
            Turn 5: positive push_forward again

        Verify:
            - All states evolve correctly
            - Persistence round-trips
            - Goal queue accumulates
            - No crashes
        """
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path / "identity"))
        rel_engine = RelationshipEngine()
        trait_engine = TraitDriftEngine()
        self_engine = SelfModelEngine()

        # Load initial states (defaults)
        rel = store.load_relationship_state()
        per = store.load_personality_state()
        sm = store.load_self_model()
        goals = store.load_goal_queue()

        # Turn scenarios
        turns = [
            {
                "reflection": FakeReflection(goal_signal="push_forward", importance_score=0.7),
                "trajectory": FakeTrajectory(dominant_trajectory="progressing"),
                "emotion": FakeEmotion(is_positive=True, primary_emotion="confidence"),
                "modulation": FakeModulation(emotional_validation=False, structure_bias="structured"),
                "goal_desc": "Improve project architecture",
            },
            {
                "reflection": FakeReflection(goal_signal="execute", importance_score=0.8),
                "trajectory": FakeTrajectory(dominant_trajectory="progressing"),
                "emotion": FakeEmotion(is_positive=True, primary_emotion="drive"),
                "modulation": FakeModulation(tone="energizing", structure_bias="structured"),
                "goal_desc": "Implement Phase 4 modules",
            },
            {
                "reflection": FakeReflection(goal_signal="stabilize", importance_score=0.6),
                "trajectory": FakeTrajectory(dominant_trajectory="declining"),
                "emotion": FakeEmotion(is_negative=True, primary_emotion="frustration"),
                "modulation": FakeModulation(tone="calm", emotional_validation=True),
                "goal_desc": None,
            },
            {
                "reflection": FakeReflection(goal_signal="recover", importance_score=0.5),
                "trajectory": FakeTrajectory(dominant_trajectory="declining"),
                "emotion": FakeEmotion(is_negative=True, primary_emotion="fatigue"),
                "modulation": FakeModulation(tone="grounding", emotional_validation=True),
                "goal_desc": None,
            },
            {
                "reflection": FakeReflection(goal_signal="push_forward", importance_score=0.75),
                "trajectory": FakeTrajectory(dominant_trajectory="progressing"),
                "emotion": FakeEmotion(is_positive=True, primary_emotion="confidence"),
                "modulation": FakeModulation(structure_bias="conversational"),
                "goal_desc": "Refactor test suite",
            },
        ]

        for i, turn in enumerate(turns):
            # Update relationship
            rel = rel_engine.update_from_turn(
                rel, turn["reflection"], turn["trajectory"], turn["emotion"]
            )
            # Update personality
            per = trait_engine.compute_drift(
                per, turn["reflection"], turn["trajectory"],
                turn["emotion"], turn["modulation"],
            )
            # Update self-model
            sm = self_engine.observe_turn(
                sm, turn["reflection"], turn["trajectory"],
                turn["emotion"], turn["modulation"],
            )
            # Add goal if present
            if turn["goal_desc"]:
                goals.add_goal(turn["goal_desc"], origin="reflection")

            # Verify invariants every turn
            assert 0.0 <= rel.trust_level <= 1.0
            assert 0.0 <= rel.respect_level <= 1.0
            assert 0.0 <= rel.dependency_risk <= 1.0
            for name in TRAIT_NAMES:
                assert 0.0 <= getattr(per, name) <= 1.0

        # After 5 turns
        assert rel.interaction_count == 5
        assert per.total_drift_events == 5
        assert sm.total_observations == 5
        assert goals.active_count == 3  # 3 turns had goals

        # Persistence round-trip
        assert store.save_all(rel, per, sm, goals)
        rel2 = store.load_relationship_state()
        per2 = store.load_personality_state()
        sm2 = store.load_self_model()
        goals2 = store.load_goal_queue()

        assert rel2.interaction_count == 5
        assert per2.total_drift_events == 5
        assert sm2.total_observations == 5
        assert goals2.active_count == 3

    def test_decay_between_sessions(self, tmp_path):
        """Simulate time passage between sessions."""
        store = IdentityStore(PersistenceConfig(data_dir=tmp_path / "identity"))
        rel_engine = RelationshipEngine()
        trait_engine = TraitDriftEngine()

        # Session 1: build up trust
        rel = RelationshipState(trust_level=0.8, dependency_risk=0.4)
        per = PersonalityTraits(warmth=0.9, assertiveness=0.1)

        # Simulate 48 hours of inactivity
        rel = rel_engine.decay_over_time(rel, hours_elapsed=48)
        per = trait_engine.apply_homeostasis(per, hours_elapsed=48)

        # Trust should have decayed toward baseline
        assert rel.trust_level < 0.8
        assert rel.dependency_risk < 0.4
        # Personality should drift toward baselines
        assert per.warmth < 0.9
        assert per.assertiveness > 0.1

        # Save and reload
        store.save_all(rel, per, SelfModel(), GoalQueue())
        rel2 = store.load_relationship_state()
        assert abs(rel2.trust_level - rel.trust_level) < 1e-4


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6b — COGNITIVE IDENTITY MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class TestCognitiveIdentityManager:
    """Tests for the unified Phase 4 orchestrator."""

    def _make_config(self, tmp_path, **overrides):
        defaults = dict(
            enable_relationship_tracking=True,
            enable_personality_drift=True,
            enable_self_model=True,
            enable_goal_queue=True,
            enable_persistence=True,
            identity_data_dir=str(tmp_path / "identity"),
            persist_every_n_turns=0,  # manual save only
        )
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def test_load_save_cycle(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path)
        mgr = CognitiveIdentityManager(cfg)
        mgr.load()  # defaults
        assert mgr.relationship.trust_level == 0.3
        # Update
        r = FakeReflection(goal_signal="push_forward", importance_score=0.7)
        t = FakeTrajectory(dominant_trajectory="progressing")
        e = FakeEmotion(is_positive=True, primary_emotion="confidence")
        m = FakeModulation()
        mgr.update_from_turn(r, t, e, m)
        assert mgr.relationship.interaction_count == 1
        # Save and reload in new manager
        assert mgr.save()
        mgr2 = CognitiveIdentityManager(cfg)
        mgr2.load()
        assert mgr2.relationship.interaction_count == 1

    def test_auto_save(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path, persist_every_n_turns=2)
        mgr = CognitiveIdentityManager(cfg)
        mgr.load()
        r = FakeReflection(goal_signal="execute", importance_score=0.6)
        e = FakeEmotion(is_positive=True)
        # Turn 1: no auto-save yet
        mgr.update_from_turn(r, None, e, None)
        # Turn 2: auto-save triggers
        mgr.update_from_turn(r, None, e, None)
        # Verify persisted
        mgr2 = CognitiveIdentityManager(cfg)
        mgr2.load()
        assert mgr2.relationship.interaction_count == 2

    def test_all_off_noop(self, tmp_path):
        """With all Phase 4 flags off, update_from_turn is a pure no-op."""
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(
            tmp_path,
            enable_relationship_tracking=False,
            enable_personality_drift=False,
            enable_self_model=False,
            enable_goal_queue=False,
            enable_persistence=False,
        )
        mgr = CognitiveIdentityManager(cfg)
        mgr.update_from_turn(None, None, None, None)
        assert mgr.relationship.interaction_count == 0
        assert mgr.personality.total_drift_events == 0

    def test_session_decay(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path)
        mgr = CognitiveIdentityManager(cfg)
        # Manually set high values
        mgr._relationship = RelationshipState(trust_level=0.9, dependency_risk=0.6)
        mgr._personality = PersonalityTraits(warmth=0.95)
        mgr.apply_session_decay(hours_elapsed=120)
        assert mgr.relationship.trust_level < 0.9
        assert mgr.relationship.dependency_risk < 0.6
        assert mgr.personality.warmth < 0.95

    def test_identity_snapshot(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path)
        mgr = CognitiveIdentityManager(cfg)
        snap = mgr.identity_snapshot()
        assert "relationship" in snap
        assert "personality" in snap
        assert "self_model" in snap
        assert "goals" in snap
        assert "dependency_analysis" in snap

    def test_goal_injection_from_reflection(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path)
        mgr = CognitiveIdentityManager(cfg)
        # High-importance push_forward should inject a goal
        r = FakeReflection(goal_signal="push_forward", importance_score=0.8)
        t = FakeTrajectory(drift_score=0.1)
        mgr.update_from_turn(r, t, None, None)
        assert mgr.goal_queue.active_count >= 1

    def test_low_importance_no_goal(self, tmp_path):
        from agent.cognitive_identity import CognitiveIdentityManager
        cfg = self._make_config(tmp_path)
        mgr = CognitiveIdentityManager(cfg)
        r = FakeReflection(goal_signal="push_forward", importance_score=0.3)
        mgr.update_from_turn(r, None, None, None)
        # Low importance → no goal added (but decay ran so there may be 0)
        assert mgr.goal_queue.active_count == 0


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BACKWARD COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:

    def test_agent_config_defaults_phase4_off(self):
        """All Phase 4 flags should default to False."""
        cfg = AgentConfig()
        assert cfg.enable_relationship_tracking is False
        assert cfg.enable_personality_drift is False
        assert cfg.enable_self_model is False
        assert cfg.enable_goal_queue is False
        assert cfg.enable_persistence is False

    def test_phase3_config_fields_still_exist(self):
        """Phase 3 fields must still be present."""
        cfg = AgentConfig()
        assert hasattr(cfg, "enable_reflection")
        assert hasattr(cfg, "trajectory_window_size")
        assert hasattr(cfg, "enable_structured_logging")
        assert hasattr(cfg, "memory_decay_enabled")
        assert hasattr(cfg, "adaptive_importance")

    def test_phase3_modules_still_importable(self):
        """Phase 3 modules must still work independently."""
        from agent.reflection_engine import ReflectionEngine, ReflectionResult
        from agent.trajectory import TrajectoryTracker, TrajectoryState
        from agent.logger import StructuredLogger
        engine = ReflectionEngine()
        tracker = TrajectoryTracker()
        logger = StructuredLogger()
        assert tracker.window_size == 10

    def test_from_dict_handles_empty(self):
        """from_dict({}) should return clean defaults, not crash."""
        r = RelationshipState.from_dict({})
        p = PersonalityTraits.from_dict({})
        m = SelfModel.from_dict({})
        assert r.trust_level == 0.3
        assert p.warmth == 0.55
        assert m.dominant_interaction_mode == "collaborating"
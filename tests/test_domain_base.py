"""
Tests for agent/domain/base.py

Coverage:
    DomainEngineMetadata  — construction, validation, serialization
    DomainEngine ABC      — contract enforcement, helpers
"""

import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.domain.base import DomainEngine, DomainEngineMetadata


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL CONCRETE ENGINE FOR TESTING
# ─────────────────────────────────────────────────────────────────────────────

class MinimalEngine(DomainEngine):
    @property
    def name(self) -> str:
        return "minimal_engine"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_permissions(self) -> frozenset:
        return frozenset({"read"})

    def run(self, input_data, sandbox_context=None) -> dict:
        return {
            "status":  "ok",
            "output":  dict(input_data),
            "engine":  self.name,
            "version": self.version,
        }

    def benchmark(self) -> dict:
        return {
            "engine":  self.name,
            "version": self.version,
            "score":   1.0,
            "metrics": {},
            "passed":  True,
        }

    def self_test(self) -> dict:
        return {
            "engine":  self.name,
            "version": self.version,
            "checks":  [{"name": "trivial", "passed": True, "detail": "ok"}],
            "passed":  True,
        }

    def deterministic_check(self) -> bool:
        canonical = {"probe": True}
        o1 = self.run(canonical)
        o2 = self.run(canonical)
        return o1 == o2


# ─────────────────────────────────────────────────────────────────────────────
# METADATA TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainEngineMetadata:

    def test_construction_valid(self):
        m = DomainEngineMetadata(
            name="test_engine",
            version="1.0.0",
            required_permissions=frozenset({"read", "write"}),
        )
        assert m.name == "test_engine"
        assert m.version == "1.0.0"
        assert "read" in m.required_permissions

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            DomainEngineMetadata(
                name="",
                version="1.0.0",
                required_permissions=frozenset(),
            )

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError):
            DomainEngineMetadata(
                name="   ",
                version="1.0.0",
                required_permissions=frozenset(),
            )

    def test_empty_version_raises(self):
        with pytest.raises(ValueError):
            DomainEngineMetadata(
                name="engine",
                version="",
                required_permissions=frozenset(),
            )

    def test_to_dict_json_serializable(self):
        m = DomainEngineMetadata(
            name="engine",
            version="2.0.0",
            required_permissions=frozenset({"read"}),
            description="Test engine",
        )
        d = m.to_dict()
        json.dumps(d)  # must not raise

    def test_to_dict_contains_required_keys(self):
        m = DomainEngineMetadata(
            name="engine",
            version="1.0.0",
            required_permissions=frozenset(),
        )
        d = m.to_dict()
        assert "name" in d
        assert "version" in d
        assert "required_permissions" in d
        assert "description" in d

    def test_to_dict_permissions_sorted(self):
        m = DomainEngineMetadata(
            name="e",
            version="1.0.0",
            required_permissions=frozenset({"z_perm", "a_perm", "m_perm"}),
        )
        perms = m.to_dict()["required_permissions"]
        assert perms == sorted(perms)

    def test_frozen_immutable(self):
        m = DomainEngineMetadata(
            name="engine",
            version="1.0.0",
            required_permissions=frozenset(),
        )
        with pytest.raises((AttributeError, TypeError)):
            m.name = "other"  # type: ignore[misc]

    def test_repr_contains_name(self):
        m = DomainEngineMetadata(
            name="my_engine",
            version="1.0.0",
            required_permissions=frozenset(),
        )
        assert "my_engine" in repr(m)

    def test_description_defaults_to_empty_string(self):
        m = DomainEngineMetadata(
            name="engine",
            version="1.0.0",
            required_permissions=frozenset(),
        )
        assert m.description == ""


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN ENGINE ABC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainEngineContract:

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            DomainEngine()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        engine = MinimalEngine()
        assert engine is not None

    def test_name_property(self):
        engine = MinimalEngine()
        assert engine.name == "minimal_engine"

    def test_version_property(self):
        engine = MinimalEngine()
        assert engine.version == "0.1.0"

    def test_required_permissions_is_frozenset(self):
        engine = MinimalEngine()
        assert isinstance(engine.required_permissions, frozenset)

    def test_run_returns_dict(self):
        engine = MinimalEngine()
        result = engine.run({"x": 1})
        assert isinstance(result, dict)

    def test_run_contains_status(self):
        engine = MinimalEngine()
        result = engine.run({"x": 1})
        assert "status" in result

    def test_run_contains_engine_name(self):
        engine = MinimalEngine()
        result = engine.run({"x": 1})
        assert result["engine"] == "minimal_engine"

    def test_run_does_not_mutate_input(self):
        engine = MinimalEngine()
        original = {"key": "value"}
        snapshot = dict(original)
        engine.run(original)
        assert original == snapshot

    def test_run_json_serializable_output(self):
        engine = MinimalEngine()
        result = engine.run({"key": "value"})
        json.dumps(result)  # must not raise

    def test_benchmark_returns_dict(self):
        engine = MinimalEngine()
        result = engine.benchmark()
        assert isinstance(result, dict)

    def test_benchmark_contains_score(self):
        engine = MinimalEngine()
        result = engine.benchmark()
        assert "score" in result
        assert math.isfinite(result["score"])

    def test_benchmark_score_in_range(self):
        engine = MinimalEngine()
        score = engine.benchmark()["score"]
        assert 0.0 <= score <= 1.0

    def test_self_test_returns_dict(self):
        engine = MinimalEngine()
        result = engine.self_test()
        assert isinstance(result, dict)

    def test_self_test_contains_passed_field(self):
        engine = MinimalEngine()
        result = engine.self_test()
        assert "passed" in result
        assert isinstance(result["passed"], bool)

    def test_self_test_checks_is_list(self):
        engine = MinimalEngine()
        checks = engine.self_test().get("checks", [])
        assert isinstance(checks, list)

    def test_deterministic_check_returns_bool(self):
        engine = MinimalEngine()
        result = engine.deterministic_check()
        assert isinstance(result, bool)

    def test_deterministic_check_passes(self):
        engine = MinimalEngine()
        assert engine.deterministic_check() is True

    def test_metadata_returns_metadata_object(self):
        engine = MinimalEngine()
        meta = engine.metadata()
        assert isinstance(meta, DomainEngineMetadata)
        assert meta.name == "minimal_engine"

    def test_validate_input_non_dict_returns_errors(self):
        engine = MinimalEngine()
        errors = engine.validate_input("not_a_dict")  # type: ignore[arg-type]
        assert len(errors) > 0

    def test_validate_input_dict_returns_empty_errors(self):
        engine = MinimalEngine()
        errors = engine.validate_input({"key": "value"})
        assert errors == []

    def test_check_floats_finite_with_nan(self):
        engine = MinimalEngine()
        assert engine._check_floats_finite({"v": float("nan")}) is False

    def test_check_floats_finite_with_inf(self):
        engine = MinimalEngine()
        assert engine._check_floats_finite({"v": float("inf")}) is False

    def test_check_floats_finite_with_valid_values(self):
        engine = MinimalEngine()
        assert engine._check_floats_finite({"v": 0.5, "nested": [1.0, 2.0]}) is True

    def test_assert_json_serializable_raises_on_bad_input(self):
        engine = MinimalEngine()
        with pytest.raises(ValueError):
            engine._assert_json_serializable({"bad": object()}, "test")

    def test_repr_contains_class_name(self):
        engine = MinimalEngine()
        assert "MinimalEngine" in repr(engine)
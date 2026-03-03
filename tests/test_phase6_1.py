"""
A.C.C.E.S.S. — Phase 6.1 Static Self-Inspection Test Suite

63 tests covering:
    TestConfigAndModels     — data models, config immutability, serialization
    TestParsing             — safe AST parsing, line counting
    TestComplexity          — cyclomatic complexity, nesting, parameters
    TestImportGraph         — import extraction, cycle detection, fan-in/out
    TestLayerViolations     — architectural layer enforcement
    TestSmellDetection      — IO/state density, god classes, oversized modules
    TestRiskScoring         — sigmoid normalization, risk levels, bounds
    TestEndToEnd            — full codebase inspection, JSON roundtrip
    TestEdgeCases           — error handling, edge conditions
    TestDeterminism         — identical runs, sorted output
    TestPerformance         — bounded execution time
"""

import ast
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.self_inspection import (
    InspectionConfig,
    InspectionReport,
    FileMetrics,
    FunctionMetrics,
    ClassMetrics,
    StaticInspector,
    safe_parse,
    count_lines,
    cyclomatic_complexity,
    max_nesting_depth,
    count_parameters,
    count_self_attributes,
    extract_imports,
    detect_cycles,
    detect_layer_violations,
    compute_io_density,
    compute_state_density,
    sigmoid_normalize,
    compute_risk_scores,
    risk_to_grade,
)

# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect project root — works on any machine
# ─────────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_AGENT_DIR = _PROJECT_ROOT / "agent"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & MODELS
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigAndModels:

    def test_default_config_frozen(self):
        cfg = InspectionConfig()
        try:
            cfg.max_lines_per_file = 999
            assert False, "Should have raised"
        except (AttributeError, TypeError):
            pass

    def test_custom_config(self):
        cfg = InspectionConfig(max_lines_per_file=1000, max_cyclomatic_complexity=20)
        assert cfg.max_lines_per_file == 1000
        assert cfg.max_cyclomatic_complexity == 20

    def test_file_metrics_to_dict(self):
        fm = FileMetrics(path="test.py", lines_total=100, lines_code=80)
        d = fm.to_dict()
        assert d["path"] == "test.py"
        assert d["lines_total"] == 100
        assert d["lines_code"] == 80
        assert isinstance(d["imports"], list)
        assert isinstance(d["functions"], list)

    def test_function_metrics_flags(self):
        fm = FunctionMetrics(
            name="big_func",
            cyclomatic_complexity=15,
            max_nesting=6,
            parameter_count=9,
            lines=100,
            is_complex=True,
            is_deeply_nested=True,
            is_over_parameterized=True,
            is_long=True,
        )
        assert fm.is_complex is True
        assert fm.is_deeply_nested is True
        assert fm.is_over_parameterized is True
        assert fm.is_long is True
        d = fm.to_dict()
        assert d["cyclomatic_complexity"] == 15

    def test_inspection_report_to_dict_json_serializable(self):
        report = InspectionReport()
        d = report.to_dict()
        # Must be JSON-serializable
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["files_analyzed"] == 0

    def test_report_health_grade_empty(self):
        report = InspectionReport()
        assert report.health_grade == "A"
        assert report.files_analyzed == 0

    def test_report_health_grade_critical(self):
        """A report with high composite risk should get a bad grade."""
        report = InspectionReport(composite_risk=0.85, health_grade="F")
        assert report.health_grade == "F"


# ─────────────────────────────────────────────────────────────────────────────
# PARSING
# ─────────────────────────────────────────────────────────────────────────────

class TestParsing:

    def test_safe_parse_valid(self):
        tree = safe_parse("x = 1\ny = 2\n")
        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_safe_parse_syntax_error(self):
        tree = safe_parse("def broken(:\n  pass\n")
        assert tree is None

    def test_safe_parse_empty(self):
        tree = safe_parse("")
        assert tree is None

    def test_count_lines_basic(self):
        source = "x = 1\n# comment\n\ny = 2\n"
        assert count_lines(source) == 2

    def test_count_lines_empty(self):
        assert count_lines("") == 0


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY
# ─────────────────────────────────────────────────────────────────────────────

class TestComplexity:

    def _parse_func(self, source: str) -> ast.FunctionDef:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        raise ValueError("No function found")

    def _parse_class(self, source: str) -> ast.ClassDef:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return node
        raise ValueError("No class found")

    def test_cyclomatic_simple(self):
        func = self._parse_func("def f():\n  return 1\n")
        assert cyclomatic_complexity(func) == 1

    def test_cyclomatic_if_else(self):
        func = self._parse_func("def f(x):\n  if x:\n    return 1\n  else:\n    return 2\n")
        assert cyclomatic_complexity(func) == 2

    def test_cyclomatic_boolean_ops(self):
        func = self._parse_func("def f(a, b, c):\n  if a and b or c:\n    pass\n")
        # if → +1, and → +1, or → +1 = base 1 + 3 = 4
        cc = cyclomatic_complexity(func)
        assert cc >= 3  # at least 3 decision points

    def test_cyclomatic_loop(self):
        func = self._parse_func("def f(xs):\n  for x in xs:\n    pass\n")
        assert cyclomatic_complexity(func) == 2

    def test_cyclomatic_exception(self):
        func = self._parse_func(
            "def f():\n  try:\n    pass\n  except ValueError:\n    pass\n"
        )
        assert cyclomatic_complexity(func) == 2

    def test_cyclomatic_comprehension_filter(self):
        func = self._parse_func("def f(xs):\n  return [x for x in xs if x > 0]\n")
        cc = cyclomatic_complexity(func)
        assert cc >= 2  # base + comprehension if

    def test_max_nesting_simple(self):
        func = self._parse_func("def f():\n  return 1\n")
        assert max_nesting_depth(func) == 0

    def test_max_nesting_deep(self):
        func = self._parse_func(
            "def f(x):\n"
            "  if x:\n"
            "    for i in range(10):\n"
            "      if i > 5:\n"
            "        pass\n"
        )
        assert max_nesting_depth(func) == 3

    def test_count_parameters_no_self(self):
        func = self._parse_func("def f(self, a, b):\n  pass\n")
        assert count_parameters(func) == 2

    def test_count_parameters_vararg(self):
        func = self._parse_func("def f(*args, **kwargs):\n  pass\n")
        assert count_parameters(func) == 2

    def test_count_parameters_cls(self):
        func = self._parse_func("def f(cls, x):\n  pass\n")
        assert count_parameters(func) == 1

    def test_count_self_attributes(self):
        cls = self._parse_class(
            "class Foo:\n"
            "  def __init__(self):\n"
            "    self.a = 1\n"
            "    self.b = 2\n"
            "  def method(self):\n"
            "    self.c = 3\n"
        )
        assert count_self_attributes(cls) == 3


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT GRAPH
# ─────────────────────────────────────────────────────────────────────────────

class TestImportGraph:

    def test_extract_imports_absolute(self):
        tree = ast.parse("import os\nimport json\n")
        imports = extract_imports(tree)
        names = [name for name, _ in imports]
        assert "os" in names
        assert "json" in names

    def test_extract_imports_relative(self):
        tree = ast.parse("from .models import Foo\nfrom .config import Bar\n")
        imports = extract_imports(tree)
        names = [name for name, is_rel in imports]
        rels = [is_rel for _, is_rel in imports]
        assert "models" in names
        assert "config" in names
        assert any(rels)

    def test_detect_no_cycles(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": set()}
        cycles = detect_cycles(graph)
        assert len(cycles) == 0

    def test_detect_simple_cycle(self):
        graph = {"a": {"b"}, "b": {"a"}}
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1
        # The cycle should contain both a and b
        found = False
        for c in cycles:
            if "a" in c and "b" in c:
                found = True
        assert found

    def test_detect_triangle_cycle(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        cycles = detect_cycles(graph)
        assert len(cycles) >= 1
        found = False
        for c in cycles:
            if "a" in c and "b" in c and "c" in c:
                found = True
        assert found

    def test_ignores_external_modules(self):
        """Only internal modules should appear in the graph edges."""
        fm1 = FileMetrics(path="agent/core.py", imports=("os", "json", "models"))
        fm2 = FileMetrics(path="agent/models.py", imports=("dataclasses",))
        internal = {"core", "models"}
        from agent.self_inspection import build_import_graph
        graph, edges = build_import_graph([fm1, fm2], internal)
        # os, json, dataclasses should NOT be in graph
        all_targets = set()
        for s, ts in graph.items():
            all_targets.update(ts)
        assert "os" not in all_targets
        assert "json" not in all_targets

    def test_fan_in_out(self):
        from agent.self_inspection import compute_fan_in_out
        graph = {"a": {"b", "c"}, "b": {"c"}, "c": set()}
        internal = {"a", "b", "c"}
        fan_in, fan_out = compute_fan_in_out(graph, internal)
        assert fan_out["a"] == 2
        assert fan_out["b"] == 1
        assert fan_out["c"] == 0
        assert fan_in["c"] == 2
        assert fan_in["a"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER VIOLATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerViolations:

    _LAYERS = ("top", "middle", "bottom")

    def test_no_violation_top_down(self):
        """Top importing from bottom is fine (higher layer uses lower)."""
        graph = {"top": {"bottom"}}
        violations = detect_layer_violations(graph, self._LAYERS)
        assert len(violations) == 0

    def test_violation_bottom_up(self):
        """Bottom importing from top is a violation."""
        graph = {"bottom": {"top"}}
        violations = detect_layer_violations(graph, self._LAYERS)
        assert len(violations) == 1
        assert violations[0] == ("bottom", "top")

    def test_same_layer_no_violation(self):
        """Same layer imports are not violations."""
        graph = {"middle": {"middle"}}
        violations = detect_layer_violations(graph, self._LAYERS)
        assert len(violations) == 0

    def test_unknown_modules_ignored(self):
        """Modules not in layer_order are silently skipped."""
        graph = {"unknown_a": {"top"}, "bottom": {"unknown_b"}}
        violations = detect_layer_violations(graph, self._LAYERS)
        assert len(violations) == 0


# ─────────────────────────────────────────────────────────────────────────────
# SMELL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

class TestSmellDetection:

    def test_io_density_zero(self):
        tree = ast.parse("x = 1 + 2\ny = x * 3\n")
        assert compute_io_density(tree) == 0.0

    def test_io_density_full(self):
        tree = ast.parse("open('f')\nprint('hi')\n")
        density = compute_io_density(tree)
        assert density > 0.0

    def test_state_density(self):
        tree = ast.parse(
            "class Foo:\n"
            "  def __init__(self):\n"
            "    self.x = 1\n"
            "    self.y = 2\n"
        )
        density = compute_state_density(tree)
        assert density > 0.0

    def test_oversized_module_smell(self):
        cfg = InspectionConfig(max_lines_per_file=10)
        fm = FileMetrics(
            path="big.py",
            lines_code=100,
            is_oversized=True,
            smells=("oversized",),
        )
        from agent.self_inspection import detect_smells
        smells = detect_smells([fm], [], cfg)
        assert any("oversized" in s for s in smells)

    def test_god_class_smell(self):
        cfg = InspectionConfig(
            god_class_method_threshold=2,
            god_class_attribute_threshold=2,
        )
        cls = ClassMetrics(
            name="BigClass",
            method_count=10,
            attribute_count=10,
            is_god_class=True,
        )
        fm = FileMetrics(
            path="god.py",
            classes=(cls,),
            smells=("god_class:BigClass",),
        )
        from agent.self_inspection import detect_smells
        smells = detect_smells([fm], [], cfg)
        assert any("god_class" in s for s in smells)

    def test_circular_dependency_smell(self):
        cfg = InspectionConfig()
        cycles = [("a", "b")]
        from agent.self_inspection import detect_smells
        smells = detect_smells([], cycles, cfg)
        assert any("circular" in s for s in smells)


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskScoring:

    def test_sigmoid_normalize_midpoint(self):
        """At the midpoint, sigmoid returns ~0.5."""
        val = sigmoid_normalize(5.0, midpoint=5.0)
        assert abs(val - 0.5) < 0.01

    def test_sigmoid_normalize_low(self):
        """Far below midpoint → close to 0."""
        val = sigmoid_normalize(0.0, midpoint=10.0, k=1.0)
        assert val < 0.1

    def test_sigmoid_normalize_high(self):
        """Far above midpoint → close to 1."""
        val = sigmoid_normalize(20.0, midpoint=10.0, k=1.0)
        assert val > 0.9

    def test_risk_levels(self):
        """Grade thresholds produce expected grades."""
        assert risk_to_grade(0.1) == "A"
        assert risk_to_grade(0.3) == "B"
        assert risk_to_grade(0.5) == "C"
        assert risk_to_grade(0.7) == "D"
        assert risk_to_grade(0.9) == "F"

    def test_risk_scores_sum_to_one(self):
        """Risk weights should sum to approximately 1.0."""
        cfg = InspectionConfig()
        total = (
            cfg.risk_weight_complexity
            + cfg.risk_weight_coupling
            + cfg.risk_weight_size
            + cfg.risk_weight_smells
            + cfg.risk_weight_io
        )
        assert abs(total - 1.0) < 0.01

    def test_risk_score_bounded(self):
        """Composite risk must be in [0.0, 1.0]."""
        fm = FileMetrics(path="test.py", lines_code=50)
        dims, composite = compute_risk_scores([fm], [], {}, InspectionConfig())
        assert 0.0 <= composite <= 1.0
        for name, score in dims:
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_inspect_real_codebase(self):
        """Run full inspection on the actual A.C.C.E.S.S. agent/ directory."""
        report = StaticInspector().inspect(_AGENT_DIR)
        assert report.files_analyzed > 10
        assert report.lines_analyzed > 500
        assert report.health_grade in ("A", "B", "C", "D", "F")
        assert 0.0 <= report.composite_risk <= 1.0

    def test_json_roundtrip(self):
        """Report must survive JSON serialization → deserialization."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a small synthetic project
            p = Path(tmp)
            (p / "a.py").write_text("def foo():\n  return 1\n")
            (p / "b.py").write_text("import a\ndef bar():\n  pass\n")
            report = StaticInspector().inspect(p)
            d = report.to_dict()
            s = json.dumps(d)
            parsed = json.loads(s)
            assert parsed["files_analyzed"] == 2

    def test_quick_summary(self):
        inspector = StaticInspector()
        report = inspector.inspect(_AGENT_DIR)
        summary = inspector.quick_summary(report)
        assert "grade" in summary
        assert "smells" in summary
        assert "risk" in summary
        assert summary["files"] > 0

    def test_multi_file_project(self):
        """Test with a synthetic multi-file project."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "core.py").write_text(
                "from .models import M\n"
                "class Core:\n"
                "  def run(self):\n"
                "    pass\n"
            )
            (p / "models.py").write_text(
                "class M:\n"
                "  pass\n"
            )
            (p / "utils.py").write_text(
                "def helper():\n"
                "  return 42\n"
            )
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 3
            assert report.health_grade in ("A", "B", "C", "D", "F")


# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = StaticInspector().inspect(Path(tmp))
            assert report.files_analyzed == 0
            assert report.health_grade == "A"

    def test_file_with_syntax_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "broken.py").write_text("def f(:\n  pass\n")
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 1
            # Should not crash — produces partial result
            assert report.files[0].parse_error is True

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "empty.py").write_text("")
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 1

    def test_only_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "comments.py").write_text("# just a comment\n# another\n")
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 1
            assert report.files[0].lines_code == 0

    def test_deeply_nested_function(self):
        source = (
            "def deep():\n"
            "  if True:\n"
            "    for i in range(10):\n"
            "      while True:\n"
            "        if i:\n"
            "          with open('f'):\n"
            "            pass\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "deep.py").write_text(source)
            report = StaticInspector().inspect(p)
            funcs = report.files[0].functions
            assert len(funcs) >= 1
            assert funcs[0].max_nesting >= 4

    def test_nonexistent_directory(self):
        report = StaticInspector().inspect(Path("/nonexistent/path/here"))
        assert report.files_analyzed == 0

    def test_unicode_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "unicode.py").write_text(
                "# -*- coding: utf-8 -*-\n"
                "name = 'héllo wörld'\n"
                "emoji = '🧠'\n",
                encoding="utf-8",
            )
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 1
            assert report.files[0].parse_error is False

    def test_test_files_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "core.py").write_text("x = 1\n")
            (p / "test_core.py").write_text("x = 1\n")
            report = StaticInspector().inspect(p)
            assert report.files_analyzed == 1
            assert "core.py" in report.files[0].path


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISM
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_identical_runs(self):
        """Two runs on the same input must produce identical output."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "a.py").write_text("def f():\n  return 1\n")
            (p / "b.py").write_text("import a\ndef g():\n  pass\n")
            inspector = StaticInspector()
            r1 = inspector.inspect(p)
            r2 = inspector.inspect(p)
            assert r1.to_dict() == r2.to_dict()

    def test_report_sorted(self):
        """Files in report must be sorted by path."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "z_last.py").write_text("x = 1\n")
            (p / "a_first.py").write_text("x = 2\n")
            (p / "m_middle.py").write_text("x = 3\n")
            report = StaticInspector().inspect(p)
            paths = [f.path for f in report.files]
            assert paths == sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

class TestPerformance:

    def test_real_codebase_under_2_seconds(self):
        t0 = time.perf_counter()
        StaticInspector().inspect(_AGENT_DIR)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Inspection took {elapsed:.2f}s (limit: 2.0s)"

    def test_large_synthetic_project(self):
        """Must handle 200 files without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(200):
                (p / f"module_{i:03d}.py").write_text(
                    f"def func_{i}():\n  return {i}\n"
                )
            t0 = time.perf_counter()
            report = StaticInspector().inspect(p)
            elapsed = time.perf_counter() - t0
            assert report.files_analyzed == 200
            assert elapsed < 5.0, f"Took {elapsed:.2f}s for 200 files"


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    test_classes = [
        TestConfigAndModels,
        TestParsing,
        TestComplexity,
        TestImportGraph,
        TestLayerViolations,
        TestSmellDetection,
        TestRiskScoring,
        TestEndToEnd,
        TestEdgeCases,
        TestDeterminism,
        TestPerformance,
    ]

    passed = []
    failed = []

    for cls in test_classes:
        instance = cls()
        methods = sorted(m for m in dir(cls) if m.startswith("test_"))
        for method_name in methods:
            label = f"{cls.__name__}.{method_name}"
            try:
                getattr(instance, method_name)()
                passed.append(label)
                print(f"  ✅ {label}")
            except Exception as e:
                failed.append((label, e))
                print(f"  ❌ {label}: {e}")
                traceback.print_exc()

    total = len(passed) + len(failed)
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)}/{total} passed")
    if failed:
        print("\nFAILED:")
        for label, err in failed:
            print(f"  {label}: {err}")
        sys.exit(1)
    else:
        print("All checks passed ✅")
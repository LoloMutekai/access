"""
A.C.C.E.S.S. — Static Self-Inspection (Phase 6.1)

Pure static analysis engine for the A.C.C.E.S.S. cognitive architecture.

Design:
    - AST-based — never executes code, never imports analyzed modules
    - Deterministic — same input → same output, always
    - Bounded — handles 1000+ files, gracefully degrades on errors
    - Configurable — all thresholds exposed via InspectionConfig
    - JSON-serializable — every output converts to dict/JSON cleanly

Capabilities:
    1. Project structure analysis (file discovery, module boundaries)
    2. Import graph analysis (cycles, layer violations, fan-in/fan-out)
    3. Complexity metrics (cyclomatic, nesting, parameters, god-class)
    4. Architectural smell detection (IO density, state density, oversized)
    5. Stability risk scoring (sigmoid-normalized composite)

Integration:
    from agent.self_inspection import StaticInspector
    report = StaticInspector().inspect(Path("agent/"))
    print(report.health_grade)  # "B"
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InspectionConfig:
    """All thresholds for static inspection. Frozen — no runtime mutation."""

    # Structure thresholds
    max_lines_per_file: int = 500
    max_functions_per_file: int = 30
    max_classes_per_file: int = 10

    # Complexity thresholds
    max_cyclomatic_complexity: int = 10
    max_nesting_depth: int = 4
    max_parameters: int = 6
    max_function_lines: int = 50

    # God-class thresholds
    god_class_method_threshold: int = 15
    god_class_attribute_threshold: int = 10

    # Import / architecture
    max_fan_in: int = 8
    max_fan_out: int = 8

    # Smell thresholds
    io_density_threshold: float = 0.3
    state_density_threshold: float = 0.4

    # Risk scoring weights (must sum to ~1.0)
    risk_weight_complexity: float = 0.25
    risk_weight_coupling: float = 0.25
    risk_weight_size: float = 0.20
    risk_weight_smells: float = 0.15
    risk_weight_io: float = 0.15

    # Architectural layers (top → bottom)
    # Higher index = lower layer. Importing upward = violation.
    layer_order: tuple[str, ...] = (
        "agent_core",
        "cognitive_identity",
        "adaptive_meta",
        "meta_strategy",
        "meta_diagnostics",
        "reflection_engine",
        "self_model",
        "relationship_state",
        "personality_state",
        "goal_queue",
        "trajectory",
        "memory_loop",
        "logger",
        "persistence",
        "self_inspection",
        "models",
        "agent_config",
        "llm_client",
    )

    # File patterns to exclude
    exclude_patterns: tuple[str, ...] = ("test_", "__pycache__", ".pyc")


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FunctionMetrics:
    """Metrics for a single function or method."""
    name: str
    lines: int = 0
    cyclomatic_complexity: int = 1
    max_nesting: int = 0
    parameter_count: int = 0
    is_complex: bool = False
    is_deeply_nested: bool = False
    is_over_parameterized: bool = False
    is_long: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lines": self.lines,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting": self.max_nesting,
            "parameter_count": self.parameter_count,
            "is_complex": self.is_complex,
            "is_deeply_nested": self.is_deeply_nested,
            "is_over_parameterized": self.is_over_parameterized,
            "is_long": self.is_long,
        }


@dataclass(frozen=True)
class ClassMetrics:
    """Metrics for a single class."""
    name: str
    method_count: int = 0
    attribute_count: int = 0
    is_god_class: bool = False
    lines: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "method_count": self.method_count,
            "attribute_count": self.attribute_count,
            "is_god_class": self.is_god_class,
            "lines": self.lines,
        }


@dataclass(frozen=True)
class FileMetrics:
    """Metrics for a single Python file."""
    path: str
    lines_total: int = 0
    lines_code: int = 0
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    imports: tuple[str, ...] = field(default_factory=tuple)
    functions: tuple[FunctionMetrics, ...] = field(default_factory=tuple)
    classes: tuple[ClassMetrics, ...] = field(default_factory=tuple)
    smells: tuple[str, ...] = field(default_factory=tuple)
    io_density: float = 0.0
    state_density: float = 0.0
    parse_error: bool = False
    is_oversized: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "lines_total": self.lines_total,
            "lines_code": self.lines_code,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "imports": list(self.imports),
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "smells": list(self.smells),
            "io_density": round(self.io_density, 4),
            "state_density": round(self.state_density, 4),
            "parse_error": self.parse_error,
            "is_oversized": self.is_oversized,
        }


@dataclass(frozen=True)
class ImportEdge:
    """A directed import relationship."""
    source: str
    target: str
    is_relative: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "is_relative": self.is_relative,
        }


@dataclass(frozen=True)
class InspectionReport:
    """
    Complete inspection report. Immutable.

    All fields are JSON-serializable via to_dict().
    """
    files: tuple[FileMetrics, ...] = field(default_factory=tuple)
    import_graph: tuple[ImportEdge, ...] = field(default_factory=tuple)
    cycles: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    layer_violations: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    fan_in: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    fan_out: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    smells: tuple[str, ...] = field(default_factory=tuple)
    risk_scores: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    composite_risk: float = 0.0
    health_grade: str = "A"

    @property
    def files_analyzed(self) -> int:
        return len(self.files)

    @property
    def lines_analyzed(self) -> int:
        return sum(f.lines_total for f in self.files)

    @property
    def smells_detected(self) -> int:
        return len(self.smells)

    @property
    def cycles_detected(self) -> int:
        return len(self.cycles)

    def to_dict(self) -> dict:
        return {
            "files_analyzed": self.files_analyzed,
            "lines_analyzed": self.lines_analyzed,
            "smells_detected": self.smells_detected,
            "cycles_detected": self.cycles_detected,
            "health_grade": self.health_grade,
            "composite_risk": round(self.composite_risk, 4),
            "files": [f.to_dict() for f in self.files],
            "import_graph": [e.to_dict() for e in self.import_graph],
            "cycles": [list(c) for c in self.cycles],
            "layer_violations": [list(v) for v in self.layer_violations],
            "fan_in": [{"module": m, "count": c} for m, c in self.fan_in],
            "fan_out": [{"module": m, "count": c} for m, c in self.fan_out],
            "smells": list(self.smells),
            "risk_scores": [{"dimension": d, "score": round(s, 4)} for d, s in self.risk_scores],
        }

    def __repr__(self) -> str:
        return (
            f"InspectionReport("
            f"files={self.files_analyzed}, "
            f"lines={self.lines_analyzed}, "
            f"smells={self.smells_detected}, "
            f"cycles={self.cycles_detected}, "
            f"grade={self.health_grade})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PARSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def safe_parse(source: str) -> Optional[ast.Module]:
    """
    Parse Python source into AST. Returns None on syntax errors.
    Never raises.
    """
    if not source or not source.strip():
        return None
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None


def count_lines(source: str) -> int:
    """
    Count non-empty, non-comment-only lines.
    """
    if not source:
        return 0
    count = 0
    for line in source.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def cyclomatic_complexity(node: ast.AST) -> int:
    """
    Approximate cyclomatic complexity of a function/method AST node.

    Counts decision points:
    - if, elif → +1 each
    - for, while → +1 each
    - except → +1 each
    - BoolOp (and, or) → +1 per operator
    - IfExp (ternary) → +1
    - comprehension if-filters → +1 each
    - assert → +1

    Base complexity = 1.
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.IfExp)):
            complexity += 1
        elif isinstance(child, (ast.For, ast.While, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            # `a and b and c` = 2 operators = +2
            complexity += len(child.values) - 1
        elif isinstance(child, ast.Assert):
            complexity += 1
        elif isinstance(child, ast.comprehension):
            complexity += len(child.ifs)
    return complexity


def max_nesting_depth(node: ast.AST) -> int:
    """
    Compute maximum nesting depth within a function body.

    Nesting constructs: if, for, while, with, try, async for, async with.
    """
    if not hasattr(node, "body"):
        return 0

    def _walk_depth(stmts: list, current_depth: int) -> int:
        max_d = current_depth
        for stmt in stmts:
            nesting_types = (
                ast.If, ast.For, ast.While, ast.With, ast.Try,
                ast.AsyncFor, ast.AsyncWith,
            )
            # Python 3.11+ has TryStar
            try:
                nesting_types = nesting_types + (ast.TryStar,)
            except AttributeError:
                pass

            if isinstance(stmt, nesting_types):
                child_depth = current_depth + 1
                max_d = max(max_d, child_depth)
                # Recurse into all sub-blocks
                for attr in ("body", "orelse", "handlers", "finalbody"):
                    block = getattr(stmt, attr, None)
                    if block and isinstance(block, list):
                        sub_stmts = []
                        for item in block:
                            if isinstance(item, ast.AST) and hasattr(item, "body"):
                                sub_stmts.append(item)
                            elif isinstance(item, ast.AST):
                                sub_stmts.append(item)
                        max_d = max(max_d, _walk_depth(block, child_depth))
            else:
                # Check sub-blocks in non-nesting statements (e.g., FunctionDef inside)
                for attr in ("body", "orelse", "handlers", "finalbody"):
                    block = getattr(stmt, attr, None)
                    if block and isinstance(block, list):
                        max_d = max(max_d, _walk_depth(block, current_depth))
        return max_d

    body = getattr(node, "body", [])
    return _walk_depth(body, 0)


def count_parameters(func_node: ast.AST) -> int:
    """
    Count function parameters, excluding `self` and `cls`.
    Includes *args and **kwargs as 1 each.
    """
    if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return 0

    args = func_node.args
    count = 0

    # Positional args
    for arg in args.args:
        if arg.arg not in ("self", "cls"):
            count += 1

    # Positional-only args
    for arg in getattr(args, "posonlyargs", []):
        if arg.arg not in ("self", "cls"):
            count += 1

    # *args
    if args.vararg:
        count += 1

    # Keyword-only args
    count += len(args.kwonlyargs)

    # **kwargs
    if args.kwarg:
        count += 1

    return count


def count_self_attributes(class_node: ast.ClassDef) -> int:
    """
    Count unique self.xxx attribute assignments within a class.
    Scans __init__ and all methods for self.attr = ... patterns.
    """
    attrs = set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute):
            # Check if it's self.xxx in an assignment context
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                attrs.add(node.attr)
    return len(attrs)


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT GRAPH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def extract_imports(tree: ast.Module, module_name: str = "") -> list[tuple[str, bool]]:
    """
    Extract import targets from an AST.

    Returns list of (target_module, is_relative).
    Only extracts the top-level module name for relative imports.
    """
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name.split(".")[0], False))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: from .foo import bar
                if node.module:
                    imports.append((node.module.split(".")[0], True))
            else:
                if node.module:
                    imports.append((node.module.split(".")[0], False))
    return imports


def build_import_graph(
    file_metrics: list[FileMetrics],
    internal_modules: set[str],
) -> tuple[dict[str, set[str]], list[ImportEdge]]:
    """
    Build directed import graph. Only includes edges between internal modules.

    Returns:
        (adjacency_dict, edge_list)
    """
    graph: dict[str, set[str]] = defaultdict(set)
    edges: list[ImportEdge] = []

    for fm in file_metrics:
        source = Path(fm.path).stem
        for target in fm.imports:
            # Only include edges to known internal modules
            target_stem = target.split(".")[0]
            if target_stem in internal_modules and target_stem != source:
                graph[source].add(target_stem)
                edges.append(ImportEdge(source=source, target=target_stem))

    return dict(graph), edges


def detect_cycles(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    """
    Detect all cycles in a directed graph using DFS.

    Returns list of cycles, each as a tuple of module names.
    Cycles are sorted for determinism.
    """
    cycles: list[tuple[str, ...]] = []
    visited: set[str] = set()
    rec_stack: list[str] = []
    rec_set: set[str] = set()

    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)

    def _dfs(node: str) -> None:
        visited.add(node)
        rec_stack.append(node)
        rec_set.add(node)

        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in visited:
                _dfs(neighbor)
            elif neighbor in rec_set:
                # Found a cycle
                idx = rec_stack.index(neighbor)
                cycle = tuple(rec_stack[idx:])
                # Normalize: rotate so smallest element is first
                min_idx = cycle.index(min(cycle))
                normalized = cycle[min_idx:] + cycle[:min_idx]
                if normalized not in cycles:
                    cycles.append(normalized)

        rec_stack.pop()
        rec_set.discard(node)

    for node in sorted(all_nodes):
        if node not in visited:
            _dfs(node)

    return sorted(cycles)


def compute_fan_in_out(
    graph: dict[str, set[str]],
    internal_modules: set[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Compute fan-in (dependents) and fan-out (dependencies) for each module.
    """
    fan_in: dict[str, int] = defaultdict(int)
    fan_out: dict[str, int] = defaultdict(int)

    for source, targets in graph.items():
        fan_out[source] = len(targets)
        for t in targets:
            fan_in[t] += 1

    # Ensure all internal modules appear
    for m in internal_modules:
        fan_in.setdefault(m, 0)
        fan_out.setdefault(m, 0)

    return dict(fan_in), dict(fan_out)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER VIOLATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_layer_violations(
    graph: dict[str, set[str]],
    layer_order: tuple[str, ...],
) -> list[tuple[str, str]]:
    """
    Detect imports that go from a lower layer to a higher layer.

    layer_order is top-to-bottom: index 0 = highest layer.
    A module at index 5 importing from index 2 is a violation
    (lower layer importing from higher layer).

    Modules not in layer_order are ignored.
    """
    layer_index = {name: i for i, name in enumerate(layer_order)}
    violations = []

    for source, targets in sorted(graph.items()):
        if source not in layer_index:
            continue
        src_idx = layer_index[source]
        for target in sorted(targets):
            if target not in layer_index:
                continue
            tgt_idx = layer_index[target]
            # Lower layer (higher index) importing from higher layer (lower index)
            if src_idx > tgt_idx:
                violations.append((source, target))

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# SMELL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# IO-related AST patterns
_IO_NAMES = frozenset({
    "open", "read", "write", "close", "print",
    "input", "readline", "readlines", "writelines",
    "Path", "mkdir", "rmdir", "unlink", "rename",
    "exists", "is_file", "is_dir", "glob", "iterdir",
    "connect", "cursor", "execute", "commit", "fetchone", "fetchall",
    "get", "post", "put", "delete", "request", "urlopen",
    "load", "dump", "dumps", "loads",
    "send", "recv", "socket", "listen", "accept",
})

# State-related patterns
_STATE_NAMES = frozenset({
    "self", "global", "nonlocal", "setattr", "delattr",
    "__setattr__", "__delattr__", "__setitem__", "__delitem__",
    "append", "extend", "insert", "pop", "remove", "clear",
    "update", "setdefault",
})


def compute_io_density(tree: ast.Module) -> float:
    """
    Compute IO density: fraction of function calls that are IO-related.
    Returns 0.0 if no calls found.
    """
    total_calls = 0
    io_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            total_calls += 1
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name in _IO_NAMES:
                io_calls += 1

    if total_calls == 0:
        return 0.0
    return io_calls / total_calls


def compute_state_density(tree: ast.Module) -> float:
    """
    Compute state mutation density: fraction of assignments that are
    attribute/subscript mutations.
    """
    total_assigns = 0
    state_assigns = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            total_assigns += 1
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]

            for t in targets:
                if isinstance(t, ast.Attribute):
                    state_assigns += 1
                elif isinstance(t, ast.Subscript):
                    state_assigns += 1

    if total_assigns == 0:
        return 0.0
    return state_assigns / total_assigns


def detect_smells(
    file_metrics: list[FileMetrics],
    cycles: list[tuple[str, ...]],
    config: InspectionConfig,
) -> list[str]:
    """
    Detect architectural smells across the codebase.
    Returns a sorted list of unique smell descriptions.
    """
    smells = []

    for fm in file_metrics:
        module = Path(fm.path).stem

        # Oversized module
        if fm.is_oversized:
            smells.append(f"oversized_module:{module}({fm.lines_code} lines)")

        # High IO density
        if fm.io_density > config.io_density_threshold:
            smells.append(f"high_io_density:{module}({fm.io_density:.2f})")

        # High state density
        if fm.state_density > config.state_density_threshold:
            smells.append(f"high_state_density:{module}({fm.state_density:.2f})")

        # God classes
        for cls in fm.classes:
            if cls.is_god_class:
                smells.append(f"god_class:{module}.{cls.name}(methods={cls.method_count},attrs={cls.attribute_count})")

        # Complex functions
        for func in fm.functions:
            if func.is_complex:
                smells.append(f"high_complexity:{module}.{func.name}(cc={func.cyclomatic_complexity})")
            if func.is_deeply_nested:
                smells.append(f"deep_nesting:{module}.{func.name}(depth={func.max_nesting})")
            if func.is_over_parameterized:
                smells.append(f"over_parameterized:{module}.{func.name}(params={func.parameter_count})")

    # Circular dependencies
    for cycle in cycles:
        smells.append(f"circular_dependency:{' → '.join(cycle)}")

    return sorted(set(smells))


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────────────────────────────────────────

def sigmoid_normalize(value: float, midpoint: float = 5.0, k: float = 0.5) -> float:
    """
    Sigmoid normalization: maps any value to [0.0, 1.0].

    At value == midpoint → returns 0.5.
    k controls steepness (higher = sharper transition).
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * (value - midpoint)))
    except OverflowError:
        return 0.0 if value < midpoint else 1.0


def compute_risk_scores(
    file_metrics: list[FileMetrics],
    cycles: list[tuple[str, ...]],
    graph: dict[str, set[str]],
    config: InspectionConfig,
) -> tuple[list[tuple[str, float]], float]:
    """
    Compute per-dimension risk scores and a composite risk score.

    Returns:
        (dimension_scores, composite_risk)
        where dimension_scores is a list of (name, score) tuples,
        and composite_risk ∈ [0.0, 1.0].
    """
    if not file_metrics:
        return [], 0.0

    # ── Complexity risk ───────────────────────────────────────────────────
    max_cc = 0
    for fm in file_metrics:
        for func in fm.functions:
            max_cc = max(max_cc, func.cyclomatic_complexity)
    complexity_risk = sigmoid_normalize(max_cc, midpoint=config.max_cyclomatic_complexity)

    # ── Coupling risk ─────────────────────────────────────────────────────
    cycle_count = len(cycles)
    total_edges = sum(len(t) for t in graph.values())
    coupling_raw = cycle_count * 3 + total_edges * 0.5
    coupling_risk = sigmoid_normalize(coupling_raw, midpoint=15.0)

    # ── Size risk ─────────────────────────────────────────────────────────
    oversized = sum(1 for fm in file_metrics if fm.is_oversized)
    size_risk = sigmoid_normalize(oversized, midpoint=3.0)

    # ── Smell risk ────────────────────────────────────────────────────────
    total_smells = sum(len(fm.smells) for fm in file_metrics)
    smell_risk = sigmoid_normalize(total_smells, midpoint=5.0)

    # ── IO risk ───────────────────────────────────────────────────────────
    avg_io = (
        sum(fm.io_density for fm in file_metrics) / len(file_metrics)
        if file_metrics else 0.0
    )
    io_risk = sigmoid_normalize(avg_io * 10, midpoint=3.0)

    dimensions = [
        ("complexity", complexity_risk),
        ("coupling", coupling_risk),
        ("size", size_risk),
        ("smells", smell_risk),
        ("io", io_risk),
    ]

    # Composite (weighted sum)
    weights = [
        config.risk_weight_complexity,
        config.risk_weight_coupling,
        config.risk_weight_size,
        config.risk_weight_smells,
        config.risk_weight_io,
    ]
    composite = sum(w * s for w, (_, s) in zip(weights, dimensions))
    composite = max(0.0, min(1.0, composite))

    return dimensions, composite


def risk_to_grade(composite_risk: float) -> str:
    """Convert composite risk score to a letter grade."""
    if composite_risk < 0.2:
        return "A"
    elif composite_risk < 0.4:
        return "B"
    elif composite_risk < 0.6:
        return "C"
    elif composite_risk < 0.8:
        return "D"
    else:
        return "F"


# ─────────────────────────────────────────────────────────────────────────────
# FILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_function(
    func_node: ast.AST,
    config: InspectionConfig,
) -> FunctionMetrics:
    """Analyze a single function/method node."""
    if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        name = func_node.name
        # Line count
        if hasattr(func_node, "end_lineno") and func_node.end_lineno and func_node.lineno:
            lines = func_node.end_lineno - func_node.lineno + 1
        else:
            lines = len(func_node.body)
    else:
        name = "<unknown>"
        lines = 0

    cc = cyclomatic_complexity(func_node)
    nesting = max_nesting_depth(func_node)
    params = count_parameters(func_node)

    return FunctionMetrics(
        name=name,
        lines=lines,
        cyclomatic_complexity=cc,
        max_nesting=nesting,
        parameter_count=params,
        is_complex=cc > config.max_cyclomatic_complexity,
        is_deeply_nested=nesting > config.max_nesting_depth,
        is_over_parameterized=params > config.max_parameters,
        is_long=lines > config.max_function_lines,
    )


def _analyze_class(
    class_node: ast.ClassDef,
    config: InspectionConfig,
) -> ClassMetrics:
    """Analyze a single class node."""
    methods = [
        n for n in class_node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    attrs = count_self_attributes(class_node)

    if hasattr(class_node, "end_lineno") and class_node.end_lineno and class_node.lineno:
        lines = class_node.end_lineno - class_node.lineno + 1
    else:
        lines = len(class_node.body)

    return ClassMetrics(
        name=class_node.name,
        method_count=len(methods),
        attribute_count=attrs,
        is_god_class=(
            len(methods) > config.god_class_method_threshold
            and attrs > config.god_class_attribute_threshold
        ),
        lines=lines,
    )


def analyze_file(
    path: Path,
    config: InspectionConfig,
) -> FileMetrics:
    """
    Analyze a single Python file. Never raises.
    Returns FileMetrics with parse_error=True on failure.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError) as exc:
        logger.warning(f"Cannot read {path}: {exc}")
        return FileMetrics(path=str(path), parse_error=True)

    lines_total = len(source.splitlines())
    lines_code = count_lines(source)

    tree = safe_parse(source)
    if tree is None:
        return FileMetrics(
            path=str(path),
            lines_total=lines_total,
            lines_code=lines_code,
            parse_error=True if source.strip() else False,
        )

    # Extract imports
    raw_imports = extract_imports(tree, path.stem)
    import_names = tuple(sorted(set(name for name, _ in raw_imports)))

    # Analyze functions (top-level and inside classes)
    functions = []
    classes = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_analyze_function(node, config))
        elif isinstance(node, ast.ClassDef):
            cls_metrics = _analyze_class(node, config)
            classes.append(cls_metrics)
            # Also analyze methods inside classes
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(_analyze_function(child, config))

    # IO and state density
    io_density = compute_io_density(tree)
    state_density = compute_state_density(tree)

    # Per-file smells
    file_smells = []
    is_oversized = lines_code > config.max_lines_per_file
    if is_oversized:
        file_smells.append("oversized")
    if io_density > config.io_density_threshold:
        file_smells.append("high_io")
    if state_density > config.state_density_threshold:
        file_smells.append("high_state_mutation")
    for cls in classes:
        if cls.is_god_class:
            file_smells.append(f"god_class:{cls.name}")

    return FileMetrics(
        path=str(path),
        lines_total=lines_total,
        lines_code=lines_code,
        function_count=len(functions),
        class_count=len(classes),
        import_count=len(import_names),
        imports=import_names,
        functions=tuple(functions),
        classes=tuple(classes),
        smells=tuple(sorted(file_smells)),
        io_density=io_density,
        state_density=state_density,
        parse_error=False,
        is_oversized=is_oversized,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INSPECTOR
# ─────────────────────────────────────────────────────────────────────────────

class StaticInspector:
    """
    Main entry point for static self-inspection.

    Usage:
        inspector = StaticInspector()
        report = inspector.inspect(Path("agent/"))
        print(report.health_grade)
        summary = inspector.quick_summary(report)
    """

    def __init__(self, config: Optional[InspectionConfig] = None):
        self._config = config or InspectionConfig()

    @property
    def config(self) -> InspectionConfig:
        return self._config

    def inspect(self, path: Path) -> InspectionReport:
        """
        Run full static inspection on a directory.

        Returns InspectionReport — always succeeds, returns empty on error.
        Never raises.
        """
        # 1. Discover files
        py_files = self._discover_files(path)
        if not py_files:
            return InspectionReport()

        # 2. Analyze each file
        file_metrics = []
        for f in py_files:
            fm = analyze_file(f, self._config)
            file_metrics.append(fm)

        # Sort for determinism
        file_metrics.sort(key=lambda fm: fm.path)

        # 3. Determine internal modules
        internal_modules = {Path(fm.path).stem for fm in file_metrics}

        # 4. Build import graph
        graph, edges = build_import_graph(file_metrics, internal_modules)

        # 5. Detect cycles
        cycles = detect_cycles(graph)

        # 6. Detect layer violations
        violations = detect_layer_violations(graph, self._config.layer_order)

        # 7. Fan-in / fan-out
        fan_in_map, fan_out_map = compute_fan_in_out(graph, internal_modules)
        fan_in_sorted = sorted(fan_in_map.items(), key=lambda x: (-x[1], x[0]))
        fan_out_sorted = sorted(fan_out_map.items(), key=lambda x: (-x[1], x[0]))

        # 8. Detect smells
        smells = detect_smells(file_metrics, cycles, self._config)

        # 9. Compute risk scores
        risk_dims, composite = compute_risk_scores(
            file_metrics, cycles, graph, self._config
        )

        # 10. Grade
        grade = risk_to_grade(composite)

        return InspectionReport(
            files=tuple(file_metrics),
            import_graph=tuple(edges),
            cycles=tuple(cycles),
            layer_violations=tuple(violations),
            fan_in=tuple(fan_in_sorted),
            fan_out=tuple(fan_out_sorted),
            smells=tuple(smells),
            risk_scores=tuple(risk_dims),
            composite_risk=composite,
            health_grade=grade,
        )

    def quick_summary(self, report: InspectionReport) -> dict:
        """
        Return a lightweight summary dict for quick display.
        """
        return {
            "files": report.files_analyzed,
            "lines": report.lines_analyzed,
            "smells": report.smells_detected,
            "cycles": report.cycles_detected,
            "grade": report.health_grade,
            "risk": round(report.composite_risk, 4),
        }

    def _discover_files(self, path: Path) -> list[Path]:
        """
        Discover Python files in a directory.
        Excludes test files and __pycache__.
        Returns empty list on error.
        """
        try:
            if not path.exists() or not path.is_dir():
                logger.error(f"Path does not exist or is not a directory: {path}")
                return []

            files = []
            for root, dirs, filenames in os.walk(str(path)):
                # Prune __pycache__ dirs
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for fname in sorted(filenames):
                    if not fname.endswith(".py"):
                        continue
                    # Exclude test files and other patterns
                    skip = False
                    for pattern in self._config.exclude_patterns:
                        if pattern in fname or pattern in root:
                            skip = True
                            break
                    if skip:
                        continue
                    files.append(Path(root) / fname)
            return sorted(files)

        except OSError as exc:
            logger.error(f"File discovery error: {exc}")
            return []
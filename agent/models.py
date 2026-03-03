"""
A.C.C.E.S.S. — Agent Models (Phase 3)

Phase 3 additions:
    - UTC-aware datetimes: datetime.now(UTC) replaces datetime.utcnow()
    - AgentResponse.reflection: Optional[ReflectionResult]
    - AgentResponse.trajectory: Optional[TrajectoryState]
    - PipelineTrace: added reflection_ms stage
    - StreamContext: updated (no field changes — UTC-internal only)

Backward compatibility:
    - AgentResponse.tool_results, .sections_used remain tuples (frozen)
    - PipelineTrace fields unchanged (reflection_ms is additive)
    - TurnRecord.build() extended to accept reflection + trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TRACE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineTrace:
    """
    Per-stage timing breakdown.
    Phase 3 adds reflection_ms.
    All values in milliseconds.
    """
    emotion_ms: float = 0.0
    modulation_ms: float = 0.0
    rag_ms: float = 0.0
    prompt_ms: float = 0.0
    llm_ms: float = 0.0
    tool_dispatch_ms: float = 0.0
    reflection_ms: float = 0.0      # Phase 3: reflection computation time
    memory_write_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "emotion_ms":        round(self.emotion_ms, 2),
            "modulation_ms":     round(self.modulation_ms, 2),
            "rag_ms":            round(self.rag_ms, 2),
            "prompt_ms":         round(self.prompt_ms, 2),
            "llm_ms":            round(self.llm_ms, 2),
            "tool_dispatch_ms":  round(self.tool_dispatch_ms, 2),
            "reflection_ms":     round(self.reflection_ms, 2),
            "memory_write_ms":   round(self.memory_write_ms, 2),
            "total_ms":          round(self.total_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"PipelineTrace("
            f"emotion={self.emotion_ms:.0f}ms, "
            f"llm={self.llm_ms:.0f}ms, "
            f"tools={self.tool_dispatch_ms:.0f}ms, "
            f"reflection={self.reflection_ms:.0f}ms, "
            f"total={self.total_ms:.0f}ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentResponse:
    """
    Immutable output of AgentCore.handle_message() or finalize_stream().

    Phase 3 additions:
        - reflection: Optional[ReflectionResult] — heuristic self-reflection
        - trajectory: Optional[TrajectoryState]  — goal trajectory snapshot
        - detected_at: UTC-aware datetime
    """
    user_input: str
    assistant_output: str
    emotional_state: object
    modulation: object
    sections_used: tuple[str, ...]
    latency_ms: float
    session_id: Optional[str] = None
    turn_index: int = 0
    trace: Optional[PipelineTrace] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    tool_results: tuple = field(default_factory=tuple)
    reflection: Optional[object] = None      # ReflectionResult (duck-typed)
    trajectory: Optional[object] = None      # TrajectoryState (duck-typed)
    coherence_score: Optional[float] = None  # Phase 4.6: meta-cognitive coherence

    def to_log_dict(self) -> dict:
        emotion = self.emotional_state
        mod = self.modulation
        d = {
            "session_id":   self.session_id,
            "turn_index":   self.turn_index,
            "latency_ms":   round(self.latency_ms, 2),
            "emotion":      getattr(emotion, "primary_emotion", "?"),
            "intensity":    round(getattr(getattr(emotion, "pad", None) or object(), "valence", 0.0), 3),
            "tone":         getattr(mod, "tone", "?"),
            "pacing":       getattr(mod, "pacing", "?"),
            "sections":     list(self.sections_used),
            "output_chars": len(self.assistant_output),
            "tool_calls":   len(self.tool_results),
            "detected_at":  self.detected_at.isoformat(),
        }
        if self.reflection is not None:
            d["reflection"] = getattr(self.reflection, "to_log_dict", lambda: {})()
        if self.trajectory is not None:
            d["trajectory"] = getattr(self.trajectory, "to_dict", lambda: {})()
        if self.coherence_score is not None:
            d["coherence_score"] = round(self.coherence_score, 4)
        return d

    def __repr__(self) -> str:
        emotion = getattr(self.emotional_state, "label", "?")
        tone = getattr(self.modulation, "tone", "?")
        parts = [
            f"turn={self.turn_index}",
            f"emotion={emotion}",
            f"tone={tone}",
            f"latency={self.latency_ms:.0f}ms",
        ]
        if self.tool_results:
            parts.append(f"tools={len(self.tool_results)}")
        if self.reflection is not None:
            imp = getattr(self.reflection, "importance_score", "?")
            parts.append(f"importance={imp:.2f}" if isinstance(imp, float) else f"importance={imp}")
        return f"AgentResponse({', '.join(parts)}, output='{self.assistant_output[:60]}...')"


# ─────────────────────────────────────────────────────────────────────────────
# STREAM CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StreamContext:
    """
    Frozen context produced by stream_message() and consumed by finalize_stream().
    Carries all pre-LLM pipeline results so finalize_stream() can produce
    a complete AgentResponse without re-running the emotion/modulation/RAG stages.
    """
    user_input: str
    session_id: Optional[str]
    turn_index: int
    emotional_state: object
    modulation: object
    messages: tuple[dict, ...]
    sections_used: tuple[str, ...]
    t_emotion_ms: float
    t_modulation_ms: float
    t_rag_ms: float
    t_prompt_ms: float
    t_stream_start: float


# ─────────────────────────────────────────────────────────────────────────────
# TURN RECORD
# ─────────────────────────────────────────────────────────────────────────────

class TurnRecord:
    """
    Mutable accumulator built during handle_message() / finalize_stream().
    Frozen into AgentResponse via build().

    Phase 3 additions:
        - reflection: Optional[ReflectionResult]
        - trajectory: Optional[TrajectoryState]
        - _t_reflection: timing for reflection stage
    """

    def __init__(self, user_input: str, session_id: Optional[str], turn_index: int):
        self.user_input = user_input
        self.session_id = session_id
        self.turn_index = turn_index
        self.detected_at = datetime.now(UTC)    # Phase 3: UTC-aware

        self.emotional_state = None
        self.modulation = None
        self.built_prompt = None
        self.assistant_output: str = ""
        self.sections_used: tuple[str, ...] = ()
        self.tool_results: list = []
        self.reflection = None      # ReflectionResult
        self.trajectory = None      # TrajectoryState
        self.coherence_score: Optional[float] = None  # Phase 4.6

        # Stage timing (milliseconds)
        self._t_emotion: float = 0.0
        self._t_modulation: float = 0.0
        self._t_rag: float = 0.0
        self._t_prompt: float = 0.0
        self._t_llm: float = 0.0
        self._t_tool_dispatch: float = 0.0
        self._t_reflection: float = 0.0     # Phase 3
        self._t_memory: float = 0.0

    def build(self, total_ms: float) -> AgentResponse:
        trace = PipelineTrace(
            emotion_ms=self._t_emotion,
            modulation_ms=self._t_modulation,
            rag_ms=self._t_rag,
            prompt_ms=self._t_prompt,
            llm_ms=self._t_llm,
            tool_dispatch_ms=self._t_tool_dispatch,
            reflection_ms=self._t_reflection,
            memory_write_ms=self._t_memory,
            total_ms=total_ms,
        )
        return AgentResponse(
            user_input=self.user_input,
            assistant_output=self.assistant_output,
            emotional_state=self.emotional_state,
            modulation=self.modulation,
            sections_used=self.sections_used,
            latency_ms=total_ms,
            session_id=self.session_id,
            turn_index=self.turn_index,
            trace=trace,
            detected_at=self.detected_at,
            tool_results=tuple(self.tool_results),
            reflection=self.reflection,
            trajectory=self.trajectory,
            coherence_score=self.coherence_score,
        )
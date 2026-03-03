"""
A.C.C.E.S.S. — Agent Core (Phase 3)

Phase 3 additions:
    1. Self-reflection: ReflectionEngine called post-LLM (blocking) or post-finalize (streaming)
    2. Trajectory tracking: TrajectoryTracker updated after each reflection
    3. Adaptive memory: reflection.importance_score used when adaptive_importance=True
    4. MemoryLoop: run_memory_maintenance() triggers maintenance operations
    5. Structured logging: StructuredLogger accumulates JSON events
    6. Telemetry hook: callable injected at construction, called on key events
    7. UTC datetimes throughout

Architecture:
    AgentCore is ORCHESTRATION-ONLY.
    All intelligence lives in injected dependencies:
        EmotionEngine          → emotional state
        ConversationModulator  → style/tone adaptation
        PromptBuilder          → LLM message construction
        LLMClient              → language generation
        MemoryManager          → persistence
        ReflectionEngine       → post-turn cognitive reflection
        TrajectoryTracker      → goal drift tracking (stateful, lives on AgentCore)
        MemoryLoop             → maintenance operations
        StructuredLogger       → event logging

Pipeline (blocking — handle_message):
    Pre-LLM:  Emotion → Modulation → RAG → Prompt
    LLM:      LLMClient.chat()
    Post-LLM: Tool dispatch → [Reflection] → [Memory write] → [Emotional protection]
    Finalize: Trajectory update → Telemetry → Logging → AgentResponse

Pipeline (streaming):
    stream_message():  Pre-LLM → LLMClient.stream() → yield tokens
    finalize_stream(): Post-LLM stages → same as blocking post-LLM
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, Optional

from .agent_config import AgentConfig
from .llm_client import LLMClientProtocol, LLMError, LLMStreamError
from .logger import StructuredLogger
from .models import AgentResponse, StreamContext, TurnRecord
from .reflection_engine import ReflectionEngine, ReflectionEngineProtocol
from .trajectory import TrajectoryTracker

logger = logging.getLogger(__name__)


class AgentCore:
    """
    Orchestrates the full agent pipeline for a single conversational turn.

    Constructor args:
        emotion_engine          : Required. EmotionEngine or compatible.
        conversation_modulator  : Required. ConversationModulator or compatible.
        prompt_builder          : Required. PromptBuilder or compatible.
        llm_client              : Required. Implements LLMClientProtocol.
        memory_manager          : Optional. MemoryManager or compatible.
        config                  : Optional. AgentConfig with all flags.
        reflection_engine       : Optional. ReflectionEngine or compatible.
                                  If None and enable_reflection=True, a default is built.
        memory_loop             : Optional. MemoryLoop instance. Enables run_memory_maintenance().
        telemetry_hook          : Optional. Callable(event_name: str, metadata: dict) → None.
                                  Called on key events. Never crashes the agent on failure.

    Phase 3 usage:
        # After each turn
        response = agent.handle_message("I feel stuck.")
        print(response.reflection.goal_signal)   # "stabilize"
        print(response.trajectory.drift_score)   # 0.0 to 1.0

        # Manual memory maintenance
        report = agent.run_memory_maintenance()
        print(report.decay.memories_updated)

        # Retrieve structured logs
        logs = agent.get_logs()
        logs_by_type = agent.get_logs_by_type("reflection_done")
    """

    def __init__(
        self,
        emotion_engine,
        conversation_modulator,
        prompt_builder,
        llm_client: LLMClientProtocol,
        memory_manager=None,
        config: Optional[AgentConfig] = None,
        reflection_engine: Optional[ReflectionEngineProtocol] = None,
        memory_loop=None,
        telemetry_hook: Optional[Callable[[str, dict], None]] = None,
        identity_manager=None,
    ):
        self._emotion = emotion_engine
        self._modulator = conversation_modulator
        self._builder = prompt_builder
        self._llm = llm_client
        self._memory = memory_manager
        self.config = config or AgentConfig()

        # Tool dispatcher
        self._tool_dispatcher = self._build_tool_dispatcher()

        # Phase 3: Reflection
        self._reflection_engine = reflection_engine or (
            ReflectionEngine() if self.config.enable_reflection else None
        )

        # Phase 3: Trajectory
        self._trajectory = TrajectoryTracker(
            window_size=self.config.trajectory_window_size
        )

        # Phase 3: Memory loop
        self._memory_loop = memory_loop

        # Phase 3: Structured logging
        self._structured_logger: Optional[StructuredLogger] = (
            StructuredLogger() if self.config.enable_structured_logging else None
        )

        # Phase 3: Telemetry hook
        self._telemetry_hook = telemetry_hook

        # Phase 4: Cognitive Identity Manager (optional)
        self._identity = identity_manager

        # Session state
        self._turn_index: int = 0
        self._session_id: Optional[str] = self.config.default_session_id
        self._conversation_history: list[dict] = []
        self._stream_context: Optional[StreamContext] = None

        logger.info(
            f"AgentCore ready — llm={self._llm.model}, "
            f"memory={'ON' if memory_manager else 'OFF'}, "
            f"tools={'ON' if self._tool_dispatcher else 'OFF'}, "
            f"reflection={'ON' if self._reflection_engine else 'OFF'}, "
            f"trajectory=ON, "
            f"identity={'ON' if self._identity else 'OFF'}, "
            f"logging={'ON' if self._structured_logger else 'OFF'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # BLOCKING PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def handle_message(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Full blocking pipeline: user_input → AgentResponse."""
        t_total_start = time.perf_counter()
        sid = session_id or self._session_id

        turn = TurnRecord(user_input=user_input, session_id=sid, turn_index=self._turn_index)

        # Stages 1–4: pre-LLM
        pre = self._run_pre_llm_stages(user_input, sid)
        turn.emotional_state = pre["emotional_state"]
        turn.modulation = pre["modulation"]
        turn.sections_used = pre["sections_used"]
        turn._t_emotion = pre["t_emotion_ms"]
        turn._t_modulation = pre["t_modulation_ms"]
        turn._t_rag = pre["t_rag_ms"]
        turn._t_prompt = pre["t_prompt_ms"]
        messages = pre["messages"]

        # Stage 5: LLM
        t0 = time.perf_counter()
        try:
            raw_output = self._llm.chat(messages)
        except LLMError as exc:
            logger.error(f"[Stage 5 — LLM] LLMError: {exc}")
            raw_output = self._fallback_response(exc)
        except Exception as exc:
            logger.error(f"[Stage 5 — LLM] Unexpected: {exc}", exc_info=True)
            raw_output = self._fallback_response(exc)
        turn._t_llm = (time.perf_counter() - t0) * 1000

        # Stage 6: Tool dispatch
        t0 = time.perf_counter()
        if self.config.enable_tool_use and self._tool_dispatcher is not None:
            try:
                dispatch_result = self._tool_dispatcher.dispatch(
                    initial_response=raw_output,
                    messages=messages,
                    llm_call=self._llm.chat,
                )
                raw_output = dispatch_result.final_response
                turn.tool_results = list(dispatch_result.tool_results)

                # Telemetry for each tool used
                for tr in dispatch_result.tool_results:
                    self._fire_telemetry("tool_used", {
                        "tool_name": tr.tool_name,
                        "success": tr.success,
                        "latency_ms": tr.latency_ms,
                    })
                    self._log_event("tool_used", tr.to_log_dict(), sid, turn.turn_index)

            except Exception as exc:
                logger.error(f"[Stage 6 — Tool] Unexpected: {exc}", exc_info=True)
        turn._t_tool_dispatch = (time.perf_counter() - t0) * 1000

        # Post-process
        raw_output = self._post_process(raw_output)
        turn.assistant_output = raw_output
        self._update_history(user_input, raw_output)

        # Stage 7: Reflection
        t0 = time.perf_counter()
        reflection = self._run_reflection(
            user_input, raw_output, turn.emotional_state, turn.modulation,
            tuple(turn.tool_results), sid,
        )
        turn.reflection = reflection
        turn._t_reflection = (time.perf_counter() - t0) * 1000

        # Stage 8: Trajectory update
        if reflection is not None:
            trajectory_state = self._trajectory.update(
                goal_signal=reflection.goal_signal,
                trajectory_signal=reflection.trajectory_signal,
            )
            turn.trajectory = trajectory_state
        else:
            turn.trajectory = self._trajectory.state

        # Stage 9: Memory write-back (with adaptive importance)
        t0 = time.perf_counter()
        if self._memory is not None:
            self._write_to_memory(turn)
        turn._t_memory = (time.perf_counter() - t0) * 1000

        # Stage 10: Emotional protection
        if self.config.apply_emotional_protection and turn.emotional_state is not None:
            try:
                self._emotion.apply_emotional_protection(turn.emotional_state)
            except Exception as exc:
                logger.warning(f"[Stage 10 — Protection] Non-critical: {exc}")

        # Stage 11: Cognitive Identity update (Phase 4)
        if self._identity is not None and reflection is not None:
            try:
                self._identity.update_from_turn(
                    reflection, turn.trajectory, turn.emotional_state, turn.modulation,
                )
                coherence = getattr(self._identity, 'current_coherence', lambda: None)()
                if coherence is not None:
                    turn.coherence_score = coherence
            except Exception as exc:
                logger.warning(f"[Stage 11 — Identity] Non-critical: {exc}")

        total_ms = (time.perf_counter() - t_total_start) * 1000
        self._turn_index += 1
        response = turn.build(total_ms=total_ms)

        # Logging + telemetry
        self._log_event("turn_completed", response.to_log_dict(), sid, response.turn_index)
        if reflection is not None:
            self._log_event("reflection_done", reflection.to_log_dict(), sid, response.turn_index)
            self._fire_telemetry("reflection_done", reflection.to_log_dict())
        self._fire_telemetry("turn_completed", response.to_log_dict())

        logger.info(
            f"Turn {response.turn_index} — "
            f"{response.latency_ms:.0f}ms | "
            f"emotion={getattr(response.emotional_state, 'primary_emotion', '?')} | "
            f"tone={getattr(response.modulation, 'tone', '?')} | "
            f"importance={getattr(reflection, 'importance_score', '?')}"
        )
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def stream_message(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming pipeline: runs pre-LLM stages, then yields LLM tokens.
        Call finalize_stream() after consuming all tokens.
        """
        if self._stream_context is not None:
            logger.warning("stream_message() called while a stream is already active.")

        sid = session_id or self._session_id
        pre = self._run_pre_llm_stages(user_input, sid)
        messages = pre["messages"]

        self._stream_context = StreamContext(
            user_input=user_input,
            session_id=sid,
            turn_index=self._turn_index,
            emotional_state=pre["emotional_state"],
            modulation=pre["modulation"],
            messages=tuple(messages),
            sections_used=pre["sections_used"],
            t_emotion_ms=pre["t_emotion_ms"],
            t_modulation_ms=pre["t_modulation_ms"],
            t_rag_ms=pre["t_rag_ms"],
            t_prompt_ms=pre["t_prompt_ms"],
            t_stream_start=time.perf_counter(),
        )

        try:
            yield from self._llm.stream(messages)
        except LLMStreamError as exc:
            logger.error(f"[Stream — LLM] LLMStreamError: {exc}")
            yield self._fallback_response(exc)
        except Exception as exc:
            logger.error(f"[Stream — LLM] Unexpected: {exc}", exc_info=True)
            yield self._fallback_response(exc)

    def finalize_stream(self, full_response: str) -> AgentResponse:
        """
        Complete a streaming turn. Must be called after consuming all tokens.
        Runs reflection, trajectory, memory write-back, emotional protection.
        """
        if self._stream_context is None:
            raise RuntimeError(
                "finalize_stream() called without a preceding stream_message()."
            )

        ctx = self._stream_context
        self._stream_context = None

        t_total_start = ctx.t_stream_start
        t_llm_ms = (time.perf_counter() - t_total_start) * 1000

        processed = self._post_process(full_response)
        self._update_history(ctx.user_input, processed)

        turn = TurnRecord(
            user_input=ctx.user_input,
            session_id=ctx.session_id,
            turn_index=ctx.turn_index,
        )
        turn.emotional_state = ctx.emotional_state
        turn.modulation = ctx.modulation
        turn.sections_used = ctx.sections_used
        turn.assistant_output = processed
        turn._t_emotion = ctx.t_emotion_ms
        turn._t_modulation = ctx.t_modulation_ms
        turn._t_rag = ctx.t_rag_ms
        turn._t_prompt = ctx.t_prompt_ms
        turn._t_llm = t_llm_ms

        # Reflection
        t0 = time.perf_counter()
        reflection = self._run_reflection(
            ctx.user_input, processed, ctx.emotional_state, ctx.modulation, (), ctx.session_id
        )
        turn.reflection = reflection
        turn._t_reflection = (time.perf_counter() - t0) * 1000

        # Trajectory
        if reflection is not None:
            turn.trajectory = self._trajectory.update(
                goal_signal=reflection.goal_signal,
                trajectory_signal=reflection.trajectory_signal,
            )
        else:
            turn.trajectory = self._trajectory.state

        # Memory write-back
        t0 = time.perf_counter()
        if self._memory is not None:
            self._write_to_memory(turn)
        turn._t_memory = (time.perf_counter() - t0) * 1000

        # Emotional protection
        if self.config.apply_emotional_protection and ctx.emotional_state is not None:
            try:
                self._emotion.apply_emotional_protection(ctx.emotional_state)
            except Exception as exc:
                logger.warning(f"[finalize_stream — Protection] Non-critical: {exc}")

        # Cognitive Identity update (Phase 4)
        if self._identity is not None and reflection is not None:
            try:
                self._identity.update_from_turn(
                    reflection, turn.trajectory, ctx.emotional_state, ctx.modulation,
                )
                coherence = getattr(self._identity, 'current_coherence', lambda: None)()
                if coherence is not None:
                    turn.coherence_score = coherence
            except Exception as exc:
                logger.warning(f"[finalize_stream — Identity] Non-critical: {exc}")

        total_ms = (time.perf_counter() - t_total_start) * 1000
        self._turn_index += 1
        response = turn.build(total_ms=total_ms)

        # Logging + telemetry
        self._log_event("stream_finalized", response.to_log_dict(), ctx.session_id, response.turn_index)
        if reflection is not None:
            self._log_event("reflection_done", reflection.to_log_dict(), ctx.session_id, response.turn_index)
            self._fire_telemetry("reflection_done", reflection.to_log_dict())
        self._fire_telemetry("turn_completed", response.to_log_dict())

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # SHARED PRE-LLM STAGES
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pre_llm_stages(self, user_input: str, session_id: Optional[str]) -> dict:
        """
        Stages 1–4 shared between handle_message() and stream_message().
        Returns dict with all outputs + per-stage timing.
        """
        result: dict = {}

        # Stage 1: Emotion
        t0 = time.perf_counter()
        try:
            result["emotional_state"] = self._emotion.process_interaction(
                user_input, session_id=session_id
            )
        except Exception as exc:
            logger.error(f"[Stage 1 — Emotion] Failed: {exc}", exc_info=True)
            result["emotional_state"] = None
        result["t_emotion_ms"] = (time.perf_counter() - t0) * 1000

        if self.config.log_emotional_state and result["emotional_state"] is not None:
            logger.info(f"Emotional state: {result['emotional_state']}")

        # Stage 2: Modulation
        t0 = time.perf_counter()
        try:
            state = result["emotional_state"]
            trend = self._emotion.emotional_trend() if state else {}
            pattern = self._emotion.dominant_pattern() if state else None
            result["modulation"] = self._modulator.build_modulation(
                state=state, trend=trend, dominant_pattern=pattern,
            )
        except Exception as exc:
            logger.error(f"[Stage 2 — Modulation] Failed: {exc}", exc_info=True)
            result["modulation"] = None
        result["t_modulation_ms"] = (time.perf_counter() - t0) * 1000

        if self.config.log_modulation and result["modulation"] is not None:
            logger.info(f"Modulation: {result['modulation']}")

        # Stage 3: RAG
        t0 = time.perf_counter()
        memory_context: Optional[str] = None
        if self.config.enable_rag and self._memory is not None:
            try:
                emotional_ctx = (
                    result["emotional_state"] if self.config.rag_emotion_aware else None
                )
                memories = self._memory.retrieve_relevant_memories(
                    query=user_input,
                    top_k=self.config.rag_top_k,
                    min_importance=self.config.rag_min_importance,
                    emotional_context=emotional_ctx,
                )
                if memories:
                    memory_context = self._memory.format_for_rag(memories)
            except Exception as exc:
                logger.error(f"[Stage 3 — RAG] Failed: {exc}", exc_info=True)
        result["t_rag_ms"] = (time.perf_counter() - t0) * 1000

        # Stage 4: Prompt
        t0 = time.perf_counter()
        sections_used: tuple[str, ...] = ()
        try:
            if self.config.enable_tool_use and self._tool_dispatcher is not None:
                tool_section = self._get_tool_prompt_section()
                if tool_section:
                    memory_context = (
                        (memory_context + "\n\n" if memory_context else "") + tool_section
                    )

            built = self._builder.build(
                user_input=user_input,
                modulation=result["modulation"],
                memory_context=memory_context,
            )
            sections_used = built.sections
            messages = self._inject_history(built.to_api_messages())

            if self.config.log_full_prompt:
                logger.debug(f"Prompt messages: {messages}")

        except Exception as exc:
            logger.error(f"[Stage 4 — Prompt] Failed: {exc}", exc_info=True)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ]
        result["t_prompt_ms"] = (time.perf_counter() - t0) * 1000
        result["messages"] = messages
        result["sections_used"] = sections_used

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # REFLECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _run_reflection(
        self,
        user_input: str,
        assistant_output: str,
        emotional_state,
        modulation,
        tool_results: tuple,
        session_id: Optional[str],
    ):
        """Run reflection if enabled. Never raises."""
        if not self.config.enable_reflection or self._reflection_engine is None:
            return None
        try:
            return self._reflection_engine.reflect(
                user_input=user_input,
                assistant_output=assistant_output,
                emotional_state=emotional_state,
                modulation=modulation,
                tool_results=tool_results,
                session_id=session_id,
            )
        except Exception as exc:
            logger.error(f"[Reflection] Failed: {exc}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY MAINTENANCE
    # ─────────────────────────────────────────────────────────────────────────

    def run_memory_maintenance(
        self,
        run_decay: bool = True,
        run_consolidation: bool = True,
        run_topics: bool = True,
        run_repetition: bool = True,
    ):
        """
        Manually trigger adaptive memory maintenance.

        If memory_loop is None, returns None.
        If memory_decay_enabled=False, run_decay is forced to False.
        """
        if self._memory_loop is None:
            logger.info("run_memory_maintenance() called but memory_loop is None")
            return None

        if not self.config.memory_decay_enabled:
            run_decay = False

        try:
            report = self._memory_loop.run(
                run_decay=run_decay,
                run_consolidation=run_consolidation,
                run_topics=run_topics,
                run_repetition=run_repetition,
            )
            self._log_event("memory_maintenance", report.to_dict(), self._session_id)
            self._fire_telemetry("memory_maintenance_done", report.to_dict())
            return report
        except Exception as exc:
            logger.error(f"[MemoryMaintenance] Failed: {exc}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    def get_logs(self) -> list[dict]:
        """Return all structured log events. Empty list if logging disabled."""
        if self._structured_logger is None:
            return []
        return self._structured_logger.get_logs()

    def get_logs_by_type(self, event_type: str) -> list[dict]:
        if self._structured_logger is None:
            return []
        return self._structured_logger.get_logs_by_type(event_type)

    def _log_event(
        self,
        event_type: str,
        payload: dict,
        session_id: Optional[str] = None,
        turn_index: Optional[int] = None,
    ) -> None:
        if self._structured_logger is None:
            return
        try:
            self._structured_logger.log_event(
                event_type=event_type,
                payload=payload,
                session_id=session_id,
                turn_index=turn_index,
            )
        except Exception as exc:
            logger.warning(f"Structured log error (suppressed): {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # TELEMETRY
    # ─────────────────────────────────────────────────────────────────────────

    def _fire_telemetry(self, event_name: str, metadata: dict) -> None:
        """Call telemetry hook. Suppresses all exceptions."""
        if self._telemetry_hook is None:
            return
        try:
            self._telemetry_hook(event_name, metadata)
        except Exception as exc:
            logger.warning(f"Telemetry hook error (suppressed): {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_tool_dispatcher(self):
        if not self.config.enable_tool_use:
            return None
        registry = self.config.tool_registry
        if registry is None:
            return None
        try:
            from tools.tool_dispatcher import ToolDispatcher
            return ToolDispatcher(
                registry=registry,
                max_iterations=self.config.max_tool_iterations,
            )
        except ImportError:
            logger.warning("tools module not found — tool use disabled")
            return None

    def _get_tool_prompt_section(self) -> str:
        registry = self.config.tool_registry
        if registry is None or not hasattr(registry, "get_prompt_section"):
            return ""
        try:
            return registry.get_prompt_section()
        except Exception:
            return ""

    def _post_process(self, response: str) -> str:
        if self.config.strip_response:
            response = response.strip()
        if self.config.max_response_chars > 0:
            response = response[:self.config.max_response_chars]
        return response

    def _write_to_memory(self, turn: TurnRecord) -> None:
        try:
            # Determine importance score
            if self.config.adaptive_importance and turn.reflection is not None:
                importance = getattr(turn.reflection, "importance_score",
                                     self.config.auto_memory_importance)
            else:
                importance = self.config.auto_memory_importance

            # Determine content / summary
            if self.config.use_reflection_summary_for_memory and turn.reflection is not None:
                content = getattr(turn.reflection, "summary", turn.user_input)
                summary = content[:120]
            else:
                content = turn.user_input
                summary = turn.user_input[:120]

            # Determine tags
            tags = self._emotion_tags(turn.emotional_state)
            if turn.reflection is not None:
                tags = tags + getattr(turn.reflection, "emotional_tags", [])
                tags = list(dict.fromkeys(tags))  # deduplicate, preserve order

            if self.config.write_user_turn_to_memory:
                self._memory.add_memory(
                    content=content,
                    summary=summary,
                    memory_type="episodic",
                    tags=tags,
                    importance_score=importance,
                    source=self.config.auto_memory_source,
                    session_id=turn.session_id,
                )
                self._log_event("memory_write", {
                    "content_chars": len(content),
                    "importance": round(importance, 3),
                    "tags": tags,
                }, turn.session_id, turn.turn_index)

            if self.config.write_assistant_turn_to_memory and turn.assistant_output:
                self._memory.add_memory(
                    content=turn.assistant_output,
                    summary=turn.assistant_output[:120],
                    memory_type="episodic",
                    tags=["assistant_response"],
                    importance_score=importance * 0.8,
                    source="agent_response",
                    session_id=turn.session_id,
                )
        except Exception as exc:
            logger.error(f"[Memory write-back] Failed: {exc}", exc_info=True)

    def _emotion_tags(self, state) -> list[str]:
        if state is None:
            return []
        emotion = getattr(state, "primary_emotion", None)
        return [emotion] if emotion else []

    def _inject_history(self, messages: list[dict]) -> list[dict]:
        if not self.config.enable_conversation_history or not self._conversation_history:
            return messages
        system = [m for m in messages if m.get("role") == "system"]
        user = [m for m in messages if m.get("role") != "system"]
        max_turns = self.config.conversation_history_max_turns
        history = self._conversation_history[-max_turns * 2:]
        return system + history + user

    def _update_history(self, user_input: str, assistant_output: str) -> None:
        if not self.config.enable_conversation_history:
            return
        self._conversation_history.append({"role": "user", "content": user_input})
        self._conversation_history.append({"role": "assistant", "content": assistant_output})
        max_msgs = self.config.conversation_history_max_turns * 2
        if len(self._conversation_history) > max_msgs:
            self._conversation_history = self._conversation_history[-max_msgs:]

    def _fallback_response(self, error: Exception) -> str:
        return "I'm having trouble connecting right now. Please try again in a moment."

    # ─────────────────────────────────────────────────────────────────────────
    # PROPERTIES / SESSION
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def is_streaming(self) -> bool:
        return self._stream_context is not None

    @property
    def trajectory(self):
        return self._trajectory.state

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id
        logger.info(f"Session ID set to: {session_id}")

    def reset_session(self) -> None:
        self._turn_index = 0
        self._conversation_history = []
        self._stream_context = None
        self._trajectory.reset()
        logger.info(f"Session reset — session_id={self._session_id}")

    def stats(self) -> dict:
        emotion_stats = {}
        try:
            emotion_stats = self._emotion.stats()
        except Exception:
            pass
        return {
            "session_id":      self._session_id,
            "turn_index":      self._turn_index,
            "llm_model":       self._llm.model,
            "memory":          self._memory is not None,
            "rag_enabled":     self.config.enable_rag,
            "tools_enabled":   self._tool_dispatcher is not None,
            "reflection":      self._reflection_engine is not None,
            "memory_loop":     self._memory_loop is not None,
            "history_len":     len(self._conversation_history),
            "is_streaming":    self.is_streaming,
            "identity":        self._identity is not None,
            "log_events":      self._structured_logger.event_count if self._structured_logger else 0,
            "trajectory":      self._trajectory.state.to_dict(),
            "emotion_stats":   emotion_stats,
        }
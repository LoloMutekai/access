"""
A.C.C.E.S.S. — LLM Client (Phase 2)

Thin, testable abstraction over the LLM API.

Phase 2 additions vs Phase 1:
- LLMClientProtocol now requires stream() — all clients must implement it
- FakeLLMClient.stream() yields fake tokens deterministically
- Stream error simulation supported in FakeLLMClient

Usage (blocking):
    client = AnthropicClient(api_key="...", model="claude-sonnet-4-6")
    response = client.chat(messages)

Usage (streaming):
    for token in client.stream(messages):
        print(token, end="", flush=True)

Testing:
    client = FakeLLMClient(response="Hello world.")
    tokens = list(client.stream(messages))
    # → ["Hello", " world."]  (default word splitter)

    client = FakeLLMClient(raise_error=LLMError("down"))          # blocking error
    client = FakeLLMClient(stream_raise_error=LLMStreamError("x")) # streaming error
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for all LLM client errors."""

class LLMTimeoutError(LLMError):
    """LLM API call exceeded configured timeout."""

class LLMAuthError(LLMError):
    """Authentication failure — invalid API key."""

class LLMRateLimitError(LLMError):
    """API rate limit exceeded."""

class LLMStreamError(LLMError):
    """Streaming response failed mid-stream."""


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    Full contract for any LLM client — blocking AND streaming.

    Both methods must be implemented.
    Clients that don't support native streaming may implement stream()
    by calling chat() and yielding the full result as a single chunk.
    """

    def chat(self, messages: list[dict]) -> str:
        """Blocking: returns the full response string."""
        ...

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Streaming: yields response tokens progressively."""
        ...

    @property
    def model(self) -> str:
        """The model identifier string."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Production LLM client using the Anthropic API.

    Requires: pip install anthropic

    Separates system message from conversation messages automatically,
    as required by the Anthropic API format.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key, timeout=self._timeout)
            except ImportError:
                raise LLMError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def _split_messages(self, messages: list[dict]) -> tuple[str, list[dict]]:
        system = ""
        conversation = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                conversation.append({"role": msg["role"], "content": msg["content"]})
        return system, conversation

    def chat(self, messages: list[dict]) -> str:
        system, conversation = self._split_messages(messages)
        if not conversation:
            raise LLMError("No user/assistant messages provided.")
        client = self._get_client()
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": conversation}
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        try:
            response = client.messages.create(**kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            text = response.content[0].text
            logger.info(f"AnthropicClient: {self._model} responded in {elapsed:.0f}ms ({len(text)} chars)")
            return text
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            logger.error(f"AnthropicClient error after {elapsed:.0f}ms: {exc}")
            msg = str(exc).lower()
            if "auth" in msg or "api_key" in msg or "unauthorized" in msg:
                raise LLMAuthError(str(exc)) from exc
            if "rate" in msg or "429" in msg:
                raise LLMRateLimitError(str(exc)) from exc
            if "timeout" in msg or "timed out" in msg:
                raise LLMTimeoutError(str(exc)) from exc
            raise LLMError(str(exc)) from exc

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield text tokens progressively from the Anthropic streaming API."""
        system, conversation = self._split_messages(messages)
        if not conversation:
            raise LLMError("No user/assistant messages provided.")
        client = self._get_client()
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": conversation}
        if system:
            kwargs["system"] = system
        try:
            with client.messages.stream(**kwargs) as stream_ctx:
                for text in stream_ctx.text_stream:
                    yield text
        except Exception as exc:
            raise LLMStreamError(str(exc)) from exc

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self._model})"


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient:
    """
    Alternative LLM client using the OpenAI API.

    Requires: pip install openai
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key, timeout=self._timeout)
            except ImportError:
                raise LLMError("openai package not installed. Run: pip install openai")
        return self._client

    def chat(self, messages: list[dict]) -> str:
        client = self._get_client()
        t0 = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=self._model, max_tokens=self._max_tokens, messages=messages,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            text = response.choices[0].message.content
            logger.info(f"OpenAIClient: {self._model} responded in {elapsed:.0f}ms")
            return text
        except Exception as exc:
            raise LLMError(str(exc)) from exc

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield text tokens from the OpenAI streaming API."""
        client = self._get_client()
        try:
            stream = client.chat.completions.create(
                model=self._model, max_tokens=self._max_tokens,
                messages=messages, stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    yield delta.content
        except Exception as exc:
            raise LLMStreamError(str(exc)) from exc

    @property
    def model(self) -> str:
        return self._model

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self._model})"


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

def _default_token_splitter(text: str) -> list[str]:
    """
    Default token splitter for FakeLLMClient.

    Splits on spaces, keeping the leading space with the following word.
    "Hello world!" → ["Hello", " world", "!"]
    
    Produces realistic word-level streaming — the same granularity as
    most production LLM streaming APIs.
    """
    if not text:
        return []
    tokens: list[str] = []
    current = ""
    for char in text:
        if char == " " and current:
            tokens.append(current)
            current = " "
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# FAKE CLIENT — deterministic, zero API calls, for testing
# ─────────────────────────────────────────────────────────────────────────────

class FakeLLMClient:
    """
    Deterministic fake LLM client for unit testing.

    Phase 2 additions:
    - stream() — yields fake tokens from response string
    - token_splitter — configurable: split words, chars, sentences, etc.
    - stream_token_latency_ms — per-token delay for latency tests
    - stream_raise_error — simulate streaming failure
    - stream_call_count, stream_call_history — call recording for assertions

    Usage:
        # Blocking (unchanged from Phase 1)
        client = FakeLLMClient(response="Hello.")
        text = client.chat(messages)

        # Streaming
        client = FakeLLMClient(response="Hello world!")
        tokens = list(client.stream(messages))
        # → ["Hello", " world", "!"]

        # Response function (dynamic based on messages)
        client = FakeLLMClient(response_fn=lambda msgs: f"Echo: {msgs[-1]['content']}")

        # Error simulation
        client = FakeLLMClient(raise_error=LLMError("down"))
        client = FakeLLMClient(stream_raise_error=LLMStreamError("cut"))

        # Custom token splitter (split into chars for granular tests)
        client = FakeLLMClient(response="Hi!", token_splitter=list)
        tokens = list(client.stream(messages))  # → ["H", "i", "!"]

    Call recording:
        assert client.call_count == 2
        assert client.stream_call_count == 1
        assert "tool_call" in client.last_system_prompt
    """

    def __init__(
        self,
        response: str = "Fake LLM response.",
        response_fn: Optional[Callable[[list[dict]], str]] = None,
        raise_error: Optional[Exception] = None,
        latency_ms: float = 0.0,
        model_name: str = "fake-model",
        # Streaming
        token_splitter: Optional[Callable[[str], list[str]]] = None,
        stream_token_latency_ms: float = 0.0,
        stream_raise_error: Optional[Exception] = None,
    ):
        self._response = response
        self._response_fn = response_fn
        self._raise_error = raise_error
        self._latency_ms = latency_ms
        self._model_name = model_name
        self._token_splitter: Callable[[str], list[str]] = token_splitter or _default_token_splitter
        self._stream_token_latency_ms = stream_token_latency_ms
        self._stream_raise_error = stream_raise_error

        # Blocking call recording
        self.call_count: int = 0
        self.call_history: list[list[dict]] = []
        # Streaming call recording
        self.stream_call_count: int = 0
        self.stream_call_history: list[list[dict]] = []

    def _resolve_response(self, messages: list[dict]) -> str:
        if self._response_fn is not None:
            return self._response_fn(messages)
        return self._response

    def chat(self, messages: list[dict]) -> str:
        self.call_count += 1
        self.call_history.append(list(messages))
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        if self._raise_error is not None:
            raise self._raise_error
        return self._resolve_response(messages)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """
        Yield fake tokens from the response string.
        Raises stream_raise_error immediately if configured.
        """
        self.stream_call_count += 1
        self.stream_call_history.append(list(messages))
        if self._stream_raise_error is not None:
            raise self._stream_raise_error
        response = self._resolve_response(messages)
        for token in self._token_splitter(response):
            if self._stream_token_latency_ms > 0:
                time.sleep(self._stream_token_latency_ms / 1000.0)
            yield token

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def last_messages(self) -> Optional[list[dict]]:
        return self.call_history[-1] if self.call_history else None

    @property
    def last_stream_messages(self) -> Optional[list[dict]]:
        return self.stream_call_history[-1] if self.stream_call_history else None

    @property
    def last_system_prompt(self) -> Optional[str]:
        msgs = self.last_messages
        if not msgs:
            return None
        for msg in msgs:
            if msg.get("role") == "system":
                return msg.get("content", "")
        return None

    def reset(self) -> None:
        self.call_count = 0
        self.call_history = []
        self.stream_call_count = 0
        self.stream_call_history = []

    def __repr__(self) -> str:
        return (
            f"FakeLLMClient("
            f"calls={self.call_count}, "
            f"streams={self.stream_call_count}, "
            f"model={self._model_name})"
        )
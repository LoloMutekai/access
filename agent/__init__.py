"""
A.C.C.E.S.S. — Agent Layer

Orchestrates the full pipeline:
    EmotionEngine → ConversationModulator → PromptBuilder → LLMClient → AgentResponse

Quick start:
    from agent import AgentCore, AnthropicClient, AgentConfig
    from agent.llm_client import FakeLLMClient  # for testing

    agent = AgentCore(
        emotion_engine=engine,
        conversation_modulator=modulator,
        prompt_builder=builder,
        llm_client=AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"]),
        memory_manager=memory,
        config=AgentConfig(),
    )

    response = agent.handle_message("I feel stuck on this problem.")
    print(response.assistant_output)
    print(response.trace.to_dict())    # timing breakdown
    print(response.to_log_dict())      # structured log
"""

from .agent_core import AgentCore
from .agent_config import AgentConfig
from .structural_meta import StructuralMetaState, StructuralMetaTracker, StructuralGate
from .patch_proposal import PatchProposalEngine, PatchSuggestion, ProposalReport
from .models import AgentResponse, TurnRecord, PipelineTrace
from .llm_client import (
    AnthropicClient,
    OpenAIClient,
    FakeLLMClient,
    LLMClientProtocol,
    LLMError,
    LLMTimeoutError,
    LLMAuthError,
    LLMRateLimitError,
)

__all__ = [
    "AgentCore",
    "AgentConfig",
    "AgentResponse",
    "TurnRecord",
    "PipelineTrace",
    "AnthropicClient",
    "OpenAIClient",
    "FakeLLMClient",
    "LLMClientProtocol",
    "LLMError",
    "LLMTimeoutError",
    "LLMAuthError",
    "LLMRateLimitError",
]
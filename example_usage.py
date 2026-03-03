"""
A.C.C.E.S.S. — Memory Layer: Example Usage
Run: python example_usage.py
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from memory import MemoryManager, MemoryConfig
from pathlib import Path


# ── Setup ──────────────────────────────────────────────────────────────────

config = MemoryConfig(data_dir=Path("data/memory_demo"))
mem = MemoryManager(config)

print("\n" + "="*60)
print("A.C.C.E.S.S. Memory Layer — Demo")
print("="*60 + "\n")


# ── 1. ADD MEMORIES ────────────────────────────────────────────────────────

# A productive session
m1 = mem.add_memory(
    content="User completed the entire Python async module in one session, no distractions. Total focus time: 2h15m.",
    summary="User completed Python async module — 2h15m deep work.",
    memory_type="episodic",
    tags=["productivity", "python", "deep_work"],
    importance_score=0.85,
    source="interaction",
    session_id="session_001",
)
print(f"Added memory 1: {m1.id[:8]}... (importance={m1.importance_score})")

# An emotional event
m2 = mem.add_memory(
    content="User expressed frustration after failing to start the project for the 3rd day in a row. Said 'I don't know where to begin'.",
    summary="User frustrated — 3 days of procrastination on project start.",
    memory_type="emotional",
    tags=["procrastination", "frustration", "project"],
    importance_score=0.7,
    source="interaction",
)
print(f"Added memory 2: {m2.id[:8]}... (importance={m2.importance_score})")

# A factual preference
m3 = mem.add_memory(
    content="User prefers working in 90-minute blocks with 15-minute breaks. Dislikes Pomodoro 25-min format.",
    summary="User works best in 90-min blocks, not Pomodoro.",
    memory_type="semantic",
    tags=["preference", "productivity", "schedule"],
    importance_score=0.9,
    source="interaction",
)
print(f"Added memory 3: {m3.id[:8]}... (importance={m3.importance_score})")

# A low-importance event
m4 = mem.add_memory(
    content="User opened Spotify at 3pm.",
    summary="User opened Spotify at 3pm.",
    memory_type="episodic",
    tags=["music", "activity"],
    importance_score=0.2,
    source="os_event",
)
print(f"Added memory 4: {m4.id[:8]}... (importance={m4.importance_score})")


# ── 2. RETRIEVE RELEVANT MEMORIES ─────────────────────────────────────────

print("\n" + "-"*60)
print("RETRIEVE: 'user is procrastinating again'")
print("-"*60)

results = mem.retrieve_relevant_memories(
    query="user is procrastinating again",
    top_k=3,
)
for r in results:
    print(f"  → [{r.record.memory_type}] sim={r.similarity:.3f} | relevance={r.relevance:.3f}")
    print(f"     {r.record.summary}")

print("\n" + "-"*60)
print("RETRIEVE: 'how does user prefer to work'")
print("-"*60)

results2 = mem.retrieve_relevant_memories(
    query="how does user prefer to work",
    top_k=3,
    min_importance=0.5,  # ignore low-importance memories
)
for r in results2:
    print(f"  → [{r.record.memory_type}] sim={r.similarity:.3f} | relevance={r.relevance:.3f}")
    print(f"     {r.record.summary}")


# ── 3. FORMAT FOR RAG ──────────────────────────────────────────────────────

print("\n" + "-"*60)
print("RAG CONTEXT (ready to inject into LLM prompt):")
print("-"*60)
rag_block = mem.format_for_rag(results2)
print(rag_block)


# ── 4. UPDATE IMPORTANCE ──────────────────────────────────────────────────

print("\n" + "-"*60)
print("UPDATE IMPORTANCE")
print("-"*60)

# Absolute
new = mem.update_importance(m2.id, new_score=0.95)
print(f"  m2 absolute update → {new:.3f}")

# Relative (reward after task completion)
new2 = mem.update_importance(m1.id, delta=+0.1)
print(f"  m1 delta +0.1 → {new2:.3f}")

# Relative (decay low-signal memory)
new3 = mem.update_importance(m4.id, delta=-0.1)
print(f"  m4 delta -0.1 → {new3:.3f}")


# ── 5. STATS ───────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("STATS")
print("-"*60)
import json
print(json.dumps(mem.stats(), indent=2))

print("\n✅ Demo complete.\n")
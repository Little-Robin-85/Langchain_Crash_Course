"""
================================================================================
Module 11: LangChain Built-In Middleware & Production Patterns (Gemini)
================================================================================

CONCEPT OVERVIEW:
-----------------
As user sessions progress, the message list expands. If left unchecked:
1. Context limits are reached, resulting in API errors.
2. Inference latency increases linearly with prompt length.
3. Every turn becomes significantly more expensive.

LangChain includes built-in middleware to handle this automatically:
- `SummarizationMiddleware`: Compresses older turns into a compact summary while
  preserving the most recent N messages.

OTHER PRODUCTION MIDDLEWARE DISCUSSED IN THE VIDEO:
---------------------------------------------------
1. Human-In-The-Loop (HITL): Pauses agent execution before dangerous actions
   (e.g., executing SQL, sending payments, deleting files) to request human approval.
2. Call Limiters (Guardrails): Caps the maximum number of model or tool calls per
   thread to prevent expensive infinite loops.
3. Model Fallback: Automatically falls back to a secondary model if the primary
   model experiences rate limits (429) or transient server errors (503).
4. PII Redaction: Intercepts messages to detect and mask sensitive personal data
   (credit cards, SSNs, emails) before sending to external LLM providers.
5. Tool Retry with Backoff: Retries flaky network tools with exponential backoff.

GEMINI SPECIFICS:
-----------------
Using `gemini-3.5-flash-lite` for summarization provides lightning-fast compression
without noticeable latency hiccups.
================================================================================
"""

import os
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL = f"google_genai:{MODEL_NAME}" if not MODEL_NAME.startswith("google_genai:") else MODEL_NAME
SUMMARY_MODEL = MODEL  # Use the fast Flash model for summarizing


# ------------------------------------------------------------------------------
# 2. Configuring SummarizationMiddleware
# ------------------------------------------------------------------------------
# SummarizationMiddleware monitors token usage:
# - `trigger`: When token usage crosses this threshold (e.g. ('tokens', 200) for demo,
#   or 4000+ for production), compression triggers.
# - `keep`: Specifies how many recent messages to retain verbatim without summarizing.
#
# NOTE ON MODERN SYNTAX:
# In modern LangChain, use `trigger=('tokens', ...)` and `keep=('messages', ...)`
# to avoid deprecation warnings.
summarizer = SummarizationMiddleware(
    model=SUMMARY_MODEL,
    trigger=("tokens", 200),  # Low threshold for demonstration
    keep=("messages", 2),     # Keep the 2 most recent messages verbatim
)

agent = create_agent(
    model=MODEL,
    middleware=[summarizer],
    system_prompt="You are a senior software engineering lead and project planner.",
)


# ------------------------------------------------------------------------------
# 3. Simulating a Conversation with Automatic Summarization
# ------------------------------------------------------------------------------
def main():
    print("================================================================")
    print("Testing Built-in SummarizationMiddleware with Google Gemini")
    print("================================================================")

    # We send a long, detailed specification message to trigger summarization
    initial_messages = [
        {
            "role": "user",
            "content": (
                "Project Architecture Document:\n"
                "We are designing a cloud-native data processing agent in Python.\n"
                "Key requirements:\n"
                "1. Must use Google Gemini 3.5 Flash for high-speed inference.\n"
                "2. Must integrate wttr.in for real-time weather alerts.\n"
                "3. Must use FAISS vector database with gemini-embedding-001.\n"
                "4. Must support multimodal chart and diagram inspection.\n"
                "5. Must enforce strict Pydantic structured output models.\n"
                "6. Target deployment is Google Cloud Run with LangSmith observability."
            ),
        },
        {
            "role": "user",
            "content": "Given these requirements, what are our top 3 technical priorities?",
        },
    ]

    print("[Agent Invocation] Invoking agent with long architectural context...")
    response = agent.invoke({"messages": initial_messages})

    final_msg = response["messages"][-1]
    print("\n--- Agent Response ---")
    print(getattr(final_msg, "text", final_msg.content))

    # --------------------------------------------------------------------------
    # 4. Educational Reference: Middleware Catalog
    # --------------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("ENTERPRISE AGENT MIDDLEWARE CATALOG (From Crash Course)")
    print("=" * 64)
    catalog = [
        ("SummarizationMiddleware", "Automatically compacts conversation history when tokens exceed budget."),
        ("Human-In-The-Loop (HITL)", "Interrupts execution and prompts human approval for destructive tools."),
        ("Model Call Rate Limiter", "Guards against infinite loops by enforcing max model calls per execution."),
        ("Tool Call Throttler",     "Prevents external API quota exhaustion and rate limit bans."),
        ("Model Fallback Router",   "Swaps to backup provider if primary model returns 429 or 503 errors."),
        ("PII Sanitizer",          "Scans and redacts sensitive data (emails, SSNs, credit cards) before LLM calls."),
        ("Exponential Backoff Retry","Automatically retries transient network failures on tool calls."),
    ]

    for name, desc in catalog:
        print(f"• {name:<26} : {desc}")
    print("=" * 64)


if __name__ == "__main__":
    main()

"""
================================================================================
Module 10: Custom Agent Middleware & Lifecycle Hooks (Google Gemini)
================================================================================

CONCEPT OVERVIEW:
-----------------
In production systems, you often need cross-cutting concerns that apply to every
interaction:
- Profiling end-to-end and model-level latency.
- Audit logging for compliance (recording who asked what and when).
- Content moderation or safety guardrails.
- Injecting tracing identifiers (e.g., correlating with Datadog or OpenTelemetry).

Instead of cluttering your core business logic, LangChain 1.0 provides the
`AgentMiddleware` base class with lifecycle hooks.

THE 4 CORE LIFECYCLE HOOKS:
---------------------------
1. `before_agent`: Runs immediately upon invocation, before any agent reasoning.
2. `before_model`: Runs right before each call to the LLM (can run multiple times
   if the agent invokes tools in a loop).
3. `after_model`: Runs immediately after the LLM responds, before tool execution.
4. `after_agent`: Runs after the final answer is generated, right before returning
   to the caller.

WHY THIS PATTERN MATTERS:
-------------------------
Decouples observability, security, and performance tracking from your agent's
prompt and tool definitions.
================================================================================
"""

import os
import time
from typing import Any
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL = f"google_genai:{MODEL_NAME}" if not MODEL_NAME.startswith("google_genai:") else MODEL_NAME


# ------------------------------------------------------------------------------
# 2. Implementing the Custom Lifecycle Hooks Middleware
# ------------------------------------------------------------------------------
class PerformanceAuditMiddleware(AgentMiddleware):
    """A production-style middleware tracking agent lifecycles and measuring latency."""

    def __init__(self) -> None:
        super().__init__()
        self.agent_start_time: float = 0.0
        self.model_start_time: float = 0.0
        self.model_call_count: int = 0

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Hook 1: Triggered once when agent execution begins."""
        self.agent_start_time = time.perf_counter()
        self.model_call_count = 0
        message_count = len(state.get("messages", []))
        print("\n" + "=" * 64)
        print(f"[Hook: BEFORE_AGENT] Agent execution initiated with {message_count} initial message(s).")
        print("=" * 64)
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Hook 2: Triggered before each individual LLM inference step."""
        self.model_call_count += 1
        self.model_start_time = time.perf_counter()
        print(f"  [Hook: BEFORE_MODEL] Dispatching request #{self.model_call_count} to Google Gemini...")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Hook 3: Triggered immediately when the LLM returns its response."""
        duration = time.perf_counter() - self.model_start_time
        print(f"  [Hook: AFTER_MODEL] Received Gemini response #{self.model_call_count} in {duration:.2f}s.")
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Hook 4: Triggered once when the entire agent workflow completes."""
        total_elapsed = time.perf_counter() - self.agent_start_time
        print("=" * 64)
        print(f"[Hook: AFTER_AGENT] Agent execution complete!")
        print(f"  Total Duration:     {total_elapsed:.2f} seconds")
        print(f"  Total Model Calls:  {self.model_call_count}")
        print("=" * 64 + "\n")
        return None


# ------------------------------------------------------------------------------
# 3. Constructing Agent with Custom Middleware
# ------------------------------------------------------------------------------
# Pass instance of PerformanceAuditMiddleware in the middleware list
agent = create_agent(
    model=MODEL,
    middleware=[PerformanceAuditMiddleware()],
    system_prompt="You are an expert AI educator. Provide clear, concise explanations.",
)


# ------------------------------------------------------------------------------
# 4. Invoking Agent and Observing the Hooks in Action
# ------------------------------------------------------------------------------
def main():
    query = "What is the primary difference between a process and a thread in one paragraph?"
    print(f"Submitting query: '{query}'")

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    )

    final_msg = response["messages"][-1]
    print("--- Final Agent Output ---")
    print(getattr(final_msg, "text", final_msg.content))


if __name__ == "__main__":
    main()

"""
================================================================================
Module 09: Dynamic Model Choice with Middleware (Tiered Model Routing)
================================================================================

CONCEPT OVERVIEW:
-----------------
In production AI engineering, using your largest, most expensive model for every
single request is wasteful. Asking "What is 2+2?" does not need a heavy reasoning
engine; a lightweight, ultra-fast model can handle it with 10x lower latency and cost.

However, when a conversation grows long, involves subtle reasoning, or requires
deep multi-step synthesis, you want to automatically route to a stronger model.

WHY THIS PATTERN MATTERS:
-------------------------
The `@wrap_model_call` decorator in LangChain's middleware package allows you to
intercept model execution at runtime:
1. It receives the `ModelRequest` (containing messages, parameters, tools).
2. You can inspect message count, estimated token length, or task complexity.
3. You can override the target model using `request.override(model=...)`.
4. Then call `handler(...)` to execute the call with the chosen model.

GEMINI SPECIFICS:
-----------------
Google Gemini offers clear model tiers:
- Tier 1 (Light / Fast): `gemini-3.5-flash-lite` (low cost, blazing speed).
- Tier 2 (Advanced / Deep): `gemini-3.6-flash` (higher capacity, complex reasoning).
This module dynamically routes between them based on conversation depth.
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
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Define the two Gemini model tiers
FAST_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
ADVANCED_MODEL_NAME = os.getenv("ADVANCED_MODEL", "gemini-3.6-flash")

print(f"[Model Tiers Configured]")
print(f"  Tier 1 (Fast/Light): '{FAST_MODEL_NAME}'")
print(f"  Tier 2 (Advanced):   '{ADVANCED_MODEL_NAME}'\n")

fast_model = init_chat_model(FAST_MODEL_NAME, model_provider="google_genai", temperature=0.1)
advanced_model = init_chat_model(ADVANCED_MODEL_NAME, model_provider="google_genai", temperature=0.1)


# ------------------------------------------------------------------------------
# 2. Interceptor Middleware: Dynamic Model Routing
# ------------------------------------------------------------------------------
# `@wrap_model_call` wraps around the actual model invocation.
# It takes the `request: ModelRequest` and the next `handler` in the execution chain.
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Inspect conversation complexity and dynamically select the optimal model."""
    msg_count = len(request.messages)

    # Educational Decision Rule:
    # If the conversation has more than 3 messages (multi-turn context requiring deeper
    # synthesis), upgrade to the advanced model; otherwise use the fast model.
    if msg_count > 3:
        chosen_model = advanced_model
        model_tier = f"Tier 2 (Advanced: {ADVANCED_MODEL_NAME})"
    else:
        chosen_model = fast_model
        model_tier = f"Tier 1 (Fast: {FAST_MODEL_NAME})"

    print(f"  [Middleware: Model Router] Messages: {msg_count} -> Routed to {model_tier}")

    # Return the call by replacing the model in the request object
    updated_request = request.override(model=chosen_model)
    return handler(updated_request)


# ------------------------------------------------------------------------------
# 3. Constructing the Agent
# ------------------------------------------------------------------------------
# The agent is instantiated with the default fast model, but the middleware
# can dynamically swap it on any turn.
agent = create_agent(
    model=fast_model,
    middleware=[dynamic_model_selection],
)


# ------------------------------------------------------------------------------
# 4. Testing Short vs. Deep Conversations
# ------------------------------------------------------------------------------
def main():
    print("================================================================")
    print("TEST 1: Short Conversation (Should route to Fast Model)")
    print("================================================================")
    short_dialogue = [
        SystemMessage("You are a helpful and concise assistant."),
        HumanMessage("What is one plus one? Answer in one word."),
    ]
    short_response = agent.invoke({"messages": short_dialogue})
    last_short = short_response["messages"][-1]
    print(f"AI: {getattr(last_short, 'text', last_short.content).strip()}\n")

    print("================================================================")
    print("TEST 2: Long Conversation (Should route to Advanced Model)")
    print("================================================================")
    long_dialogue = [
        SystemMessage("You are a senior software architect."),
        HumanMessage("What is Python?"),
        AIMessage("Python is a high-level interpreted programming language."),
        HumanMessage("What makes it popular in machine learning?"),
        AIMessage("Extensive libraries like PyTorch and NumPy, combined with simple syntax."),
        HumanMessage("Compare its memory model with Rust's ownership model in two bullet points."),
    ]
    long_response = agent.invoke({"messages": long_dialogue})
    last_long = long_response["messages"][-1]
    print(f"AI:\n{getattr(last_long, 'text', last_long.content).strip()}\n")

    # Inspect metadata for both responses
    for label, res in [("Short Request", short_response), ("Long Request", long_response)]:
        msg = res["messages"][-1]
        meta = msg.response_metadata or {}
        model_used = meta.get("model_name") or meta.get("model") or "Gemini"
        print(f"[{label}] Output length: {len(getattr(msg, 'text', msg.content))} chars | Model: {model_used}")


if __name__ == "__main__":
    main()

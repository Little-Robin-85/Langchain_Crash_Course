"""
================================================================================
Module 02: Standalone Model Inference (Direct LLM Calls without an Agent)
================================================================================

CONCEPT OVERVIEW:
-----------------
Not every AI application requires an "Agent". Agents introduce latency and token
overhead because they must reason about which tools to call, execute the tools,
and summarize the results.

When your task is pure text transformation, classification, translation, or
direct Q&A that does not need real-time data or external side effects, you should
use "Standalone Model Inference".

WHY THIS PATTERN MATTERS:
-------------------------
LangChain provides `init_chat_model()`, a universal factory function that
initializes chat models across different providers (Google Gemini, OpenAI,
Anthropic, etc.) using a consistent interface.

Switching from OpenAI to Google Gemini or vice-versa requires changing only a
single configuration string without modifying any downstream application code.

GEMINI SPECIFICS:
-----------------
Using `google_genai:gemini-3.5-flash-lite`:
- Extremely fast time-to-first-token (TTFT).
- High rate limits and low inference costs.
- The returned `AIMessage` object contains the text content, token usage
  metadata (`usage_metadata`), and response metadata.
================================================================================
"""

import os
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

# Synchronize API keys so LangChain's Google integration can locate it
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

# Import LangChain's universal chat model factory
from langchain.chat_models import init_chat_model


# ------------------------------------------------------------------------------
# 2. Model Initialization via `init_chat_model`
# ------------------------------------------------------------------------------
# Syntax options for `init_chat_model`:
# Option A: `init_chat_model("google_genai:gemini-3.5-flash-lite", ...)`
# Option B: `init_chat_model("gemini-3.5-flash-lite", model_provider="google_genai", ...)`
#
# Parameters:
# - `temperature`: Controls randomness. 0.0 is deterministic and focused;
#   higher values (e.g., 0.7+) make the output more creative.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
print(f"[Initialization] Initializing Gemini model: '{MODEL_NAME}'...")

model = init_chat_model(
    MODEL_NAME,
    model_provider="google_genai",
    temperature=0.1,
)


# ------------------------------------------------------------------------------
# 3. Invoking the Model
# ------------------------------------------------------------------------------
def main():
    prompt = "Hello! In two sentences, what makes Python so popular among AI developers?"
    print(f"\n--- Sending Prompt: '{prompt}' ---")

    # `model.invoke()` sends the prompt to the Google Gemini API and waits for
    # the complete response. It returns an `AIMessage` object.
    response = model.invoke(prompt)

    # In LangChain, the generated text can be retrieved via `.text` (recommended)
    # or `.content`.
    content = getattr(response, "text", response.content)
    print("\n--- Model Response ---")
    print(content)

    # Educational Insight: Inspecting Token Usage Metadata
    # Gemini returns detailed token metrics (input tokens, output tokens, total tokens)
    # which are automatically parsed into `response.usage_metadata`.
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        print("\n--- Token Usage Metadata ---")
        usage = response.usage_metadata
        print(f"Input Tokens:  {usage.get('input_tokens')}")
        print(f"Output Tokens: {usage.get('output_tokens')}")
        print(f"Total Tokens:  {usage.get('total_tokens')}")


if __name__ == "__main__":
    main()

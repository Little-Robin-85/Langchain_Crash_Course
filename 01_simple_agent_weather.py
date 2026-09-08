"""
================================================================================
Module 01: Simple AI Agent with Tool Calling (Google Gemini Edition)
================================================================================

CONCEPT OVERVIEW:
-----------------
What is an AI Agent?
A standard LLM call takes a prompt and produces text based solely on the data
it was trained on. It cannot access real-time information or take actions in the
real world.

An "AI Agent" combines an LLM with "Tools" and a decision-making loop
(often called the ReAct loop: Reason -> Act -> Observe -> Repeat).
1. The user asks a question (e.g., "What is the weather like in Vienna?").
2. The Agent reasons that it does not possess live weather data.
3. The Agent inspects available tools and decides to call `get_weather(city="Vienna")`.
4. The framework executes the function on behalf of the Agent and feeds the
   output back to the LLM.
5. The LLM synthesizes the tool result into a natural, helpful final answer.

WHY THIS PATTERN MATTERS:
-------------------------
In LangChain 1.0+, `create_agent` provides a standardized, high-level API
built directly on top of LangGraph. This replaces legacy, brittle agent executors
with declarative, production-ready state graphs.

GEMINI SPECIFICS:
-----------------
Google Gemini models (like `gemini-3.5-flash-lite` and `gemini-3.6-flash`) have
native tool-calling capabilities. When you pass a `@tool` to `create_agent`,
LangChain automatically translates the Python function signature and docstring
into the OpenAPI-compatible JSON schema that Gemini expects.
================================================================================
"""

import os
import requests
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment Configuration & API Key Synchronization
# ------------------------------------------------------------------------------
# Load environment variables from the .env file located in the project root.
load_dotenv()

# Google Gemini API key handling:
# LangChain Google GenAI integration (`langchain-google-genai`) checks for
# `GOOGLE_API_KEY` or `GEMINI_API_KEY`. To ensure robust compatibility across
# all environments, we guarantee that `GOOGLE_API_KEY` is populated.
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

# Select the model:
# "gemini-3.5-flash-lite" is the recommended default for fast, cost-effective tool use.
# The format "google_genai:<model_name>" tells LangChain to use the official
# `langchain-google-genai` provider.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL = f"google_genai:{MODEL_NAME}" if not MODEL_NAME.startswith("google_genai:") else MODEL_NAME

# Import the core LangChain 1.0 agent and tool abstractions
from langchain.agents import create_agent
from langchain.tools import tool


# ------------------------------------------------------------------------------
# 2. Defining Tools using the `@tool` Decorator
# ------------------------------------------------------------------------------
# The `@tool` decorator converts a regular Python function into a StructuredTool
# object that an LLM can understand and invoke.
#
# CRITICAL RULE FOR TOOL DEFINITIONS:
# The function name, argument type hints, and the `description` (or docstring)
# constitute the "prompt" given to the LLM when deciding which tool to invoke.
# Always provide clear descriptions and explicit types.
@tool(
    "get_weather",
    description="Return real-time weather information for a given city.",
    return_direct=False,
)
def get_weather(city: str) -> dict:
    """Fetch current weather data for `city` using the free wttr.in JSON API."""
    print(f"\n[Tool Execution] Calling external weather API for city: '{city}'...")
    # wttr.in with '?format=j1' returns structured JSON weather data
    response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=20)
    response.raise_for_status()
    data = response.json()
    
    # Extract only the key elements (current condition) to conserve context tokens
    current = data.get("current_condition", [{}])[0]
    simplified_weather = {
        "city": city,
        "temperature_C": current.get("temp_C"),
        "temperature_F": current.get("temp_F"),
        "condition": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
        "humidity": current.get("humidity"),
        "windspeed_kmph": current.get("windspeedKmph"),
    }
    print(f"[Tool Execution] Received weather data: {simplified_weather}\n")
    return simplified_weather


# ------------------------------------------------------------------------------
# 3. Constructing the Agent
# ------------------------------------------------------------------------------
# `create_agent` combines:
# 1. `model`: The chat model (Google Gemini via google_genai provider).
# 2. `tools`: A list of callable tools available to the model.
# 3. `system_prompt`: High-level persona and behavioral instructions guiding the agent.
agent = create_agent(
    model=MODEL,
    tools=[get_weather],
    system_prompt=(
        "You are a helpful and very serious weather assistant powered by Google Gemini. "
        "When asked about the weather in a city, always use the `get_weather` tool "
        "to retrieve live data before answering. Be concise, accurate, and include "
        "a tip or observation, exagerate the importance even though its only the weather!."
        "Always default to Degrees Celcious and MPH."
    ),
)


# ------------------------------------------------------------------------------
# 4. Invoking the Agent
# ------------------------------------------------------------------------------
# Agents accept an input dictionary containing a list of `messages`.
# Each message can be a dictionary (`{"role": "user", "content": ...}`)
# or an explicit LangChain message object (`HumanMessage(...)`).
def main():
    user_query = "What is the weather like in Liverpool today?"
    print(f"--- User Query: {user_query} ---")

    # The agent orchestrates:
    # Query -> Model decides to call get_weather -> Tool executes -> Model formats final reply
    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_query,
                }
            ]
        }
    )

    # In LangGraph-backed agents, the return value contains the full list of messages
    # generated during the run (HumanMessage -> AIMessage with tool_calls -> ToolMessage -> AIMessage final).
    # The final answer is the last message in the sequence.
    final_message = response["messages"][-1]
    
    # Use `.text` if available (standard in modern LangChain) or fall back to `.content`
    answer = getattr(final_message, "text", final_message.content)
    
    print("--- Final Agent Response ---")
    print(answer)


if __name__ == "__main__":
    main()

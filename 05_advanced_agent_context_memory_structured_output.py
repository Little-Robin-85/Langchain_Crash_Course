"""
================================================================================
Module 05: Advanced Agent (Context, Thread Memory & Structured Output)
================================================================================

CONCEPT OVERVIEW:
-----------------
Real-world enterprise AI agents require more than just a single prompt-and-reply:
1. RUNTIME CONTEXT: Tools frequently need application context (e.g. current
   user ID, organization ID, auth tokens) that the user didn't explicitly type
   and that the LLM shouldn't have to guess or hallucinate.
2. THREAD MEMORY (CHECKPOINTING): An agent must maintain conversation state
   across multiple invocations using persistent checkpoints (`InMemorySaver`
   or database savers like Postgres).
3. STRUCTURED OUTPUT: Instead of free-form unstructured markdown text, downstream
   software systems need predictable, typed data schemas validated by Pydantic.

WHY THIS PATTERN MATTERS:
-------------------------
In LangChain 1.0 (powered by LangGraph):
- `context_schema`: Defines the Python dataclass injected at runtime into tools
  via `ToolRuntime[Context]`.
- `checkpointer`: Maintains graph execution state across turns keyed by `thread_id`.
- `response_format`: Forces the agent's final answer into a validated Pydantic model.

GEMINI SPECIFICS:
-----------------
Google Gemini natively supports both structured schema output (`response_schema` /
JSON mode) and function calling. LangChain orchestrates both seamlessly:
Gemini first calls `locate_user` -> then `get_weather` -> then formats the final
payload matching `WeatherResponse`.
================================================================================
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver


# ------------------------------------------------------------------------------
# 2. Defining Runtime Context Schema
# ------------------------------------------------------------------------------
# The Context object holds application state that is passed to the agent
# invocation and made accessible to tools without being exposed to the user prompt.
@dataclass
class UserContext:
    user_id: str
    preferred_unit: str = "metric"


# ------------------------------------------------------------------------------
# 3. Defining Pydantic Schema for Structured Response
# ------------------------------------------------------------------------------
# Downstream applications (APIs, databases, frontends) need strongly-typed data.
class WeatherResponse(BaseModel):
    summary: str = Field(
        description="A concise natural-language weather summary and recommendation."
    )
    city: str = Field(
        description="The city for which the weather was retrieved."
    )
    temperature_celsius: Optional[float] = Field(
        default=None, description="Temperature in degrees Celsius."
    )
    temperature_fahrenheit: Optional[float] = Field(
        default=None, description="Temperature in degrees Fahrenheit."
    )
    condition: Optional[str] = Field(
        default=None, description="Current weather condition (e.g., Sunny, Cloudy)."
    )
    humidity: Optional[float] = Field(
        default=None, description="Relative humidity percentage."
    )


# ------------------------------------------------------------------------------
# 4. Defining Tools with Runtime Context Injection
# ------------------------------------------------------------------------------
@tool("get_weather", description="Return weather information for a given city.")
def get_weather(city: str) -> dict:
    """Fetch current weather data for a city from wttr.in."""
    print(f"  [Tool: get_weather] Querying weather for: '{city}'...")
    response = requests.get(f"https://wttr.in/{city}?format=j1", timeout=20)
    response.raise_for_status()
    data = response.json()
    current = data.get("current_condition", [{}])[0]
    return {
        "city": city,
        "temp_C": float(current.get("temp_C", 0)),
        "temp_F": float(current.get("temp_F", 0)),
        "condition": current.get("weatherDesc", [{}])[0].get("value", "Unknown"),
        "humidity": float(current.get("humidity", 0)),
    }


@tool("locate_user", description="Look up a user's location based on runtime context.")
def locate_user(runtime: ToolRuntime[UserContext]) -> str:
    """Access runtime context securely without relying on user prompt."""
    uid = runtime.context.user_id
    print(f"  [Tool: locate_user] Looking up profile for user_id='{uid}'...")
    # Simulated internal user database lookup
    mock_db = {
        "USR-101": "Vienna",
        "USR-202": "London",
        "USR-303": "Tokyo",
    }
    location = mock_db.get(uid, "unknown")
    print(f"  [Tool: locate_user] Resolved location: '{location}'")
    return location


# ------------------------------------------------------------------------------
# 5. Constructing the Advanced Agent
# ------------------------------------------------------------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
model = init_chat_model(MODEL_NAME, model_provider="google_genai", temperature=0.2)

# InMemorySaver acts as the agent's short-term memory checkpointer
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, locate_user],
    system_prompt=(
        "You are an advanced weather assistant. "
        "1. If the user does not specify a city, call `locate_user` first. "
        "2. Once the city is known, call `get_weather`. "
        "3. Provide your final response strictly formatted according to the WeatherResponse schema."
    ),
    context_schema=UserContext,
    response_format=WeatherResponse,
    checkpointer=checkpointer,
)


# ------------------------------------------------------------------------------
# 6. Invoking the Agent Across Multiple Turns (Testing Memory & Schema)
# ------------------------------------------------------------------------------
def main():
    # Thread ID identifies the conversation session in the checkpointer
    thread_config = {"configurable": {"thread_id": "weather-session-42"}}
    runtime_context = UserContext(user_id="USR-202", preferred_unit="metric")

    print("================================================================")
    print("TURN 1: Ambiguous user query (requires locate_user + get_weather)")
    print("================================================================")
    user_query_1 = "What is the weather like right now?"
    print(f"User: '{user_query_1}' (Notice: user did not say their city!)")

    turn1_output = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_1}]},
        context=runtime_context,
        config=thread_config,
    )

    # In LangChain 1.0 with response_format, the validated Pydantic model
    # is available under `structured_response`
    structured_result = turn1_output.get("structured_response")
    print("\n--- Structured Response Object (Turn 1) ---")
    if structured_result:
        print(f"Type:        {type(structured_result).__name__}")
        print(f"City:        {structured_result.city}")
        print(f"Condition:   {structured_result.condition}")
        print(f"Celsius:     {structured_result.temperature_celsius}°C")
        print(f"Fahrenheit:  {structured_result.temperature_fahrenheit}°F")
        print(f"Humidity:    {structured_result.humidity}%")
        print(f"Summary:     {structured_result.summary}")
    else:
        # Fallback to message content if structured response key differs
        last_msg = turn1_output["messages"][-1]
        print(getattr(last_msg, "text", last_msg.content))

    print("\n================================================================")
    print("TURN 2: Follow-up question relying on Thread Memory")
    print("================================================================")
    user_query_2 = "Is this temperature considered warm or cold for this city?"
    print(f"User: '{user_query_2}' (Notice: relies on Vienna and the previous temp!)")

    turn2_output = agent.invoke(
        {"messages": [{"role": "user", "content": user_query_2}]},
        context=runtime_context,
        config=thread_config,
    )

    turn2_structured = turn2_output.get("structured_response")
    if turn2_structured:
        print(f"\nSummary: {turn2_structured.summary}")
    else:
        last_msg = turn2_output["messages"][-1]
        print("\nAI:", getattr(last_msg, "text", last_msg.content))


if __name__ == "__main__":
    main()

"""
================================================================================
Module 08: Dynamic System Prompts with Middleware (Google Gemini Edition)
================================================================================

CONCEPT OVERVIEW:
-----------------
In production AI systems, a "one-size-fits-all" system prompt often leads to poor
user experiences. An expert data scientist asking about PCA (Principal Component
Analysis) needs mathematical rigor and eigenvalues, while a 10-year-old child
needs a simple, visual metaphor.

Instead of creating separate agent instances for every user persona, LangChain
provides "Agent Middleware" using the `@dynamic_prompt` decorator.

WHY THIS PATTERN MATTERS:
-------------------------
Middleware executes in the agent pipeline before the request hits the model.
With `@dynamic_prompt`, you can:
1. Inspect runtime context (e.g., user role, subscription tier, language preference).
2. Generate customized system instructions on-the-fly.
3. Keep your agent architecture clean, single-instance, and completely DRY
   (Don't Repeat Yourself).

GEMINI SPECIFICS:
-----------------
Google Gemini models adhere strictly to `system_instruction`. Changing the dynamic
prompt directly shapes the model's tone, vocabulary, and conceptual depth without
altering the user's raw question.
================================================================================
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL = f"google_genai:{MODEL_NAME}" if not MODEL_NAME.startswith("google_genai:") else MODEL_NAME


# ------------------------------------------------------------------------------
# 2. Defining the Context Schema
# ------------------------------------------------------------------------------
# This dataclass captures user-specific metadata passed at invocation time
@dataclass
class UserSession:
    user_role: str  # Options: "child", "beginner", "expert"
    user_name: str = "Learner"


# ------------------------------------------------------------------------------
# 3. Implementing Dynamic System Prompt Middleware
# ------------------------------------------------------------------------------
# `@dynamic_prompt` intercepts the `ModelRequest` and returns a string that replaces
# or prepends the system instruction for that specific invocation turn.
@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Inspect runtime context and return an adaptive system prompt."""
    context: UserSession = request.runtime.context
    role = getattr(context, "user_role", "beginner")

    print(f"  [Middleware: @dynamic_prompt] Adapting instructions for role: '{role}'...")

    match role:
        case "expert":
            return (
                "You are an elite machine learning professor and mathematician. "
                "Explain concepts with technical rigor, mentioning linear algebra, "
                "matrices, eigenvalues/eigenvectors, and dimensionality reduction. "
                "Do not dumb down explanations."
            )
        case "child":
            return (
                "You are a friendly storyteller explaining science to an 8-year-old child. "
                "Use very simple words, a playful metaphor (like organizing a toy box), "
                "and no math or jargon."
            )
        case _:
            return (
                "You are an approachable coding tutor. Explain clearly and concisely "
                "for someone learning to program, focusing on intuition without heavy math."
            )


# ------------------------------------------------------------------------------
# 4. Constructing the Agent with Middleware
# ------------------------------------------------------------------------------
agent = create_agent(
    model=MODEL,
    middleware=[user_role_prompt],
    context_schema=UserSession,
)


# ------------------------------------------------------------------------------
# 5. Invoking the Agent with Different Personas
# ------------------------------------------------------------------------------
def main():
    question = "Explain Principal Component Analysis (PCA) in two short paragraphs."
    personas = ["child", "beginner", "expert"]

    print("================================================================")
    print(f"Testing Prompt Adaptability for Question: '{question}'")
    print("================================================================")

    for role in personas:
        print(f"\n>>> Running Invocation with Persona: [{role.upper()}] <<<")
        session_context = UserSession(user_role=role, user_name="Alex")

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            context=session_context,
        )

        final_msg = response["messages"][-1]
        content = getattr(final_msg, "text", final_msg.content)
        print(f"\n--- Gemini Output for [{role.upper()}] ---")
        print(content)
        print("-" * 64)


if __name__ == "__main__":
    main()

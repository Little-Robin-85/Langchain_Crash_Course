"""
================================================================================
Module 12: LangSmith Tracing & Observability Deep Dive (Google Gemini)
================================================================================

CONCEPT OVERVIEW:
-----------------
Yes! Every single module in this crash course (01 through 11) is ALREADY
configured with and actively sending telemetry to LangSmith.

HOW LANGSMITH TRACING WORKS UNDER THE HOOD:
-------------------------------------------
LangChain and LangGraph have native, zero-code telemetry instrumentation built in.
Whenever a Python script calls `load_dotenv()`, it reads your `.env` file:
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_API_KEY="lsv2_pt_..."
    LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"
    LANGCHAIN_PROJECT="LangGraph-Tutorial"

The moment `LANGCHAIN_TRACING_V2="true"` is detected in `os.environ`:
1. Every LLM call (Gemini prompt tokens, completion tokens, latency, temperature).
2. Every Agent execution graph (ReAct reasoning steps, node transitions).
3. Every Tool call (input arguments, execution time, returned payloads).
4. Every Vector retrieval query and similarity score.
5. Every Middleware lifecycle hook.

...are AUTOMATICALLY streamed in real time to your LangSmith project!

WHAT THIS MODULE DEMONSTRATES:
------------------------------
1. Verifying active LangSmith connectivity.
2. Capturing the direct, clickable LangSmith Trace URL in Python using `collect_runs()`.
3. Adding custom run names, tags, and metadata to your agent invocations.
4. Using the `@traceable` decorator to trace regular Python helper functions.
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

from langsmith import Client, traceable
from langchain_core.tracers.context import collect_runs
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "LangGraph-Tutorial")
ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://eu.api.smith.langchain.com")

print("=" * 70)
print(f"LANGSMITH CONFIGURATION DETECTED:")
print(f"  • Tracing Enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"  • Project Name:    {PROJECT_NAME}")
print(f"  • Endpoint:        {ENDPOINT}")
print("=" * 70)


# ------------------------------------------------------------------------------
# 2. Tracing Custom Python Functions with `@traceable`
# ------------------------------------------------------------------------------
# You can trace any regular Python function (data preprocessing, business logic)
# alongside your LLM calls using LangSmith's `@traceable` decorator.
@traceable(run_type="tool", name="discount_calculator")
def calculate_discount(price: float, member_tier: str) -> float:
    """Calculate discounted price based on customer loyalty tier."""
    discount_rates = {"gold": 0.20, "silver": 0.10, "bronze": 0.05}
    rate = discount_rates.get(member_tier.lower(), 0.0)
    return round(price * (1.0 - rate), 2)


# ------------------------------------------------------------------------------
# 3. Defining a Tool that Incorporates Custom Traced Functions
# ------------------------------------------------------------------------------
@tool("get_product_price", description="Look up pricing and discounts for a product.")
def get_product_price(product_name: str, member_tier: str = "standard") -> dict:
    """Return pricing details for a product with discount applied."""
    base_prices = {"laptop": 1200.0, "headphones": 150.0, "keyboard": 90.0}
    base = base_prices.get(product_name.lower(), 100.0)
    final = calculate_discount(base, member_tier)
    return {
        "product": product_name,
        "base_price": base,
        "member_tier": member_tier,
        "final_price": final,
    }


# ------------------------------------------------------------------------------
# 4. Constructing and Tracing an Agent Run with Custom Metadata & URL
# ------------------------------------------------------------------------------
def main():
    client = Client()
    model = init_chat_model(MODEL_NAME, model_provider="google_genai", temperature=0.1)

    agent = create_agent(
        model=model,
        tools=[get_product_price],
        system_prompt="You are an e-commerce sales assistant. Help customers check product prices.",
    )

    # We can pass custom metadata, tags, and run names via the `config` dictionary.
    # These tags make it easy to filter, monitor, and evaluate runs in the LangSmith dashboard.
    run_config = {
        "run_name": "ECommerce_Customer_Inquiry",
        "tags": ["gemini-3.5", "crash-course-demo", "pricing-agent"],
        "metadata": {
            "environment": "production",
            "customer_id": "CUST-9921",
            "session_source": "web_chat",
        },
    }

    user_query = "How much would a laptop cost for a gold tier member?"
    print(f"\nUser Query: '{user_query}'")
    print("[Tracing] Executing agent within `collect_runs()` context...")

    # `collect_runs()` captures the exact run object(s) generated during execution
    with collect_runs() as cb:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            config=run_config,
        )

        final_msg = response["messages"][-1]
        print("\n--- Agent Response ---")
        print(getattr(final_msg, "text", final_msg.content))

        # Retrieve the LangSmith run object and URL
        if cb.traced_runs:
            root_run = cb.traced_runs[0]
            trace_url = client.get_run_url(run=root_run)

            print("\n" + "=" * 70)
            print("🎯 DIRECT LANGSMITH TRACE GENERATED:")
            print(f"Run ID:   {root_run.id}")
            print(f"Run Name: {root_run.name}")
            print(f"View in LangSmith: \n👉 {trace_url}")
            print("=" * 70)


if __name__ == "__main__":
    main()

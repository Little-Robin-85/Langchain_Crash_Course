"""
================================================================================
Module 07: Retrieval-Augmented Generation (RAG) with Gemini & FAISS
================================================================================

CONCEPT OVERVIEW:
-----------------
Large Language Models have static knowledge: they only know what they were trained
on, and they hallucinate when asked about private documents or rapidly changing data.

Retrieval-Augmented Generation (RAG) solves this by:
1. EMBEDDING: Converting text chunks into high-dimensional numerical vectors
   (embeddings) that capture semantic meaning.
2. VECTOR STORE: Storing vectors in an index (e.g., FAISS) optimized for fast
   nearest-neighbor mathematical search.
3. RETRIEVAL: When a user asks a question, finding the top-K most semantically
   relevant chunks.
4. GENERATION: Passing those retrieved chunks to the LLM (or an Agent with a
   retrieval tool) to ground the answer strictly in facts.

WHY THIS PATTERN MATTERS:
-------------------------
Rather than hardcoding a rigid retrieval chain, LangChain enables exposing your
vector database to an agent as a tool via `create_retriever_tool`.
The agent can decide *whether*, *when*, and *how many times* to search the
knowledge base before formulating a response.

GEMINI SPECIFICS:
-----------------
This module uses Google's `gemini-embedding-001` (producing 3,072-dimensional
embeddings) and `gemini-3.5-flash-lite` for agent synthesis.
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
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configuration
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
AGENT_MODEL = f"google_genai:{CHAT_MODEL}" if not CHAT_MODEL.startswith("google_genai:") else CHAT_MODEL

# Embedding model: In .env, Embedding="gemini-embedding-001"
EMBEDDING_MODEL = os.getenv("Embedding", os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))
print(f"[RAG Setup] Embedding Model: '{EMBEDDING_MODEL}', Chat Model: '{AGENT_MODEL}'\n")

# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


# ------------------------------------------------------------------------------
# 2. Part 1: Direct Vector Store Similarity Search (Intuition Building)
# ------------------------------------------------------------------------------
def demonstrate_similarity_search():
    print("--- PART 1: Demonstrating Semantic Vector Similarity ---")
    texts = [
        "Apple makes very good computers and mobile devices.",
        "I believe Apple is an innovative tech company.",
        "I love eating crisp apples in the morning.",
        "I am a huge fan of MacBooks for software development.",
        "I enjoy sweet citrus oranges.",
        "I like Lenovo ThinkPads for their comfortable keyboards.",
        "I think pears and peaches taste very good.",
    ]

    print(f"Indexing {len(texts)} sentences into FAISS vector database...")
    # FAISS.from_texts generates embeddings for each string and creates an in-memory vector index
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)

    # Query 1: Fruit semantic search
    fruit_query = "Apples and pears are delicious fruit."
    print(f"\n[Query 1]: '{fruit_query}'")
    fruit_matches = vectorstore.similarity_search(fruit_query, k=3)
    for i, doc in enumerate(fruit_matches, 1):
        print(f"  {i}. {doc.page_content}")

    # Query 2: Operating system & Laptop semantic search
    tech_query = "What laptops are great for software engineers?"
    print(f"\n[Query 2]: '{tech_query}'")
    tech_matches = vectorstore.similarity_search(tech_query, k=3)
    for i, doc in enumerate(tech_matches, 1):
        print(f"  {i}. {doc.page_content}")


# ------------------------------------------------------------------------------
# 3. Part 2: RAG Agent with Retriever Tool
# ------------------------------------------------------------------------------
def demonstrate_rag_agent():
    print("\n================================================================")
    print("--- PART 2: Connecting the Knowledge Base to an AI Agent ---")
    print("================================================================")

    # A private personal preferences knowledge base
    knowledge_documents = [
        "I love apples.",
        "I enjoy oranges.",
        "I think pears taste very good.",
        "I hate bananas.",
        "I dislike raspberries.",
        "I despise mangoes.",
        "I love working on Arch Linux.",
        "I dislike proprietary Windows telemetry.",
    ]

    print(f"Building personal preference knowledge index ({len(knowledge_documents)} facts)...")
    kb_store = FAISS.from_texts(knowledge_documents, embedding=embeddings)

    # Convert the vector store into a LangChain Retriever
    # search_kwargs={"k": 4} instructs the retriever to return the top 4 closest documents
    retriever = kb_store.as_retriever(search_kwargs={"k": 4})

    # Wrap the retriever as a StructuredTool that the agent can call
    # Note: The tool name and description tell the agent what kind of information
    # this database contains and when it should search it.
    retriever_tool = create_retriever_tool(
        retriever,
        name="kb_search",
        description=(
            "Search the user's personal knowledge base for their personal "
            "preferences, liked/disliked fruits, and favorite technology."
        ),
    )

    # Create the Agent using create_agent
    agent = create_agent(
        model=AGENT_MODEL,
        tools=[retriever_tool],
        system_prompt=(
            "You are a helpful personal assistant with access to the user's "
            "personal knowledge base via the `kb_search` tool. "
            "Always query `kb_search` to verify personal preferences before answering. "
            "Do not guess. Ground your answers strictly in the retrieved facts."
        ),
    )

    user_question = "Which three fruits does this person like, and which three fruits do they hate?"
    print(f"\nUser Question: '{user_question}'")
    print("[Agent Loop] Agent is reasoning, searching vector store, and synthesizing...")

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_question,
                }
            ]
        }
    )

    final_msg = response["messages"][-1]
    answer = getattr(final_msg, "text", final_msg.content)
    print("\n--- Grounded Agent Answer ---")
    print(answer)


def main():
    demonstrate_similarity_search()
    demonstrate_rag_agent()


if __name__ == "__main__":
    main()

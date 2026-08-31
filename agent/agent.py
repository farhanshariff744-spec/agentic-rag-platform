"""
LangGraph Agent
===============
A multi-step agent that decides, per question, whether it needs to look
up SEC filings at all before answering. Simple conversational input
("hello", "what can you help with") gets answered directly with no
retrieval; substantive questions go through the same cache -> retrieve
-> generate pipeline built in inference/answer.py.

This is what turns the platform from "one function that always searches"
into something that actually reasons about what to do first.

Usage:
    python agent/agent.py "What are Apple's main business risks?"
    python agent/agent.py "hello, what can you do?"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TypedDict

import requests
from langgraph.graph import StateGraph, START, END

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.build_index import get_model, get_client, retrieve_chunks
from cache.semantic_cache import check_cache, store_in_cache
from prompts.prompt_templates import ROUTER_PROMPT, ANSWER_PROMPT, DIRECT_PROMPT, PROMPT_VERSION

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GROQ_API_KEY", "")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


class AgentState(TypedDict):
    question: str
    needs_retrieval: bool
    answer: str
    sources: list
    from_cache: bool


def call_groq(prompt: str, max_tokens: int = 500, retries: int = 3) -> str:
    """Shared helper for calling Groq -- same pattern used in guardrails/inference.

    Retries with backoff on 429 (rate limit) rather than crashing outright --
    Groq's free tier has a requests-per-minute cap that's easy to hit during
    heavy testing.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(retries):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            wait = 8 * (attempt + 1)
            print(f"Rate limited by Groq, waiting {wait}s before retry ({attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""

    raise RuntimeError("Groq rate limit exceeded after repeated retries")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def route_node(state: AgentState) -> AgentState:
    response = call_groq(ROUTER_PROMPT + state["question"], max_tokens=150)
    start, end = response.find("{"), response.rfind("}")
    try:
        parsed = json.loads(response[start:end + 1])
        needs_retrieval = bool(parsed.get("needs_retrieval", True))
        print(f"Router decision: needs_retrieval={needs_retrieval} ({parsed.get('reason', '')})")
    except (ValueError, json.JSONDecodeError):
        needs_retrieval = True  # default to the safer/more thorough path on parse failure
        print("Router decision: defaulting to needs_retrieval=True (parse failed)")
    return {**state, "needs_retrieval": needs_retrieval}


def direct_answer_node(state: AgentState) -> AgentState:
    answer = call_groq(DIRECT_PROMPT.format(question=state["question"]), max_tokens=300)
    return {**state, "answer": answer, "sources": [], "from_cache": False}


def retrieval_answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    model = get_model()
    client = get_client()

    cached = check_cache(question, model, client)
    if cached:
        print(f"Cache hit (similarity={cached['similarity']:.3f})")
        return {**state, "answer": cached["answer"], "sources": cached["sources"], "from_cache": True}

    chunks = retrieve_chunks(question, top_k=5, model=model, client=client)
    if not chunks:
        return {**state, "answer": "No relevant filings found in the index.",
                 "sources": [], "from_cache": False}

    context = "\n\n---\n\n".join(
        f"[{c['ticker']} {c['form']} filed {c['filing_date']}]\n{c['text']}" for c in chunks
    )
    answer = call_groq(ANSWER_PROMPT.format(context=context, question=question), max_tokens=2000)
    sources = [{"ticker": c["ticker"], "form": c["form"],
                "filing_date": c["filing_date"], "score": c["score"]} for c in chunks]

    store_in_cache(question, answer, sources, model, client)
    return {**state, "answer": answer, "sources": sources, "from_cache": False}


def route_decision(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else "direct"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("direct", direct_answer_node)
    graph.add_node("retrieve", retrieval_answer_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", route_decision, {"direct": "direct", "retrieve": "retrieve"})
    graph.add_edge("direct", END)
    graph.add_edge("retrieve", END)

    return graph.compile()


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Run: export $(cat .env | xargs)")

    parser = argparse.ArgumentParser(description="Ask the agent a question")
    parser.add_argument("question", help="The question to ask")
    args = parser.parse_args()

    agent = build_agent()
    result = agent.invoke({"question": args.question, "needs_retrieval": False,
                            "answer": "", "sources": [], "from_cache": False})

    print("\n" + "=" * 60)
    print(f"ANSWER: (used retrieval: {result['needs_retrieval']}, from cache: {result['from_cache']})")
    print(result["answer"])
    if result["sources"]:
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s['ticker']} {s['form']} ({s['filing_date']}) score={s['score']:.3f}")

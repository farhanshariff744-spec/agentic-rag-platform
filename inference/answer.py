"""
Inference Layer
================
Takes a user question, retrieves relevant chunks via the retrieval module,
and asks an LLM (Groq) to answer using ONLY that retrieved context.

This is the first point in the pipeline where you get an actual answer,
rather than raw chunks -- retrieval finds the evidence, inference reads
it and responds.

Usage:
    python inference/answer.py "What are Apple's main business risks?"
"""

import argparse
import os
import sys
from pathlib import Path

import requests

# Allow importing the retrieval module from the sibling folder, since this
# script runs as `python inference/answer.py` rather than as part of an
# installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.build_index import get_model, get_client, retrieve_chunks
from cache.semantic_cache import check_cache, store_in_cache
from prompts.prompt_templates import ANSWER_PROMPT, PROMPT_VERSION

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # same current, non-deprecated model used in guardrails


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[{c['ticker']} {c['form']} filed {c['filing_date']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(question: str, top_k: int = 5) -> dict:
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Run: export $(cat .env | xargs)")

    model = get_model()
    client = get_client()

    cached = check_cache(question, model, client)
    if cached:
        print(f"Cache hit (similarity={cached['similarity']:.3f}, "
              f"matched: \"{cached['cached_question']}\")")
        return {"answer": cached["answer"], "sources": cached["sources"], "from_cache": True}

    print(f"Cache miss. Retrieving top {top_k} chunks for: {question}")
    chunks = retrieve_chunks(question, top_k=top_k, model=model, client=client)

    if not chunks:
        return {"answer": "No relevant filings found in the index.", "sources": [], "from_cache": False}

    context = build_context(chunks)
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    print("Asking the model...")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 2000,  # gpt-oss-20b spends some budget on internal reasoning
                             # before writing the final answer; too tight a limit
                             # risks truncating mid-answer on longer responses.
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]
    answer_text = message.get("content") or ""

    if not answer_text.strip():
        # Fall back to a reasoning field if the provider exposes one, and
        # print the raw response so this is debuggable rather than silent.
        answer_text = message.get("reasoning", "")
        if not answer_text.strip():
            print("WARNING: model returned empty content. Raw response:")
            print(resp.json())
            answer_text = "(No answer text returned by the model -- see raw response above.)"

    sources = [
        {"ticker": c["ticker"], "form": c["form"],
         "filing_date": c["filing_date"], "score": c["score"]}
        for c in chunks
    ]
    store_in_cache(question, answer_text, sources, model, client)

    return {"answer": answer_text, "sources": sources, "from_cache": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask a question against the indexed filings")
    parser.add_argument("question", help="The question to ask")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    result = answer_question(args.question, args.top_k)

    print("\n" + "=" * 60)
    print(f"ANSWER: (from cache: {result['from_cache']})")
    print(result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(f"  - {s['ticker']} {s['form']} ({s['filing_date']}) score={s['score']:.3f}")

"""
Evaluation Harness
==================
Implements the core RAG evaluation metrics that RAGAS/DeepEval provide --
faithfulness, answer relevancy, and context precision -- as direct
LLM-as-judge calls, using the same request/JSON-parsing pattern already
proven reliable elsewhere in this codebase (guardrails Stage 2, the
agent's router).

Metrics:
- Faithfulness: does every claim in the answer trace back to the
  retrieved context, or did the model add unsupported information?
- Answer relevancy: does the answer actually address the question asked?
- Context precision: what fraction of the retrieved chunks were
  actually relevant, rather than noise the retriever pulled in?

Usage:
    python eval/run_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.agent import build_agent
from retrieval.build_index import get_model, get_client, retrieve_chunks

API_KEY = os.environ.get("GROQ_API_KEY", "")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

TEST_CASES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "regression_cases.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

FAITHFULNESS_PROMPT = """You are evaluating an AI assistant's answer for FAITHFULNESS to its \
source context -- whether every factual claim in the answer is actually supported by the \
context, with no hallucinated or invented information.

Respond ONLY with valid JSON:
{{"faithfulness_score": 0.0 to 1.0, "unsupported_claims": ["list any claims not backed by context"]}}

CONTEXT:
{context}

ANSWER TO EVALUATE:
{answer}
"""

RELEVANCY_PROMPT = """You are evaluating how RELEVANT an AI assistant's answer is to the \
question asked -- whether it actually addresses what was asked, without padding or going off-topic.

Respond ONLY with valid JSON:
{{"relevancy_score": 0.0 to 1.0, "reason": "one short sentence"}}

QUESTION: {question}

ANSWER: {answer}
"""

CONTEXT_PRECISION_PROMPT = """You are evaluating retrieval quality. Below is a question and a \
chunk of text that was retrieved as potentially relevant context for answering it. Judge whether \
this specific chunk is actually relevant and useful for answering the question.

Respond ONLY with valid JSON:
{{"relevant": true or false}}

QUESTION: {question}

RETRIEVED CHUNK:
{chunk}
"""


def call_judge(prompt: str, max_tokens: int = 600, retries: int = 3) -> dict | None:
    """Call Groq and parse a JSON object out of the response, or return None on failure.

    Retries with backoff on 429 (rate limit) rather than failing the whole
    evaluation run -- this harness fires several calls per question in
    quick succession and free-tier rate limits are easy to hit.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,  # judge prompts include a lot of context; 300 was
                                   # too tight for this reasoning model and left
                                   # some responses empty/truncated before valid JSON.
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(retries):
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            wait = 8 * (attempt + 1)
            print(f"    Rate limited by Groq, waiting {wait}s before retry "
                  f"({attempt + 1}/{retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""

        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            result = json.loads(content[start:end + 1])
            time.sleep(1.5)  # space out calls even on success, to stay under rate limits
            return result
        except json.JSONDecodeError:
            return None

    print("    Giving up after repeated rate limiting.")
    return None


def evaluate_question(question: str) -> dict:
    model = get_model()
    client = get_client()

    chunks = retrieve_chunks(question, top_k=5, model=model, client=client)
    context = "\n\n---\n\n".join(c["text"] for c in chunks)

    agent = build_agent()
    result = agent.invoke({
        "question": question, "needs_retrieval": False,
        "answer": "", "sources": [], "from_cache": False,
    })
    answer = result["answer"]

    faith = call_judge(FAITHFULNESS_PROMPT.format(context=context[:6000], answer=answer))
    rel = call_judge(RELEVANCY_PROMPT.format(question=question, answer=answer))

    relevant_flags = []
    for c in chunks:
        verdict = call_judge(CONTEXT_PRECISION_PROMPT.format(question=question, chunk=c["text"][:1000]))
        relevant_flags.append(bool(verdict.get("relevant", False)) if verdict else False)
    context_precision = sum(relevant_flags) / len(relevant_flags) if relevant_flags else None

    return {
        "question": question,
        "answer": answer,
        "faithfulness_score": faith.get("faithfulness_score") if faith else None,
        "unsupported_claims": faith.get("unsupported_claims", []) if faith else [],
        "relevancy_score": rel.get("relevancy_score") if rel else None,
        "context_precision": context_precision,
        "used_retrieval": result["needs_retrieval"],
    }


def run_eval():
    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        if not case["must_contain_any"] or "SEC" in case["must_contain_any"]:
            continue  # skip the non-retrieval greeting case, nothing to evaluate retrieval on
        print(f"Evaluating: {case['question']}")
        r = evaluate_question(case["question"])
        results.append(r)
        print(f"  faithfulness={r['faithfulness_score']}, relevancy={r['relevancy_score']}, "
              f"context_precision={r['context_precision']}")
        if r["unsupported_claims"]:
            print(f"  Unsupported claims flagged: {r['unsupported_claims']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    def avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print(f"  Avg faithfulness:      {avg('faithfulness_score')}")
    print(f"  Avg answer relevancy:  {avg('relevancy_score')}")
    print(f"  Avg context precision: {avg('context_precision')}")
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Run: export $(cat .env | xargs)")
    run_eval()

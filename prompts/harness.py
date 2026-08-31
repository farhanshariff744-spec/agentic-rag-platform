"""
Prompt Regression Harness
==========================
Re-runs a fixed set of test questions through the agent and checks that
each answer contains at least one of the expected key facts/phrases.

Run this after changing any prompt in prompts/prompts.py to catch
regressions -- e.g. if a reworded ANSWER_PROMPT causes the model to stop
citing filing dates, or a reworded ROUTER_PROMPT starts misrouting
questions that need retrieval into the direct-answer path.

Usage:
    python prompts/harness.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.agent import build_agent
from prompts.prompt_templates import PROMPT_VERSION

TEST_CASES_PATH = Path(__file__).parent / "regression_cases.json"


def run_harness() -> bool:
    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)

    agent = build_agent()
    results = []

    print(f"Running {len(cases)} regression cases against prompt version '{PROMPT_VERSION}'\n")

    for case in cases:
        result = agent.invoke({
            "question": case["question"], "needs_retrieval": False,
            "answer": "", "sources": [], "from_cache": False,
        })
        answer_lower = result["answer"].lower()
        matched = [kw for kw in case["must_contain_any"] if kw.lower() in answer_lower]
        passed = len(matched) > 0

        results.append({"id": case["id"], "passed": passed})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case['id']}: \"{case['question']}\"")
        if not passed:
            print(f"    Expected one of: {case['must_contain_any']}")
            print(f"    Got: {result['answer'][:200]}...")
        print()

    n_passed = sum(r["passed"] for r in results)
    print(f"{n_passed}/{len(results)} passed (prompt version: {PROMPT_VERSION})")

    return n_passed == len(results)


if __name__ == "__main__":
    success = run_harness()
    sys.exit(0 if success else 1)

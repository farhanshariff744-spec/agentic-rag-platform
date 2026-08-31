"""
API Layer
=========
Exposes the agent over HTTP using FastAPI, so it can be called from a
browser, curl, Postman, or a frontend -- not just the command line.

The agent (and everything it depends on -- the embedding model, the
Qdrant client) is built ONCE at startup and reused across requests,
rather than rebuilt per-request, which would be extremely slow and
would also risk the same Qdrant file-lock collision fixed earlier.

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Then, from another terminal or the browser:
    curl -X POST http://localhost:8000/ask \\
         -H "Content-Type: application/json" \\
         -d '{"question": "What are Apple'"'"'s main business risks?"}'
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.agent import build_agent

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not set. Export it before starting the server: "
            "export $(cat .env | xargs)"
        )
    print("Building agent (loading embedding model, connecting to Qdrant)...")
    _agent = build_agent()
    print("Agent ready.")
    yield
    


app = FastAPI(
    title="Agentic RAG Platform API",
    description="Ask questions about companies using their SEC filings.",
    version="0.1.0",
    lifespan=lifespan,
)



class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    ticker: str
    form: str
    filing_date: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    used_retrieval: bool
    from_cache: bool



@app.get("/health")
def health():
    """Simple liveness check -- useful for deployment platforms to verify the service is up."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    result = _agent.invoke({
        "question": request.question,
        "needs_retrieval": False,
        "answer": "",
        "sources": [],
        "from_cache": False,
    })

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        used_retrieval=result["needs_retrieval"],
        from_cache=result["from_cache"],
    )

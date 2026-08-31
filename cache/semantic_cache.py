"""
Semantic Cache
==============
Before answering a question, checks whether a sufficiently SIMILAR
question (not just an exact match) has already been answered, using
the same embedding model as retrieval. If a close match is found, the
cached answer is returned instantly with no LLM call at all.

Uses the same local Qdrant instance as retrieval, in a separate
collection, so no new infrastructure or account is needed.

Usage (as a library -- see inference/answer.py for integration):
    from cache.semantic_cache import check_cache, store_in_cache
"""

import uuid
import sys
from pathlib import Path

import numpy as np
from qdrant_client.models import Distance, VectorParams, PointStruct

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.build_index import embed_one

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_COLLECTION = "answer_cache"
EMBEDDING_DIM = 384  # must match the embedding model used in retrieval
# Cosine similarity threshold above which a past question counts as "the
# same question" for caching purposes. 1.0 = identical. This is a judgment
# call -- too low returns stale/wrong answers for different questions, too
# high means the cache almost never hits. 0.95 is a reasonably strict start.
SIMILARITY_THRESHOLD = 0.95


def ensure_cache_collection(client):
    """Create the cache collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if CACHE_COLLECTION not in existing:
        client.create_collection(
            collection_name=CACHE_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def check_cache(question: str, model, client, threshold: float = SIMILARITY_THRESHOLD) -> dict | None:
    """Return a cached result if a sufficiently similar question exists, else None."""
    ensure_cache_collection(client)

    query_vector = np.array(embed_one(model, question))  # keep as numpy array for the manual check below
    response = client.query_points(
        collection_name=CACHE_COLLECTION,
        query=query_vector.tolist(),
        limit=1,
        with_vectors=True,  # need the raw vector back to verify the score ourselves
    )

    if not response.points:
        return None

    best = response.points[0]

    # Qdrant's own returned score has proven unreliable for this collection --
    # it gets reopened across many separate short-lived CLI processes, and its
    # local/embedded mode has known rough edges with that access pattern on
    # small collections. Recompute real cosine similarity directly instead of
    # trusting the score field, the same way debug_cache.py verified it.
    cached_vector = np.array(best.vector)
    similarity = float(
        np.dot(query_vector, cached_vector)
        / (np.linalg.norm(query_vector) * np.linalg.norm(cached_vector))
    )

    if similarity < threshold:
        return None  # closest match isn't close enough to count as a hit

    return {
        "answer": best.payload["answer"],
        "sources": best.payload["sources"],
        "cached_question": best.payload["question"],
        "similarity": similarity,
    }


def store_in_cache(question: str, answer: str, sources: list[dict], model, client):
    """Store a question/answer pair in the cache for future lookups."""
    ensure_cache_collection(client)

    query_vector = embed_one(model, question)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=query_vector,
        payload={
            "question": question,
            "answer": answer,
            "sources": sources,
        },
    )
    client.upsert(collection_name=CACHE_COLLECTION, points=[point])

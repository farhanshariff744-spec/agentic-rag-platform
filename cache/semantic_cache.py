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
from pathlib import Path

import numpy as np
from qdrant_client.models import Distance, VectorParams, PointStruct

CACHE_COLLECTION = "answer_cache"
EMBEDDING_DIM = 384  
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

    query_vector = model.encode(question)  
    response = client.query_points(
        collection_name=CACHE_COLLECTION,
        query=query_vector.tolist(),
        limit=1,
        with_vectors=True,  
    )

    if not response.points:
        return None

    best = response.points[0]

    cached_vector = np.array(best.vector)
    similarity = float(
        np.dot(query_vector, cached_vector)
        / (np.linalg.norm(query_vector) * np.linalg.norm(cached_vector))
    )

    if similarity < threshold:
        return None  

    return {
        "answer": best.payload["answer"],
        "sources": best.payload["sources"],
        "cached_question": best.payload["question"],
        "similarity": similarity,
    }


def store_in_cache(question: str, answer: str, sources: list[dict], model, client):
    """Store a question/answer pair in the cache for future lookups."""
    ensure_cache_collection(client)

    query_vector = model.encode(question).tolist()
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

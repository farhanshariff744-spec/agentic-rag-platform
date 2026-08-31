"""
Diagnostic: dump everything in the semantic cache to check for bugs.

Usage:
    python cache/debug_cache.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.build_index import get_model, get_client
from cache.semantic_cache import CACHE_COLLECTION

client = get_client()
model = get_model()

existing = [c.name for c in client.get_collections().collections]
if CACHE_COLLECTION not in existing:
    print(f"No '{CACHE_COLLECTION}' collection exists yet.")
    sys.exit(0)

count = client.count(CACHE_COLLECTION).count
print(f"Cache collection has {count} point(s).\n")

points, _ = client.scroll(collection_name=CACHE_COLLECTION, limit=100, with_vectors=True)
for p in points:
    print(f"ID: {p.id}")
    print(f"  Question: {p.payload.get('question')}")
    print(f"  Vector (first 5 dims): {p.vector[:5]}")
    print(f"  Vector norm: {np.linalg.norm(p.vector):.4f}")
    print()

if len(points) >= 1:
    q1 = "What are Apple's main business risks?"
    q2 = "What was Apple's revenue growth like?"
    v1 = np.array(model.encode(q1))
    v2 = np.array(model.encode(q2))
    manual_cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"Manually computed cosine similarity between:")
    print(f"  '{q1}'")
    print(f"  '{q2}'")
    print(f"  = {manual_cosine:.4f}")
    print("\nIf this manual number is well below 0.95 but the agent reported a cache")
    print("hit at ~1.000 for these two questions, the bug is in how the score from")
    print("Qdrant's query_points() is being read/interpreted, not in the embeddings.")

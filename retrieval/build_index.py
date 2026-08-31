"""
Retrieval Pipeline (Embedding + Vector Search)
================================================
Embeds the chunks produced by ingestion using a free local model
(sentence-transformers, no API key needed), stores them in a LOCAL
Qdrant instance (embedded mode -- no server, no account, no signup),
and provides a search function to query them.

Usage:
    # Build the index from a chunks file produced by ingestion:
    python retrieval/build_index.py --input data/processed/AAPL_10K_chunks.json

    # Then run a test search:
    python retrieval/build_index.py --query "What are Apple's main risks?"
"""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, free, runs on CPU fine
EMBEDDING_DIM = 384                   # this model's output vector size
QDRANT_PATH = "data/qdrant_db"        # local on-disk store, no server needed
COLLECTION_NAME = "sec_filings"

_client_instance = None
_model_instance = None

def get_client() -> QdrantClient:
    global _client_instance
    if _client_instance is None:
        Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        _client_instance = QdrantClient(path=QDRANT_PATH)
    return _client_instance


def get_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        print("Loading embedding model (first run downloads ~90MB)...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _model_instance


def ensure_collection(client: QdrantClient):
    """Create the collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, adding to it")


def build_index(input_path: str):
    with open(input_path) as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {input_path}")

    model = get_model()
    client = get_client()
    ensure_collection(client)

    print("Embedding chunks (this may take a minute on CPU)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    points = []
    # Use a running counter as the point ID, offset so re-running on a
    # different file doesn't collide with existing IDs in the collection.
    existing_count = client.count(COLLECTION_NAME).count
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=existing_count + i,
            vector=vector.tolist(),
            payload={
                "text": chunk["text"],
                "ticker": chunk["ticker"],
                "form": chunk["form"],
                "filing_date": chunk["filing_date"],
                "chunk_index": chunk["chunk_index"],
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant at {QDRANT_PATH}")


def retrieve_chunks(query: str, top_k: int = 5, model=None, client=None) -> list[dict]:
    """Return top-k chunks for a query as plain dicts, with no printing.

    Other modules (like the inference layer) should call this directly
    rather than the CLI-oriented search() below. Pass in an existing
    model/client to avoid reloading them on every call.
    """
    if model is None:
        model = get_model()
    if client is None:
        client = get_client()

    query_vector = model.encode(query).tolist()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return [
        {
            "text": hit.payload["text"],
            "ticker": hit.payload["ticker"],
            "form": hit.payload["form"],
            "filing_date": hit.payload["filing_date"],
            "chunk_index": hit.payload["chunk_index"],
            "score": hit.score,
        }
        for hit in response.points
    ]


def search(query: str, top_k: int = 5):
    model = get_model()
    client = get_client()
    results = retrieve_chunks(query, top_k=top_k, model=model, client=client)

    print(f"\nTop {len(results)} results for: '{query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [score={r['score']:.3f}] {r['ticker']} {r['form']} "
              f"({r['filing_date']}, chunk {r['chunk_index']})")
        print(f"   {r['text'][:200]}...\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and query the retrieval index")
    parser.add_argument("--input", help="Path to a chunks JSON file to index")
    parser.add_argument("--query", help="A test query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    if args.input:
        build_index(args.input)
    if args.query:
        search(args.query, args.top_k)
    if not args.input and not args.query:
        parser.error("Provide --input to build the index, --query to search, or both")

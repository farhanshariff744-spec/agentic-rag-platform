"""
One-off: clear the answer cache collection only, leaving the filings
index (sec_filings) untouched. Run this after a prompt/token change
that would make previously-cached answers stale.

Usage:
    python cache/clear_cache.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.build_index import get_client
from cache.semantic_cache import CACHE_COLLECTION

client = get_client()
existing = [c.name for c in client.get_collections().collections]

if CACHE_COLLECTION in existing:
    client.delete_collection(CACHE_COLLECTION)
    print(f"Deleted '{CACHE_COLLECTION}' collection -- cache is now empty.")
else:
    print(f"'{CACHE_COLLECTION}' collection doesn't exist -- nothing to clear.")

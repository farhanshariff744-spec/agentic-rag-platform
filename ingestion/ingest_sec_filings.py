"""
SEC EDGAR Ingestion Pipeline
============================
Pulls recent filings (10-K, 10-Q) for a given company from SEC EDGAR,
strips HTML down to clean text, and chunks it for embedding.

SEC EDGAR is completely free and requires no API key — but it DOES
require a descriptive User-Agent header with a real contact (SEC will
rate-limit or block generic/missing ones). Replace the placeholder
below with your own name/email before running.

Usage:
    python ingestion/ingest_sec_filings.py AAPL --filing-type 10-K --count 3
"""

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
USER_AGENT = "Agentic RAG Platform farhanshariff744@gmail.com"  # <-- replace this
HEADERS = {"User-Agent": USER_AGENT}
TICKER_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
OUTPUT_DIR = Path("data/processed")
CHUNK_SIZE = 1000      # characters per chunk (rough proxy for tokens)
CHUNK_OVERLAP = 150    # characters of overlap between chunks


def get_cik_for_ticker(ticker: str) -> str:
    """Look up a company CIK (10-digit, zero-padded) from its ticker symbol."""
    resp = requests.get(TICKER_LOOKUP_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in SEC ticker list")


def get_recent_filings(cik: str, filing_type: str, count: int) -> list[dict]:
    """Return metadata for the most recent filings of a given type."""
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=15)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]

    matches = []
    for i, form in enumerate(recent["form"]):
        if form == filing_type:
            matches.append({
                "accession_number": recent["accessionNumber"][i].replace("-", ""),
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
                "form": form,
            })
        if len(matches) >= count:
            break
    return matches


def strip_html_tags(raw_html: str) -> str:
    """Strip HTML/XML tags with plain regex instead of a DOM parser.

    BeautifulSoup + lxml can hang or misbehave on iXBRL filings, which mix
    XML and HTML with unusual deeply-nested inline tagging. Regex-based
    stripping is far lighter and has no DOM to build, so it can't hang the
    way a C-based parser occasionally can on malformed/pathological markup.
    It's less "correct" than a real parser, but for extracting readable
    body text out of a filing it's more than good enough.
    """
    # Drop script/style blocks entirely, tags and content included.
    raw_html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html,
                       flags=re.DOTALL | re.IGNORECASE)
    # Strip every remaining tag, leaving just the text between them.
    text = re.sub(r"<[^>]+>", " ", raw_html)
    # Unescape HTML entities like &amp; &nbsp; &#39; etc.
    text = html.unescape(text)
    # Collapse all whitespace runs to single spaces.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def download_filing_text(cik: str, filing: dict) -> str:
    """Download a filing primary document and strip it down to plain text."""
    cik_no_padding = str(int(cik))
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik_no_padding}/"
           f"{filing['accession_number']}/{filing['primary_document']}")

    RAW_BYTE_CAP = 3_000_000  # ~3MB of raw HTML is far more than enough content
    raw_bytes = bytearray()
    print(f"    Connecting to {url[:80]}...", flush=True)
    with requests.get(url, headers=HEADERS, timeout=30, stream=True) as resp:
        resp.raise_for_status()
        print(f"    Connected, status {resp.status_code}, streaming...", flush=True)
        chunk_count = 0
        for chunk in resp.iter_content(chunk_size=65536):
            raw_bytes.extend(chunk)
            chunk_count += 1
            if chunk_count % 5 == 0:
                print(f"    ...{len(raw_bytes) / 1_000_000:.1f}MB downloaded", flush=True)
            if len(raw_bytes) >= RAW_BYTE_CAP:
                break
    print(f"    Download complete: {len(raw_bytes) / 1_000_000:.1f}MB total", flush=True)

    raw_html = raw_bytes.decode("utf-8", errors="ignore")
    del raw_bytes

    print("    Stripping tags...", flush=True)
    text = strip_html_tags(raw_html)
    print(f"    Done, {len(text)} chars of text extracted", flush=True)

    del raw_html
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking on sentence boundaries where possible."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            search_from = start + chunk_size // 2
            last_period = text.rfind(". ", search_from, end)
            if last_period != -1:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start:  # safety net: force forward progress no matter what
            next_start = end
        start = next_start
    return chunks


def ingest(ticker: str, filing_type: str, count: int):
    print(f"Looking up CIK for {ticker}...")
    cik = get_cik_for_ticker(ticker)
    print(f"  CIK: {cik}")

    print(f"Fetching {count} most recent {filing_type} filings...")
    filings = get_recent_filings(cik, filing_type, count)
    print(f"  Found {len(filings)} filings")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for filing in filings:
        print(f"  Downloading {filing['form']} filed {filing['filing_date']}...")
        text = download_filing_text(cik, filing)
        print(f"    Chunking {len(text)} chars...", flush=True)
        chunks = chunk_text(text)
        print(f"    -> {len(chunks)} chunks ({len(text)} chars)")

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "ticker": ticker.upper(),
                "form": filing["form"],
                "filing_date": filing["filing_date"],
                "chunk_index": i,
                "text": chunk,
            })

        time.sleep(0.2)  # be polite to SEC servers -- they do rate-limit

    out_path = OUTPUT_DIR / f"{ticker.upper()}_{filing_type.replace('-', '')}_chunks.json"
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved {len(all_chunks)} chunks to {out_path}")
    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings for a ticker")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--filing-type", default="10-K", help="Filing type (10-K, 10-Q, etc.)")
    parser.add_argument("--count", type=int, default=3, help="Number of filings to fetch")
    args = parser.parse_args()

    ingest(args.ticker, args.filing_type, args.count)

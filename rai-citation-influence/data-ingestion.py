import os
import re
import time
import requests
import pandas as pd
from typing import List, Dict, Optional

# ============================================================
# CONFIG
# ============================================================

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"

BIBTEX_DIR = "./bibtex-proceedings-2018-2023"
OUTPUT_DIR = "./output"
OUTPUT_CSV = f"{OUTPUT_DIR}/aies_papers.csv"
FAILED_DOI_CSV = f"{OUTPUT_DIR}/failed_dois.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "AIES-ResearchBot/1.0 (academic research; contact: msjo242@uky.edu)"
}
if SEMANTIC_SCHOLAR_API_KEY:
    HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

# --- Rate limiting ---
REQUESTS_PER_SECOND = 1.0
REQUEST_DELAY = 1.0 / REQUESTS_PER_SECOND

# ============================================================
# 1️⃣ DOI RANGES FOR 2024–2025
# ============================================================

DOI_RANGES = [
    # 2025
    ("10.1609/aies.v8i1.36526", "10.1609/aies.v8i1.36606"),
    ("10.1609/aies.v8i2.36607", "10.1609/aies.v8i2.36691"),
    ("10.1609/aies.v8i3.36692", "10.1609/aies.v8i3.36803"),
    # 2024
    ("10.1609/aies.v7i2.31892", "10.1609/aies.v7i2.31911"),
]

# ============================================================
# REGEX
# ============================================================

DOI_REGEX = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

# ============================================================
# STEP 1 — DOI RANGE EXPANSION
# ============================================================

def expand_doi_range(start: str, end: str) -> List[str]:
    prefix, start_num = start.rsplit(".", 1)
    _, end_num = end.rsplit(".", 1)
    return [f"{prefix}.{i}" for i in range(int(start_num), int(end_num) + 1)]

def collect_range_dois() -> List[str]:
    all_dois = []
    for start, end in DOI_RANGES:
        all_dois.extend(expand_doi_range(start, end))
    return all_dois

# ============================================================
# STEP 2 — EXTRACT DOIs FROM BIBTEX FILES
# ============================================================

def extract_dois_from_bibtex_file(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return DOI_REGEX.findall(text)

def collect_bibtex_dois(bibtex_dir: str) -> List[str]:
    all_dois = []
    for filename in os.listdir(bibtex_dir):
        if not filename.lower().endswith(".bib"):
            continue
        path = os.path.join(bibtex_dir, filename)
        dois = extract_dois_from_bibtex_file(path)
        print(f"[INFO] {filename}: found {len(dois)} DOIs")
        all_dois.extend(dois)
    return all_dois

# ============================================================
# STEP 3 — SEMANTIC SCHOLAR DOI LOOKUPS
# ============================================================

def semantic_scholar_request_by_doi(doi: str) -> Optional[dict]:
    """
    Query Semantic Scholar by DOI with:
    - 1 req/sec steady-state rate limiting
    - On 429: hard pause for 30 seconds, then retry same DOI
    - Resume cleanly from the failed paper
    """
    url = SEMANTIC_SCHOLAR_API_URL + doi
    params = {
        "fields": "title,authors,year,venue,externalIds,citationCount,url,paperId"
    }

    while True:
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)

            if resp.status_code == 429:
                print(f"[WARN] Rate limit hit for {doi}. Pausing 30s and retrying...")
                time.sleep(30)
                continue

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] DOI lookup failed for {doi}: {e}")
            return None

# ============================================================
# PIPELINE
# ============================================================

def run_pipeline():
    print("\n========== AIES DOI INGESTION PIPELINE ==========\n")

    # ---- Step 1: Collect DOIs from ranges (2024–2025) ----
    print("[INFO] Collecting DOIs from ranges (2024–2025)...")
    range_dois = collect_range_dois()
    print(f"[INFO] Collected {len(range_dois)} DOIs from ranges.")

    # ---- Step 2: Collect DOIs from BibTeX files (2018–2023) ----
    print("\n[INFO] Collecting DOIs from BibTeX files (2018–2023)...")
    bibtex_dois = collect_bibtex_dois(BIBTEX_DIR)
    print(f"[INFO] Collected {len(bibtex_dois)} DOIs from BibTeX.")

    # ---- Combine & dedupe ----
    all_dois = sorted(set(range_dois + bibtex_dois))
    print(f"\n[INFO] Total unique DOIs collected: {len(all_dois)}")

    # ---- Step 3: Query Semantic Scholar ----
    print("\n[INFO] Querying Semantic Scholar by DOI...\n")

    records = []
    failed_dois = []

    for i, doi in enumerate(all_dois, 1):
        print(f"[{i}/{len(all_dois)}] {doi}")
        result = semantic_scholar_request_by_doi(doi)
        if not result:
            failed_dois.append({"doi": doi})
            continue

        records.append({
            "title": result.get("title"),
            "authors": ", ".join(a["name"] for a in result.get("authors", [])),
            "year": result.get("year"),
            "venue": result.get("venue"),
            "doi": result.get("externalIds", {}).get("DOI"),
            "url": result.get("url"),
            "citation_count": result.get("citationCount"),
            "semantic_scholar_id": result.get("paperId"),
        })

    # ---- Write outputs ----
    df = pd.DataFrame(records).sort_values(by=["year", "doi"])
    df.to_csv(OUTPUT_CSV, index=False)

    failed_df = pd.DataFrame(failed_dois)
    failed_df.to_csv(FAILED_DOI_CSV, index=False)

    print("\n==================== SUMMARY ====================")
    print(f"Total DOIs processed: {len(all_dois)}")
    print(f"Papers found: {len(records)}")
    print(f"DOIs not found in Semantic Scholar: {len(failed_dois)}")
    print(f"Main output written to: {OUTPUT_CSV}")
    print(f"Failed DOI log written to: {FAILED_DOI_CSV}")
    print("=================================================\n")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_pipeline()
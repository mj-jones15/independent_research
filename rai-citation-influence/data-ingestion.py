import os
import re
import time
import requests
import pandas as pd
import glob
import json
from urllib.parse import quote
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# Title search endpoint (primary lookup method)
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
# DOI querying endpoint (kept for fallback)
# SEMANTIC_SCHOLAR_DOI_API_URL = "https://api.semanticscholar.org/graph/v1/paper/DOI:"

# Years of NeurIPS papers to ingest directly from Semantic Scholar
NEURIPS_YEARS = list(range(1987, 2025))

OUTPUT_DIR = "./data"
OUTPUT_CSV = f"{OUTPUT_DIR}/neurips_papers.csv"
FAILED_DOI_CSV = f"{OUTPUT_DIR}/failed_neurips_dois.csv"
ENRICHED_CSV = f"{OUTPUT_DIR}/neurips_papers_enriched.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Neurips-ResearchBot/1.0 (academic research; contact: msjo242@uky.edu)"
}
if SEMANTIC_SCHOLAR_API_KEY:
    HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

# --- Rate limiting ---
REQUEST_DELAY = 25  # fixed delay between requests

def enrich_and_save_results(new_df, failed_dois, all_urls, records):
    """
    Safely enrich citation statistics and write outputs.
    Never crashes even if dataframe is empty or malformed.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensure required columns exist
    required_cols = ["title","authors","year","venue","doi","url",
                     "citation_count","semantic_scholar_id","abstract"]

    for col in required_cols:
        if col not in new_df.columns:
            new_df[col] = None

    # Convert numeric fields safely
    new_df["citation_count"] = pd.to_numeric(new_df["citation_count"], errors="coerce").fillna(0)
    new_df["year"] = pd.to_numeric(new_df["year"], errors="coerce")

    # If dataframe is empty, still save raw output
    if new_df.empty:
        print("[WARN] No records returned. Saving empty outputs.")

        new_df.to_csv(OUTPUT_CSV, index=False)
        new_df.to_csv(ENRICHED_CSV, index=False)

        failed_df = pd.DataFrame(failed_dois)
        failed_df.to_csv(FAILED_DOI_CSV, index=False)

        return

    # ===============================
    # ENRICHMENT STATISTICS
    # ===============================

    global_mean = new_df["citation_count"].mean()
    global_std = new_df["citation_count"].std()

    new_df["year_mean"] = new_df.groupby("year")["citation_count"].transform("mean")
    new_df["year_std"] = new_df.groupby("year")["citation_count"].transform("std").replace(0, pd.NA)

    new_df["year_zscore"] = (new_df["citation_count"] - new_df["year_mean"]) / new_df["year_std"]

    new_df["global_z"] = (new_df["citation_count"] - global_mean) / global_std

    percentiles = [95,85,70,50]
    thresholds = {p: new_df["citation_count"].quantile(p/100) for p in percentiles}

    def percentile_bucket(c):
        if c >= thresholds[95]:
            return "Top 5%"
        elif c >= thresholds[85]:
            return "Top 15%"
        elif c >= thresholds[70]:
            return "Top 30%"
        elif c >= thresholds[50]:
            return "Top 50%"
        return "Bottom 50%"

    new_df["percentile_bucket"] = new_df["citation_count"].apply(percentile_bucket)

    new_df.sort_values(by=["year","citation_count"], ascending=[True,False], inplace=True)

    # ===============================
    # SAVE RESULTS
    # ===============================

    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        combined_df.drop_duplicates(subset=["doi"], keep="last", inplace=True)

        combined_df.sort_values(by=["year","doi"], inplace=True)

        combined_df.to_csv(OUTPUT_CSV, index=False)
        combined_df.to_csv(ENRICHED_CSV, index=False)

    else:
        new_df.to_csv(OUTPUT_CSV, index=False)
        new_df.to_csv(ENRICHED_CSV, index=False)

    # Save failed lookups
    failed_df = pd.DataFrame(failed_dois)
    failed_df.to_csv(FAILED_DOI_CSV, index=False)

    print("\n==================== SUMMARY ====================")
    print(f"Total URLs processed: {len(all_urls)}")
    print(f"Papers found: {len(records)}")
    print(f"Failed lookups: {len(failed_dois)}")
    print(f"Main output: {OUTPUT_CSV}")
    print(f"Enriched output: {ENRICHED_CSV}")
    print(f"Failed log: {FAILED_DOI_CSV}")
    print("=================================================\n")

# ============================================================
# 1️⃣ DOI RANGES FOR 2024–2025 - AIES Papers Only
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


URL_REGEX = re.compile(r"https?://[^\s\}]+", re.I)


# ============================================================
# NEW: BIBTEX PARSING (lightweight)
# ============================================================

def split_bibtex_entries(content: str) -> List[str]:
    """
    Robustly split a BibTeX file into entries, handling multi-line and nested braces.
    Returns a list of entry strings.
    """
    entries = []
    pos = 0
    length = len(content)
    while pos < length:
        # Find the next entry
        m = re.search(r'@(\w+)\s*\{', content[pos:], re.I)
        if not m:
            break
        entry_start = pos + m.start()
        brace_open = content.find('{', entry_start)
        if brace_open == -1:
            break
        brace_count = 1
        i = brace_open + 1
        while i < length and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1
        entry_end = i
        entries.append(content[entry_start:entry_end])
        pos = entry_end
    return entries

def parse_bibtex_files(bib_dir: str) -> List[Dict]:
    """
    Parse all .bib files in bib_dir robustly, extracting all entries.
    Uses brace counting to extract all entries, including multi-line and nested braces.
    """
    records = []
    for filepath in glob.glob(os.path.join(bib_dir, "*.bib")):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        entries = split_bibtex_entries(content)
        for entry in entries:
            def extract(field):
                m = re.search(rf"(?<!\w){field}\s*=\s*\{{((?:[^{{}}]|{{[^{{}}]*}})*)\}}", entry, re.I|re.S)
                if m:
                    return m.group(1).strip()
                m = re.search(rf"(?<!\w){field}\s*=\s*\"([^\"]*)\"", entry, re.I|re.S)
                if m:
                    return m.group(1).strip()
                return None
            title = extract("title")
            authors = extract("author")
            year = extract("year")
            # Only filter out entries with missing title
            if not title:
                continue
            records.append({
                "title": title.strip(),
                "authors": authors.strip() if authors else None,
                "year": year.strip() if year else None,
                "venue": "NeurIPS",
                "doi": extract("doi")
            })
    return records

# ============================================================
# NEW: OPENALEX QUERY
# ============================================================

def extract_abstract(paper):
    # OpenAlex: reconstruct from inverted index or use direct abstract
    if "abstract" in paper and paper["abstract"]:
        return paper["abstract"]
    inv_idx = paper.get("abstract_inverted_index")
    if not inv_idx:
        return None
    words = []
    for word, positions in inv_idx.items():
        for pos in positions:
            words.append((pos, word))
    return " ".join(w for _, w in sorted(words))


def query_openalex(title: str) -> Optional[dict]:
    url = f"https://api.openalex.org/works?search={quote(title)}&per-page=1"
    retry = 0
    rate_limit_hits = []
    while retry < 3:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                now = time.time()
                rate_limit_hits.append(now)

                # Keep only last 2 minutes
                rate_limit_hits = [t for t in rate_limit_hits if now - t < 120]

                if len(rate_limit_hits) >= 3:
                    print("[OpenAlex] 3 rate limits in 2 minutes → sleeping 10 minutes...")
                    time.sleep(600)
                    rate_limit_hits.clear()
                else:
                    print("[OpenAlex] Rate limit hit → sleeping 25s...")
                    time.sleep(25)

                retry += 1
                continue
            r.raise_for_status()
            data = r.json()
            if not data.get("results"):
                return None
            paper = data["results"][0]
            return {
                "abstract": extract_abstract(paper),
                "doi": paper.get("doi"),
                "url": paper.get("id"),
                "citation_count": paper.get("cited_by_count")
            }
        except Exception as e:
            print(f"[OpenAlex] Query failed for '{title}': {e}")
            retry += 1
            time.sleep(REQUEST_DELAY)
    return None

# ============================================================
# NEW: ARXIV QUERY
# ============================================================

def query_arxiv(title: str) -> Optional[str]:
    url = f"http://export.arxiv.org/api/query?search_query=ti:{quote(title)}&max_results=1"
    retry = 0
    while retry < 3:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 429:
                print("[arXiv] Rate limit hit, sleeping 10s...")
                time.sleep(10)
                retry += 1
                continue
            r.raise_for_status()
            text = r.text
            m = re.search(r"<summary>(.*?)</summary>", text, re.S)
            return m.group(1).strip().replace("\n", " ") if m else None
        except Exception as e:
            print(f"[arXiv] Query failed for '{title}': {e}")
            retry += 1
            time.sleep(2)
    return None


# For AIES Papers Only
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
# STEP 2 — FETCH NEURIPS PAPERS FROM SEMANTIC SCHOLAR
# ============================================================

def fetch_neurips_papers_by_year(year: int) -> List[dict]:
    """
    Fetch all NeurIPS papers for a given year directly from Semantic Scholar
    using venue + year filtering with pagination.
    """

    all_papers = []
    offset = 0

    while True:

        params = {
            "query": "Neural Information Processing Systems",
            "year": year,
            "limit": 100,
            "offset": offset,
            "fields": "title,authors,year,venue,externalIds,citationCount,url,paperId,abstract"
        }

        try:
            resp = requests.get(
                SEMANTIC_SCHOLAR_API_URL,
                headers=HEADERS,
                params=params,
                timeout=30
            )

            if resp.status_code == 429:
                print("[WARN] Rate limit hit. Sleeping 10 seconds...")
                time.sleep(10)
                continue

            resp.raise_for_status()

            data = resp.json()
            papers = data.get("data", [])

            if not papers:
                break

            all_papers.extend(papers)

            offset += 100
            time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed fetching year {year}: {e}")
            break

    return all_papers

# ============================================================
# STEP 3 — SEMANTIC SCHOLAR URL LOOKUPS
# ============================================================

#
# def semantic_scholar_request_by_title(title: str) -> Optional[dict]:
#     """
#     Query Semantic Scholar using paper title search.
#     """
#
#     params = {
#         "query": title,
#         "limit": 1,
#         "fields": "title,authors,year,venue,externalIds,citationCount,url,paperId,abstract"
#     }
#
#     while True:
#         try:
#             resp = requests.get(SEMANTIC_SCHOLAR_API_URL, headers=HEADERS, params=params, timeout=20)
#
#             if resp.status_code == 429:
#                 print(f"[WARN] Rate limit hit for title search. Pausing 30s...")
#                 time.sleep(30)
#                 continue
#
#             resp.raise_for_status()
#
#             data = resp.json()
#             time.sleep(REQUEST_DELAY)
#
#             if not data.get("data"):
#                 return None
#
#             return data["data"][0]
#
#         except requests.exceptions.RequestException as e:
#             print(f"[ERROR] Title lookup failed for '{title}': {e}")
#             return None

# ============================================================
# PIPELINE
# ============================================================

def run_aies_pipeline():
    print("\n========== AIES DOI INGESTION PIPELINE ==========\n")

    # ---- Step 1: Collect DOIs from ranges (2024–2025) ----
    print("[INFO] Collecting DOIs from ranges (2024–2025)...")
    range_dois = collect_range_dois()
    print(f"[INFO] Collected {len(range_dois)} DOIs from ranges.")

    print("[INFO] Fetching NeurIPS papers directly from Semantic Scholar...")

    # ---- Step 3: Query Semantic Scholar ----
    print("\n[INFO] Querying Semantic Scholar by venue/year...\n")

    records = []

    # --- Streaming save setup (prevents data loss) ---
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=[
            "title","authors","year","venue","doi","url",
            "citation_count","semantic_scholar_id","abstract"
        ]).to_csv(OUTPUT_CSV, index=False)

    failed_dois = []

    for year in NEURIPS_YEARS:

        print(f"\n[INFO] Fetching year {year}...")

        papers = fetch_neurips_papers_by_year(year)

        print(f"[INFO] Retrieved {len(papers)} papers for {year}")

        for paper in papers:

            record = {
                "title": paper.get("title"),
                "authors": ", ".join(a["name"] for a in paper.get("authors", [])),
                "year": paper.get("year"),
                "venue": paper.get("venue"),
                "doi": paper.get("externalIds", {}).get("DOI"),
                "url": paper.get("url"),
                "citation_count": paper.get("citationCount"),
                "semantic_scholar_id": paper.get("paperId"),
                "abstract": paper.get("abstract")
            }

            records.append(record)

            pd.DataFrame([record]).to_csv(
                OUTPUT_CSV,
                mode="a",
                header=False,
                index=False
            )

    # ---- Final enrichment pass on saved data ----
    if os.path.exists(OUTPUT_CSV):
        final_df = pd.read_csv(OUTPUT_CSV)
    else:
        final_df = pd.DataFrame()

    enrich_and_save_results(
        final_df,
        failed_dois,
        NEURIPS_YEARS,
        records
    )

# ============================================================
# NEW: NEURIPS BIBTEX INGESTION WITH OPENALEX + ARXIV
# ============================================================

RUN_DURATION = 60 * 60        # 1 hour
SLEEP_DURATION = 2 * 60   # 1 hours


def run_neurips_bibtex_pipeline():
    print("\n========== NEURIPS BIBTEX INGESTION ==========")

    bib_records = parse_bibtex_files("./bibtex/neurips/")
    print(f"[INFO] Parsed {len(bib_records)} BibTeX entries")

    # Ensure CSV exists with header
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=[
            "title","authors","year","venue","doi","url",
            "citation_count","semantic_scholar_id","abstract"
        ]).to_csv(OUTPUT_CSV, index=False)

    # Load existing titles
    existing_titles = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_df = pd.read_csv(OUTPUT_CSV)
            if "title" in existing_df.columns:
                existing_titles = set(existing_df["title"].dropna().str.strip())
                print(f"[INFO] Found {len(existing_titles)} existing titles. Skipping them.")
        except Exception as e:
            print(f"[WARN] Could not read existing CSV: {e}")

    failed = []
    records = []

    index = 0  # 👈 Track position across cycles

    while index < len(bib_records):
        print(f"\n[CYCLE] Starting new run at {datetime.now()}")

        start_time = time.time()
        end_time = start_time + RUN_DURATION

        while index < len(bib_records):
            # ⛔ Stop after 1 hour
            if time.time() > end_time:
                print("[CYCLE] 1 hour reached. Pausing ingestion.")
                break

            rec = bib_records[index]
            index += 1  # 👈 advance pointer

            title = rec.get("title", "").strip()
            if not title:
                continue

            if title in existing_titles:
                continue

            print(f"[INFO] Processing: {title}")

            try:
                oa = query_openalex(title)

                abstract = None
                doi = rec.get("doi")
                url = None
                citation_count = None

                if oa:
                    abstract = oa.get("abstract")
                    doi = doi or oa.get("doi")
                    url = oa.get("url")
                    citation_count = oa.get("citation_count")

                record = {
                    "title": title,
                    "authors": rec.get("authors"),
                    "year": rec.get("year"),
                    "venue": rec.get("venue"),
                    "doi": doi,
                    "url": url,
                    "citation_count": citation_count,
                    "semantic_scholar_id": None,
                    "abstract": abstract
                }

                records.append(record)
                existing_titles.add(title)

                cols = ["title","authors","year","venue","doi","url","citation_count","semantic_scholar_id","abstract"]

                pd.DataFrame([{k: record.get(k) for k in cols}]).to_csv(
                    OUTPUT_CSV,
                    mode="a",
                    header=False,
                    index=False
                )

            except Exception as e:
                print(f"[ERROR] Failed for '{title}': {e}")
                failed.append(title)

            # ⏱️ your delay (increase to 25s if desired)
            time.sleep(25)

        # 💤 Sleep AFTER each 1-hour chunk
        if index < len(bib_records):
            print(f"[CYCLE] Sleeping for 1 hour at {datetime.now()}\n")
            time.sleep(SLEEP_DURATION)

    # Final enrichment AFTER everything completes
    df = pd.read_csv(OUTPUT_CSV)
    enrich_and_save_results(df, failed, bib_records, records)


# ============================================================
# KEYWORD ENRICHMENT (via generate_embeddings module)
# ============================================================

NEURIPS_ENRICHED_CSV = "./data/neurips_papers_enriched.csv"

def run_neurips_keyword_enrichment():
    """
    Runs E5 embedding-based keyword similarity enrichment on the
    NeurIPS enriched CSV and writes results back to the same file.
    """
    try:
        from generate_embeddings import run_keyword_enrichment
    except ImportError as e:
        print(f"[ERROR] Could not import generate_embeddings: {e}")
        print("[HINT] Ensure generate_embeddings.py is in the same directory.")
        return

    if not os.path.exists(NEURIPS_ENRICHED_CSV):
        print(f"[ERROR] Enriched CSV not found at {NEURIPS_ENRICHED_CSV}.")
        print("[HINT] Run run_neurips_bibtex_pipeline() first to generate it.")
        return

    run_keyword_enrichment(NEURIPS_ENRICHED_CSV)



# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_neurips_bibtex_pipeline()
    #run_neurips_keyword_enrichment()
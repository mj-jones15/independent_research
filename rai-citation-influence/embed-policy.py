import os
import re
import pandas as pd
import numpy as np
import random # for testing
import matplotlib.pyplot as plt
from typing import List
from sentence_transformers import SentenceTransformer, util
from pdf_cleaning import load_pdf_text, clean_text

# ===================== CONFIG ============================
MODEL_NAME = "intfloat/e5-large-v2"
POLICY_DIR = "./data/policy_documents"

OUTPUT_ALL = os.path.join(POLICY_DIR, "policy_keyword_scores_all.csv")
OUTPUT_FILTERED = os.path.join(POLICY_DIR, "policy_keyword_scores_filtered.csv")

CHUNK_SIZE = 120
CHUNK_OVERLAP = 60
SIMILARITY_THRESHOLD = 0.90


# ===================== KEYWORDS =========================
KEYWORDS = [
    "artificial intelligence fairness",
    "artificial intelligence equality",
    "artificial intelligence equity",
    "artificial intelligence equitable",
    "artificial intelligence privacy",
    "artificial intelligence anonymity",
    "artificial intelligence confidentiality",
    "artificial intelligence confidential",
    "artificial intelligence explainability",
    "explainable artificial intelligence",
    "accountable artificial intelligence",
    "artificial intelligence accountability",
    "artificial intelligence transparency",
    "artificial intelligence auditability",
    "artificial intelligence governance",
    "artificial intelligence compliance",
    "artificial intelligence accountability mechanisms",
    "artificial intelligence algorithmic accountability",
    "green artificial intelligence",
    "energy-efficient artificial intelligence",
    "artificial intelligence carbon footprint",
    "artificial intelligence environmental impact",
    "eco-friendly artificial intelligence",
    "artificial intelligence energy consumption",
    "artificial intelligence green computing",
]


# ===================== STRUCTURE SPLIT =========================

def split_into_paragraphs(text: str) -> List[str]:
    # Step 1: Try true paragraph split
    paragraphs = re.split(r'\n{2,}', text)

    # Step 2: If too few, fall back to SINGLE newline splits
    if len(paragraphs) < 20:
        paragraphs = re.split(r'\n', text)

    # Step 3: Clean + filter (less aggressive)
    cleaned = []
    for p in paragraphs:
        p = p.strip()

        # Keep shorter paragraphs now (was too aggressive before)
        if len(p) > 40:
            cleaned.append(p)

    return cleaned

# ===================== CHUNKING =========================

def chunk_text(paragraphs: List[str], model: SentenceTransformer) -> List[str]:
    """
    Semantic-aware chunking:
    - Starts with paragraph units
    - Splits based on semantic shifts using embeddings
    - Maintains chunk size + overlap
    """

    chunks = []
    current_chunk = []
    current_length = 0

    # Precompute paragraph embeddings for semantic splitting
    para_inputs = [f"passage: {p}" for p in paragraphs]
    para_embeddings = model.encode(
        para_inputs,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    for i, para in enumerate(paragraphs):
        words = para.split()
        para_len = len(words)

        # Semantic boundary detection
        if i > 0:
            sim = util.cos_sim(para_embeddings[i], para_embeddings[i - 1]).item()

            # If semantic shift is large, force new chunk
            if sim < 0.8 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        # Size-based chunking
        if current_length + para_len > CHUNK_SIZE:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap_words = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
            current_chunk = overlap_words + words
            current_length = len(current_chunk)
        else:
            current_chunk.extend(words)
            current_length += para_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ===================== MAIN PIPELINE =========================

def process_documents():
    print("\n========== POLICY EMBEDDING PIPELINE ==========\n")

    print("[INFO] Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    # Encode keywords once
    print("[INFO] Encoding keywords...")
    e5_keywords = [f"query: {kw}" for kw in KEYWORDS]
    keyword_embeddings = model.encode(
        e5_keywords,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    all_results = []
    global_scores = []

    total_chunks_all_docs = 0

    for filename in os.listdir(POLICY_DIR):
        if not filename.endswith(".pdf"):
            continue

        print(f"\n----------------------------------------")
        print(f"[INFO] Processing: {filename}")

        path = os.path.join(POLICY_DIR, filename)

        raw_text = load_pdf_text(path)

        if len(raw_text.strip()) == 0:
            print("[WARN] Empty text extracted. Skipping.")
            continue

        cleaned = clean_text(raw_text)

        paragraphs = split_into_paragraphs(cleaned)
        chunks = chunk_text(paragraphs, model)

        num_chunks = len(chunks)
        total_chunks_all_docs += num_chunks

        print(f"[INFO] Paragraphs: {len(paragraphs)}")
        print(f"[INFO] Chunks generated: {num_chunks}")

        # Embed chunks
        e5_chunks = [f"passage: {c}" for c in chunks]

        chunk_embeddings = model.encode(
            e5_chunks,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        cosine_scores = util.cos_sim(chunk_embeddings, keyword_embeddings)

        doc_scores = []

        for i in range(cosine_scores.shape[0]):
            for j in range(cosine_scores.shape[1]):
                score = cosine_scores[i][j].item()

                doc_scores.append(score)
                global_scores.append(score)

                all_results.append({
                    "document": filename,
                    "chunk_index": i,
                    "keyword": KEYWORDS[j],
                    "similarity": score,
                    "chunk_preview": chunks[i]
                })

        # ===== Logging =====
        doc_scores_np = np.array(doc_scores)

        print("[STATS] Similarity Distribution:")
        print(f"  Mean: {doc_scores_np.mean():.4f}")
        print(f"  Std:  {doc_scores_np.std():.4f}")
        print(f"  Max:  {doc_scores_np.max():.4f}")
        print(f"  95th percentile: {np.percentile(doc_scores_np, 95):.4f}")

        matches = np.sum(doc_scores_np >= SIMILARITY_THRESHOLD)
        print(f"[STATS] Matches ≥ {SIMILARITY_THRESHOLD}: {matches}")

    # ===================== SAVE RESULTS =========================

    print("\n========================================")
    print("[INFO] Saving results...")

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(OUTPUT_ALL, index=False)

    df_filtered = df_all[df_all["similarity"] >= SIMILARITY_THRESHOLD]
    df_filtered.to_csv(OUTPUT_FILTERED, index=False)

    print(f"[DONE] All scores saved to: {OUTPUT_ALL}")
    print(f"[DONE] Filtered scores saved to: {OUTPUT_FILTERED}")

    # ===================== GLOBAL LOGGING =========================

    global_scores_np = np.array(global_scores)

    print("\n========== GLOBAL STATS ==========")
    print(f"Total chunks processed: {total_chunks_all_docs}")
    print(f"Total comparisons: {len(global_scores)}")

    if len(global_scores_np) > 0:
        print(f"Mean similarity: {global_scores_np.mean():.4f}")
        print(f"Std similarity:  {global_scores_np.std():.4f}")
        print(f"Max similarity:  {global_scores_np.max():.4f}")
        print(f"99th percentile: {np.percentile(global_scores_np, 99):.4f}")

        total_matches = np.sum(global_scores_np >= SIMILARITY_THRESHOLD)
        print(f"Total matches ≥ {SIMILARITY_THRESHOLD}: {total_matches}")

    print("=================================\n")

# ===================== TEST MODE =========================

def test_run(sample_size: int = 2):
    """
    Run a small-scale test on a subset of PDFs before full execution.

    What this does:
    - Processes only `sample_size` PDFs
    - Prints detailed debug info
    - Does NOT save CSVs (safe for experimentation)

    How to use:
    >>> python embed-policy.py
    Then temporarily replace process_documents() with test_run()

    What to look for:
    1. Text Extraction Quality:
        - Does printed text look readable?
        - Any weird spacing, broken words, or missing content?

    2. Paragraph Count:
        - Too low (<5)? → PDF likely flattened → bad extraction
        - Too high? → over-splitting

    3. Chunk Count:
        - Reasonable range: ~20–200 depending on doc size
        - Too many → chunking too aggressive
        - Too few → chunking too coarse

    4. Similarity Distribution:
        - Mean ~0.6–0.8 is typical
        - If mean >0.9 → something is wrong (likely duplication)
        - If mean <0.5 → embeddings not aligning

    5. High Similarity Matches:
        - Are ≥0.90 matches actually meaningful?
        - Print chunk_preview and manually inspect

    6. Semantic Splits:
        - Check if chunks break at logical topic shifts
        - If random → threshold (0.7) may need tuning

    """

    print("\n========== TEST RUN ==========\n")

    model = SentenceTransformer(MODEL_NAME)

    e5_keywords = [f"query: {kw}" for kw in KEYWORDS]
    keyword_embeddings = model.encode(
        e5_keywords,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    pdf_files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".pdf")]
    pdf_files = pdf_files[:sample_size]

    for filename in pdf_files:
        print(f"\n[TEST] Processing: {filename}")

        path = os.path.join(POLICY_DIR, filename)

        raw_text = load_pdf_text(path)
        print("\n--- RAW TEXT SAMPLE ---")
        print(raw_text[:1000])

        cleaned = clean_text(raw_text)
        paragraphs = split_into_paragraphs(cleaned)
        chunks = chunk_text(paragraphs, model)

        print(f"\n[TEST] Paragraphs: {len(paragraphs)}")
        print(f"[TEST] Chunks: {len(chunks)}")

        if len(chunks) > 0:
            print("\n--- SAMPLE CHUNK ---")
            print(chunks[0][:500])

        e5_chunks = [f"passage: {c}" for c in chunks]

        chunk_embeddings = model.encode(
            e5_chunks,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        cosine_scores = util.cos_sim(chunk_embeddings, keyword_embeddings)
        scores = cosine_scores.cpu().numpy().flatten()

        print("\n[TEST] Similarity Stats:")
        print(f"Mean: {scores.mean():.4f}")
        print(f"Std: {scores.std():.4f}")
        print(f"Max: {scores.max():.4f}")

        print("\n--- RANDOM CHUNKS ---")
        for i in random.sample(range(len(chunks)), min(3, len(chunks))):
            print(f"\n[Chunk {i}]")
            print(chunks[i][:500])

        print("\n--- TOP MATCHES ---")

        # reshape scores: (num_chunks, num_keywords)
        scores_matrix = cosine_scores.cpu().numpy()

        top_indices = np.argsort(scores_matrix.max(axis=1))[::-1][:5]

        for idx in top_indices:
            best_score = scores_matrix[idx].max()
            best_kw = KEYWORDS[scores_matrix[idx].argmax()]
            
            print(f"\nScore: {best_score:.4f} | Keyword: {best_kw}")
            print(chunks[idx][:500])

        high_matches = np.where(scores >= SIMILARITY_THRESHOLD)[0]
        print(f"[TEST] Matches ≥ {SIMILARITY_THRESHOLD}: {len(high_matches)}")

        plt.figure()
        plt.hist(scores, bins=50)
        plt.title("Similarity Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.show()

    print("\n========== END TEST ==========\n")

# ===================== RUN =========================

if __name__ == "__main__":
    process_documents()
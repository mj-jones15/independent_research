import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ===================== CONFIG ============================
MODEL_NAME = "intfloat/e5-large-v2"

NEURIPS_CSV = "./data/neurips_papers_enriched.csv"
AIES_CSV = "./data/aies_papers_enriched.csv"
OUTPUT_CSV  = "./data/final_papers_with_keyword_similarities.csv"

BATCH_SIZE = 128  # Safe for large datasets


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

# ===================== LOAD + COMBINE =========================
def load_and_combine():
    print("[INFO] Loading datasets...")

    neurips = pd.read_csv(NEURIPS_CSV)
    aies    = pd.read_csv(AIES_CSV)

    neurips["source"] = "NeurIPS"
    aies["source"]    = "AIES"

    df = pd.concat([neurips, aies], ignore_index=True)

    if "abstract" not in df.columns:
        raise ValueError("Missing abstract column")

    # Track empty abstracts BEFORE filtering
    df["abstract"] = df["abstract"].fillna("")
    df["is_empty_abstract"] = df["abstract"].str.strip() == ""

    print(f"[INFO] Combined dataset size: {len(df)}")
    print(f" [INFO]: Number of NeurIPS papers: {len(neurips)}")
    print(f" [INFO]: Number of AIES papers: {len(aies)}")
    print(f"[INFO] Empty abstracts: {df['is_empty_abstract'].sum()}")
    return df

# ===================== EMBEDDINGS =========================
def compute_embeddings(df):
    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Filter ONLY valid abstracts for embedding
    valid_df = df[~df["is_empty_abstract"]].copy()


    e5_abstracts = [f"passage: {a}" for a in valid_df["abstract"].tolist()]
    e5_keywords  = [f"query: {kw}" for kw in KEYWORDS]

    print("[INFO] Encoding abstracts in batches...")
    abstract_embeddings = model.encode(
        e5_abstracts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("[INFO] Encoding keywords...")
    keyword_embeddings = model.encode(
        e5_keywords,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("[INFO] Computing cosine similarity matrix...")
    cosine_scores = util.cos_sim(abstract_embeddings, keyword_embeddings)

    return cosine_scores, valid_df

# ===================== BUILD OUTPUT =========================
def build_output(df, cosine_scores, valid_df):
    print("[INFO] Building output dataframe...")

    cosine_np = cosine_scores.cpu().numpy()

    # Initialize all keyword columns with NaN
    for kw in KEYWORDS:
        col_name = f"kw_sim_{kw.replace(' ', '_')}"
        df[col_name] = np.nan

    # Primary keyword
    max_indices = cosine_np.argmax(axis=1)
    max_scores  = cosine_np.max(axis=1)

    df["primary_keyword"] = np.nan
    df["primary_keyword_score"] = np.nan

    # Fill only valid rows
    valid_indices = valid_df.index

    max_indices = cosine_np.argmax(axis=1)
    max_scores  = cosine_np.max(axis=1)

    df.loc[valid_indices, "primary_keyword"] = [KEYWORDS[i] for i in max_indices]
    df.loc[valid_indices, "primary_keyword_score"] = max_scores

    for i, kw in enumerate(KEYWORDS):
        col_name = f"kw_sim_{kw.replace(' ', '_')}"
        df.loc[valid_indices, col_name] = cosine_np[:, i]

    return df

# ===================== SAVE =========================
def save(df):
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[DONE] Saved full dataset to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")

# ===================== MAIN =========================
def run():
    print("\n========== FULL KEYWORD EMBEDDING PIPELINE ==========\n")

    df = load_and_combine()
    cosine_scores, valid_df = compute_embeddings(df)
    df = build_output(df, cosine_scores, valid_df)
    save(df)

    print("\n====================================================\n")

# ===================== RUN =========================
if __name__ == "__main__":
    run()
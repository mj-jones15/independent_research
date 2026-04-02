import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ===================== CONFIG ============================
MODEL_NAME = "intfloat/e5-large-v2"

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


def run_keyword_enrichment(csv_path: str) -> None:
    """
    Load a CSV with an 'abstract' column, compute E5 embedding similarity
    against KEYWORDS, and write closest_keyword, closest_keyword_similarity,
    and keyword_similarity_score back to the same file.
    """
    print(f"\n========== KEYWORD ENRICHMENT ==========")
    print(f"[INFO] Reading: {csv_path}")
    df = pd.read_csv(csv_path)

    if "abstract" not in df.columns:
        raise ValueError("CSV must contain an 'abstract' column.")

    abstracts = df["abstract"].fillna("").tolist()

    e5_abstracts = [f"passage: {a}" for a in abstracts]
    e5_keywords  = [f"query: {kw}" for kw in KEYWORDS]

    print(f"[INFO] Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Encoding abstracts...")
    abstract_embeddings = model.encode(
        e5_abstracts,
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

    print("[INFO] Computing cosine similarities...")
    cosine_scores = util.cos_sim(abstract_embeddings, keyword_embeddings)

    max_similarities, max_indices = cosine_scores.max(dim=1)
    max_similarities = max_similarities.cpu().numpy()
    max_indices      = max_indices.cpu().numpy()

    df["closest_keyword"]            = [KEYWORDS[i] for i in max_indices]
    df["closest_keyword_similarity"] = max_similarities
    df["keyword_similarity_score"]   = cosine_scores.mean(dim=1).cpu().numpy()

    df.to_csv(csv_path, index=False)
    print(f"[DONE] Keyword enrichment saved to {csv_path}")
    print("=========================================\n")


# ===================== STANDALONE ========================
if __name__ == "__main__":
    DEFAULT_CSV = "./data/neurips_papers_enriched.csv"
    run_keyword_enrichment(DEFAULT_CSV)
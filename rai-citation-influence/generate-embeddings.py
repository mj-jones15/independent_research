import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ===================== CONFIG ============================
ENRICHED_CSV = "./data/aies_papers_enriched.csv"
MODEL_NAME = "intfloat/e5-large-v2"

# ===================== KEYWORDS =========================
keywords = [
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

# =========================================================
# Load data
df = pd.read_csv(ENRICHED_CSV)

# Check abstracts
if "abstract" not in df.columns:
    raise ValueError("The enriched CSV must contain an 'abstract' column.")

abstracts = df["abstract"].fillna("").tolist()

# Format inputs for E5: prefix abstracts and keywords
e5_abstracts = [f"passage: {abs_text}" for abs_text in abstracts]
e5_keywords = [f"query: {kw}" for kw in keywords]

# ===================== MODEL ============================
print(f"Loading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Generating embeddings for abstracts...")
abstract_embeddings = model.encode(
    e5_abstracts, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True
)

print("Generating embeddings for keywords...")
keyword_embeddings = model.encode(
    e5_keywords, convert_to_tensor=True, normalize_embeddings=True
)

# ===================== SIMILARITY ========================
# Compute cosine similarity between each abstract and each keyword
# Result: matrix (num_abstracts x num_keywords)
print("Calculating cosine similarities...")
cosine_scores = util.cos_sim(abstract_embeddings, keyword_embeddings)  # torch tensor

# For each abstract, find the keyword with the highest similarity
max_similarities, max_indices = cosine_scores.max(dim=1)
max_similarities = max_similarities.cpu().numpy()
max_indices = max_indices.cpu().numpy()

# Add closest keyword and similarity scores to dataframe
df["closest_keyword"] = [keywords[idx] for idx in max_indices]
df["closest_keyword_similarity"] = max_similarities

# Optional: add average similarity score as well
avg_similarity = cosine_scores.mean(dim=1).cpu().numpy()
df["keyword_similarity_score"] = avg_similarity

# ===================== SAVE =============================
df.to_csv(ENRICHED_CSV, index=False)
print(f"[DONE] Closest keyword and similarity scores saved to {ENRICHED_CSV}")
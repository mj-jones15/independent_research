import pandas as pd
import numpy as np

# ===================== CONFIG =====================
CHUNKED_POLICY_CSV = "./data/policy_documents/policy_keyword_scores_all.csv"
ABSTRACT_SIMILARITY_CSV = "./data/final_papers_with_keyword_similarities.csv"

OUTPUT_ALIGNMENT_CSV = "./data/results/policy_research_alignment.csv"
OUTPUT_TOPK_CSV = "./data/results/top_policy_abstract_matches.csv"
OUTPUT_NORMALIZED_CSV = "./data/results/policy_research_alignment_normalized.csv"

# Top-K percent for keywords in the policy matrix (if we want to filter out low-relevance keywords)
TOP_K_PERCENT = 0.10

# For top k results, we can set a minimum alignment score threshold to filter out weak matches
TOP_K_RESULTS = 5
MIN_ALIGNMENT_SCORE = 0  # Adjust based on desired strictness
MIN_CHUNKS = 3  # Minimum number of chunks to consider for top-k mean (to avoid too few samples)

# ===================== LOAD POLICY MATRIX =====================
# Top-k mean function
def top_k_mean(scores, k=TOP_K_PERCENT):
    scores = sorted(scores, reverse=True)
    cutoff = max(MIN_CHUNKS, int(len(scores) * k))
    return np.mean(scores[:cutoff])

# P: (documents x keywords)
# We expect format: document, chunk, keyword, similarity (multi-keyword format from chunked dataset)
raw_policy_df = pd.read_csv(CHUNKED_POLICY_CSV, index_col=0)
# Group by document and keyword, then aggregate similarity scores into lists for top-k mean calculation
grouped = raw_policy_df.groupby(["document", "keyword"])["similarity"].apply(list)
# Apply top-k mean to each group and reset index for pivoting
aggregated = grouped.apply(lambda x: top_k_mean(x, k=TOP_K_PERCENT)).reset_index()
# Pivot to document x keyword matrix
policy_matrix_df = aggregated.pivot(index="document", columns="keyword", values="similarity").fillna(0.0)
# Now we have a policy matrix where each cell represents the top-k mean similarity for that document-keyword pair
P = policy_matrix_df.values
# Clip negative values to zero (if any) to ensure non-negativity for matrix multiplication
P = np.clip(P, 0, None)
# Store document and keyword labels for later use
policy_docs = policy_matrix_df.index.tolist()
keywords = policy_matrix_df.columns.tolist()

# ===================== BUILD ABSTRACT MATRIX =====================
# We expect format:
# title, keyword, similarity (multi-keyword format from final combined dataset)
abs_df = pd.read_csv(ABSTRACT_SIMILARITY_CSV)

# ===================== FILTER NON-RAI PAPERS =====================
# Step 1: Max similarity per paper
paper_max = abs_df.groupby("title")["primary_keyword_score"].max()

# Step 2: Map paper -> source (AIES vs NeurIPS)
paper_source = abs_df.groupby("title")["source"].first()

# Step 3: Keep all AIES + only strong NeurIPS
valid_titles = [
    title for title in paper_max.index
    if (
        paper_source[title] == "AIES"  # keep ALL AIES
        or paper_max[title] >= 0.8     # filter ONLY NeurIPS
    )
]

# Step 4: Apply filter
abs_df = abs_df[abs_df["title"].isin(valid_titles)]

print(f"[INFO] Total papers after filtering: {len(valid_titles)}")
print(f"[INFO] AIES kept: {sum(paper_source[t] == 'AIES' for t in valid_titles)}")
print(f"[INFO] NeurIPS kept: {sum(paper_source[t] == 'NeurIPS' for t in valid_titles)}")

# ===================== CONTINUE ABSTRACT MATRIX =====================

# Melt wide -> long
kw_cols = [c for c in abs_df.columns if c.startswith("kw_sim_")]

abs_long = abs_df.melt(
    id_vars=["title", "source", "primary_keyword_score"],  # keep metadata
    value_vars=kw_cols,
    var_name="keyword",
    value_name="similarity"
)

# Strip prefix AND fix underscores → spaces to match policy CSV
# e.g. "kw_sim_artificial_intelligence_fairness" → "artificial intelligence fairness"
abs_long["keyword"] = (
    abs_long["keyword"]
    .str.replace("^kw_sim_", "", regex=True)
    .str.replace("_", " ")
)

# Pivot to keyowrd x abstract matrix
abstract_matrix_df = abs_long.pivot_table(
    index="keyword",
    columns="title",
    values="similarity",
    aggfunc="mean",  # Average similarity if multiple entries (shouldn't happen after filtering)
).fillna(0.0)

# If this is 0, you have a naming mismatch between your policy and abstract datasets. Check the keyword names in both CSVs.
matched = set(keywords) & set(abstract_matrix_df.index)
print(f"[SANITY] Keywords matched: {len(matched)} / {len(keywords)}")
print("Policy keywords sample:", keywords[:3])
print("Abstract keywords sample:", list(abstract_matrix_df.index)[:3])

# Align keywords to match policy matrix order
abstract_matrix_df = abstract_matrix_df.reindex(keywords).fillna(0.0)

# R: (keywords x abstracts)
R = abstract_matrix_df.values
# Clip negative values to zero (if any) to ensure non-negativity for matrix multiplication
R = np.clip(R, 0, None)
abstract_ids = abstract_matrix_df.columns.tolist()  # titles used as IDs

# ===================== ROW NORMALIZATION =====================
def row_normalize(matrix):
    matrix = np.clip(matrix, 0, None)  # drop negative similarities
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return matrix / row_sums

# ===================== MATRIX MULTIPLICATION =====================
M = P @ R  # (documents x keywords) @ (keywords x abstracts) → (documents x abstracts)
# Optional: Row normalize the alignment matrix to get relative scores per policy document
# M = row_normalize(M)

alignment_df = pd.DataFrame(M, index=policy_docs, columns=abstract_ids)
# Updated saving block
alignment_df.to_csv(
    OUTPUT_ALIGNMENT_CSV, 
    index=True,
    index_label="policy_document", # This labels the first column
    columns=abstract_ids,          # This ensures the abstract titles are the column headers
    header=True                    # This ensures the paper titles are the top row
)

# Normalization: Subtract the mean score of each paper across all policies
# This highlights where a paper is UNUSUALLY relevant to a specific policy
alignment_norm = alignment_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Save the normalized alignment matrix as well
alignment_norm.to_csv(
    OUTPUT_NORMALIZED_CSV,
    index=True, index_label="policy_document",
    columns=abstract_ids,
    header=True
)

print("[DONE] Alignment matrix created.")
print(f"Shape: {alignment_df.shape}")

# ===================== TOP-K MATCHES =====================
results = []

for i, doc in enumerate(policy_docs):
    scores = M[i]
    top_indices = np.argsort(scores)[-TOP_K_RESULTS:][::-1]

    for rank, idx in enumerate(top_indices, start=1):
        if scores[idx] >= MIN_ALIGNMENT_SCORE:
            results.append({
                "policy_document": doc,
                "abstract_title": abstract_ids[idx],
                "alignment_score": scores[idx],
                "rank": rank
            })

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_TOPK_CSV, index=False)

print("[DONE] Top-k matches saved.")


# ===================== Analyze Results =====================
# Create a small summary snapshot for quick viewing
top_policies = alignment_df.mean(axis=1).nlargest(10).index
top_papers = alignment_df.mean(axis=0).nlargest(10).index

snapshot = alignment_df.loc[top_policies, top_papers]
print("\n--- Top 10x10 Alignment Snapshot ---")
print(snapshot)

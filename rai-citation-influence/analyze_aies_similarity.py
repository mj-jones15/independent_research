import pandas as pd
import numpy as np

# ===================== CONFIG ============================
INPUT_CSV = "./data/final_papers_with_keyword_similarities.csv"

# ===================== MAIN ============================
def analyze_aies_similarity():
    print("\n========== AIES SIMILARITY ANALYSIS ==========\n")

    # Load data
    df = pd.read_csv(INPUT_CSV)

    if "source" not in df.columns:
        raise ValueError("Missing 'source' column")
    if "primary_keyword_score" not in df.columns:
        raise ValueError("Missing 'primary_keyword_score' column")

    # Filter AIES papers
    aies_df = df[df["source"] == "AIES"].copy()

    print(f"[INFO] Total AIES papers: {len(aies_df)}")

    # Drop NaN scores (empty abstracts)
    scores = aies_df["primary_keyword_score"].dropna()

    print(f"[INFO] Valid AIES embeddings: {len(scores)}")

    if len(scores) == 0:
        print("[ERROR] No valid scores found.")
        return

    # ===================== STATS ============================
    mean_score = scores.mean()
    std_score  = scores.std()

    print("\n--- CORE STATS ---")
    print(f"Mean: {mean_score:.4f}")
    print(f"Std:  {std_score:.4f}")
    print(f"Min:  {scores.min():.4f}")
    print(f"Max:  {scores.max():.4f}")

    # ===================== DISTRIBUTION ============================
    percentiles = [50, 75, 85, 90, 95, 99]

    print("\n--- PERCENTILES ---")
    for p in percentiles:
        val = np.percentile(scores, p)
        print(f"{p}th percentile: {val:.4f}")

    # ===================== SUGGESTED THRESHOLDS ============================
    print("\n--- SUGGESTED THRESHOLDS ---")
    print(f"Mean + 1 std: {mean_score + std_score:.4f}")
    print(f"Mean + 2 std: {mean_score + 2*std_score:.4f}")

    print("\n=============================================\n")


# ===================== RUN ============================
if __name__ == "__main__":
    analyze_aies_similarity()
import os
import pandas as pd
import numpy as np

# ===================== CONFIG ============================
INPUT_CSV = "./data/neurips_papers.csv"
OUTPUT_CSV = "./data/neurips_papers_enriched.csv"

# ===================== ENRICHMENT FUNCTION ============================

def enrich_neurips_csv(input_path: str, output_path: str):
    print("\n========== NEURIPS ENRICHMENT ==========\n")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] Input CSV not found: {input_path}")

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)

    # ===============================
    # Ensure required columns exist
    # ===============================
    required_cols = [
        "title","authors","year","venue","doi","url",
        "citation_count","semantic_scholar_id","abstract"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # ===============================
    # Clean numeric columns
    # ===============================
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)

    # Convert year safely → missing → NaN (NOT 0)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # ===============================
    # GLOBAL STATS
    # ===============================
    global_mean = df["citation_count"].mean()
    global_std = df["citation_count"].std()

    df["global_z"] = (df["citation_count"] - global_mean) / global_std

    # ===============================
    # YEAR-LEVEL STATS (ONLY WHERE YEAR EXISTS)
    # ===============================

    # Compute only for valid years
    df["year_mean"] = df.groupby("year")["citation_count"].transform("mean")
    df["year_std"] = df.groupby("year")["citation_count"].transform("std")

    # Avoid division by zero
    df["year_std"] = df["year_std"].replace(0, np.nan)

    df["year_zscore"] = (df["citation_count"] - df["year_mean"]) / df["year_std"]

    # If year is NaN → all year stats remain NaN (correct behavior)

    # ===============================
    # PERCENTILE BUCKETS (GLOBAL)
    # ===============================
    percentiles = [95, 85, 70, 50]
    thresholds = {p: df["citation_count"].quantile(p / 100) for p in percentiles}

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

    df["percentile_bucket"] = df["citation_count"].apply(percentile_bucket)

    # ===============================
    # SORTING (SAFE WITH NaN YEARS)
    # ===============================
    df.sort_values(by=["year", "citation_count"], ascending=[True, False], inplace=True)

    # ===============================
    # SAVE
    # ===============================
    df.to_csv(output_path, index=False)

    print(f"[DONE] Enriched file saved to: {output_path}")

    # ===============================
    # LOGGING
    # ===============================
    print("\n========== SUMMARY ==========")
    print(f"Total papers: {len(df)}")
    print(f"Missing years: {df['year'].isna().sum()}")
    print(f"Mean citations: {global_mean:.2f}")
    print(f"Std citations: {global_std:.2f}")
    print("=============================\n")


# ===================== RUN ============================
if __name__ == "__main__":
    enrich_neurips_csv(INPUT_CSV, OUTPUT_CSV)
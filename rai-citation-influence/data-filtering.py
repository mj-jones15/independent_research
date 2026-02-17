import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "./data/aies_papers.csv"
OUTPUT_DIR = "./graphs_and_buckets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PLOT_YEAR_MEAN = f"{OUTPUT_DIR}/citations_by_year.png"
PLOT_YEAR_Z = f"{OUTPUT_DIR}/year_normalized_zscores.png"
PLOT_TAIL = f"{OUTPUT_DIR}/citation_tail_histogram.png"
PLOT_LOG = f"{OUTPUT_DIR}/citation_log_distribution.png"
PLOT_PERCENTILES = f"{OUTPUT_DIR}/citation_percentile_buckets.png"
PLOT_LORENZ = f"{OUTPUT_DIR}/citation_lorenz_curve.png"
ENRICHED_CSV = "./data/aies_papers_enriched.csv"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)
df["year"] = df["year"].astype(int)

# ============================================================
# GLOBAL STATS
# ============================================================

global_mean = df["citation_count"].mean()
global_std = df["citation_count"].std()

print("========== GLOBAL CITATION STATS ==========")
print(f"Mean: {global_mean:.2f}")
print(f"Std Dev: {global_std:.2f}")
print("===========================================\n")

# ============================================================
# 1️⃣ YEAR-NORMALIZED Z-SCORES
# ============================================================

df["year_mean"] = df.groupby("year")["citation_count"].transform("mean")
df["year_std"] = df.groupby("year")["citation_count"].transform("std").replace(0, np.nan)
df["year_zscore"] = (df["citation_count"] - df["year_mean"]) / df["year_std"]

# ============================================================
# 2️⃣ PLOT: MEAN CITATIONS BY YEAR
# ============================================================

year_means = df.groupby("year")["citation_count"].mean()

plt.figure(figsize=(8, 5))
year_means.plot(marker="o")
plt.title("Mean Citation Count by Publication Year (AIES)")
plt.xlabel("Year")
plt.ylabel("Mean Citations")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_YEAR_MEAN)
plt.close()

print(f"[DONE] Saved: {PLOT_YEAR_MEAN}")

# ============================================================
# 3️⃣ PLOT: YEAR-NORMALIZED Z-SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(df["year_zscore"].dropna(), bins=30)
plt.title("Distribution of Year-Normalized Citation Z-Scores")
plt.xlabel("Z-score (within publication year)")
plt.ylabel("Number of papers")
plt.tight_layout()
plt.savefig(PLOT_YEAR_Z)
plt.close()

print(f"[DONE] Saved: {PLOT_YEAR_Z}")

# ============================================================
# 4️⃣ TAIL HISTOGRAM — 0.33σ BUCKETS
# ============================================================

df["global_z"] = (df["citation_count"] - global_mean) / global_std
bins = np.arange(df["global_z"].min(), df["global_z"].max() + 0.33, 0.33)

plt.figure(figsize=(8, 5))
plt.hist(df["global_z"], bins=bins)
plt.title("Citation Distribution Tail (0.33σ bins)")
plt.xlabel("Global citation z-score")
plt.ylabel("Number of papers")
plt.tight_layout()
plt.savefig(PLOT_TAIL)
plt.close()

print(f"[DONE] Saved: {PLOT_TAIL}")

# ============================================================
# 5️⃣ LOG-SCALE RAW CITATION DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(df["citation_count"], bins=50, log=True)
plt.title("Log-Scale Citation Distribution")
plt.xlabel("Citation count")
plt.ylabel("Number of papers (log scale)")
plt.tight_layout()
plt.savefig(PLOT_LOG)
plt.close()

print(f"[DONE] Saved: {PLOT_LOG}")

# ============================================================
# 6️⃣ PERCENTILE BUCKETS (TOP 1%, 5%, 10%)
# ============================================================

percentiles = [95, 85, 70, 50]
thresholds = {p: np.percentile(df["citation_count"], p) for p in percentiles}

def percentile_bucket(c):
    if c >= thresholds[95]:
        return "Top 5%"
    elif c >= thresholds[85]:
        return "Top 15%"
    elif c >= thresholds[70]:
        return "Top 30%"
    elif c >= thresholds[50]:
        return "Top 50%"
    else:
        return "Bottom 50%"

df["percentile_bucket"] = df["citation_count"].apply(percentile_bucket)

bucket_counts = df["percentile_bucket"].value_counts().reindex(
    ["Top 5%", "Top 15%", "Top 30%", "Top 50%", "Bottom 50%"]
)

plt.figure(figsize=(8, 5))
bucket_counts.plot(kind="bar")
plt.title("Citation Percentile Buckets (AIES)")
plt.xlabel("Percentile group")
plt.ylabel("Number of papers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(PLOT_PERCENTILES)
plt.close()

print(f"[DONE] Saved: {PLOT_PERCENTILES}")

# ============================================================
# 7️⃣ LORENZ CURVE + GINI
# ============================================================

citations_sorted = np.sort(df["citation_count"].values)
cum_citations = np.cumsum(citations_sorted)
cum_citations = np.insert(cum_citations, 0, 0)
cum_share = cum_citations / cum_citations[-1]
pop_share = np.linspace(0, 1, len(cum_share))

# Gini coefficient
gini = 1 - 2 * np.trapezoid(cum_share, pop_share)

plt.figure(figsize=(6, 6))
plt.plot(pop_share, cum_share, label="Lorenz Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Equality Line")
plt.title(f"Lorenz Curve of Citation Distribution\nGini = {gini:.3f}")
plt.xlabel("Fraction of papers")
plt.ylabel("Fraction of citations")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_LORENZ)
plt.close()

print(f"[DONE] Saved: {PLOT_LORENZ}")
print(f"[INFO] Gini coefficient: {gini:.3f}")

# ============================================================
# SAVE ENRICHED DATASET
# ============================================================

df.sort_values(by="citation_count", ascending=False).to_csv(ENRICHED_CSV, index=False)
print(f"[DONE] Enriched dataset written to: {ENRICHED_CSV}")

print("\n========== SUMMARY ==========")
print("New columns added:")
print(" - year_mean")
print(" - year_std")
print(" - year_zscore")
print(" - global_z")
print(" - percentile_bucket")
print("Plots generated:")
print(" - citations_by_year.png")
print(" - year_normalized_zscores.png")
print(" - citation_tail_histogram.png")
print(" - citation_log_distribution.png")
print(" - citation_percentile_buckets.png")
print(" - citation_lorenz_curve.png")
print("================================\n")


# What have we learned?
# The mean citation count is 34.59
# The standard deviation is 104.74
# Bucket counts:
# 0–1σ    501
# 1–2σ     15
# 2–3σ      9
# 3+σ       5
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

ABSTRACT_CSV = "./data/final_papers_with_keyword_similarities.csv"
OUTPUT_DIR = "./data/results"

categories = {
    "fairness": [
        "artificial intelligence fairness",
        "artificial intelligence equality",
        "artificial intelligence equity",
        "artificial intelligence equitable",
    ],
    "privacy": [
        "artificial intelligence privacy",
        "artificial intelligence anonymity",
        "artificial intelligence confidentiality",
        "artificial intelligence confidential",
    ],
    "explainability": [
        "artificial intelligence explainability",
        "explainable artificial intelligence",
    ],
    "accountability": [
        "accountable artificial intelligence",
        "artificial intelligence accountability",
        "artificial intelligence transparency",
        "artificial intelligence auditability",
        "artificial intelligence governance",
        "artificial intelligence compliance",
        "artificial intelligence accountability mechanisms",
        "artificial intelligence algorithmic accountability",
    ],
    "sustainability": [
        "green artificial intelligence",
        "energy-efficient artificial intelligence",
        "artificial intelligence carbon footprint",
        "artificial intelligence environmental impact",
        "eco-friendly artificial intelligence",
        "artificial intelligence energy consumption",
        "artificial intelligence green computing",
    ]
}
category_order = ["fairness", "accountability", "explainability", "privacy", "sustainability"]


palette = {
    "fairness": "#4C72B0",
    "accountability": "#C44E52",
    "explainability": "#55A868",
    "privacy": "#DD8452",
    "sustainability": "#8172B2",
}
bar_colors = [palette[c] for c in category_order]

keyword_to_category = {
    kw: cat
    for cat, kws in categories.items()
    for kw in kws
}

# ============================================================
# LOAD + FILTER
# ============================================================

df = pd.read_csv(ABSTRACT_CSV)
df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)

# Apply the same filter logic established in alignment pipeline
paper_max = df.groupby("title")["primary_keyword_score"].max()
paper_source = df.groupby("title")["source"].first()

valid_titles = [
    t for t in paper_max.index
    if paper_source[t] == "AIES" or paper_max[t] >= 0.8
]
df = df[df["title"].isin(valid_titles)].copy()

# De-duplicate to one row per paper (the CSV has one row per keyword per paper)
# Keep the row where primary_keyword_score is highest — that's the paper's closest keyword
paper_df = (
    df.sort_values("primary_keyword_score", ascending=False)
    .drop_duplicates(subset="title")
    .copy()
)

# Recompute year_zscore fresh within this dataset
# Baseline = all RAI papers (AIES + filtered NeurIPS) in that year
# This gives within-RAI comparison (how does this category perform vs other RAI work?)
year_stats = paper_df.groupby("year")["citation_count"].agg(["mean", "std"])
paper_df["year_mean_rai"] = paper_df["year"].map(year_stats["mean"])
paper_df["year_std_rai"] = paper_df["year"].map(year_stats["std"]).replace(0, pd.NA)
paper_df["year_zscore_rai"] = (
    (paper_df["citation_count"] - paper_df["year_mean_rai"]) / paper_df["year_std_rai"]
)

print("\n[INFO] Z-score recomputed within RAI population by year")
print(f"Years with <2 papers (z undefined): {paper_df['year_zscore_rai'].isna().sum()}")

# Assign category from primary_keyword column
paper_df["category"] = paper_df["primary_keyword"].map(keyword_to_category)

# Papers whose primary keyword isn't in any category get dropped
# (shouldn't happen if your keywords are consistent, but safety net)
unmatched = paper_df["category"].isna().sum()
if unmatched > 0:
    print(f"[WARNING] {unmatched} papers had a primary keyword not in any category — dropped")
paper_df = paper_df.dropna(subset=["category"])

print(f"[INFO] Papers after filtering and dedup: {len(paper_df)}")
print(f"[INFO] AIES: {(paper_df['source'] == 'AIES').sum()}")
print(f"[INFO] NeurIPS: {(paper_df['source'] == 'NeurIPS').sum()}")
print(f"\nCategory distribution:\n{paper_df['category'].value_counts()}")

# ============================================================
# FIGURE 1: Research Production by RAI Category
# Split by venue (AIES vs NeurIPS)
# ============================================================

category_counts = (
    paper_df.groupby("category")
    .size()
    .reindex(category_order, fill_value=0)
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    category_order,
    category_counts,
    color=bar_colors,
    alpha=0.88,
    edgecolor="white",
    width=0.55
)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1, str(int(h)),
            ha="center", va="bottom", fontsize=10)

ax.set_xticks(range(len(category_order)))
ax.set_xticklabels([c.capitalize() for c in category_order], fontsize=11)
ax.set_ylabel("Number of Papers", fontsize=11)
ax.set_title(
    "RAI Research Production by Category",
    fontsize=13
)
ax.annotate(
    "Sources: All AIES papers + NeurIPS papers with primary keyword score ≥ 0.80",
    xy=(0.01, 0.01), xycoords="axes fraction", fontsize=8, color="gray"
)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rai_production_by_category.png", dpi=150)
plt.show()
# ============================================================
# FIGURE 2: Citation Impact by RAI Category (Year-Normalized)
# ============================================================

# Compute mean and std of year_zscore per category
zscore_stats = (
    paper_df.groupby("category")["year_zscore_rai"]
    .agg(["mean", "std", "count"])
    .reindex(category_order)
)

# Standard error for error bars
zscore_stats["se"] = zscore_stats["std"] / np.sqrt(zscore_stats["count"])

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(
    category_order,
    zscore_stats["mean"],
    color=bar_colors,
    alpha=0.85,
    edgecolor="white",
    width=0.5,
    yerr=zscore_stats["se"],
    capsize=5,
    error_kw={"elinewidth": 1.5, "ecolor": "gray"}
)

# Draw zero line — this is the average paper across all years
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Overall average")

# Annotate each bar with n= and mean
for i, (cat, row) in enumerate(zscore_stats.iterrows()):
    ax.text(
        i, row["mean"] + row["se"] + 0.015,
        f"n={int(row['count'])}\nz={row['mean']:.2f}",
        ha="center", va="bottom", fontsize=8.5
    )

ax.set_xticklabels([c.capitalize() for c in category_order], fontsize=11)
ax.set_ylabel("Mean Year-Normalized Citation Z-Score", fontsize=11)
ax.set_title(
    "RAI Research Citation Impact by Category\n"
    "(z-score relative to average RAI paper published that year)",
    fontsize=13
)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="Average RAI paper (that year)")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/rai_citation_impact_by_category.png", dpi=150)
plt.show()

print("[DONE] Both figures saved.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import statsmodels.formula.api as smf
from collections import Counter
from itertools import chain
from scipy.stats import f_oneway

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "./data/aies_papers_enriched.csv"
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


def analyze_closest_keywords():
    """
    Analyze embedding-assigned closest keywords.
    Examines:
        - Frequency distribution
        - Percentage share
        - Mean citation count per keyword
        - Mean year-normalized z-score per keyword
        - Correlation between similarity score and citation count
    """

    if "closest_keyword" not in df.columns:
        print("[WARNING] 'closest_keyword' column not found.")
        return

    working_df = df[df["closest_keyword"].notna()].copy()

    if working_df.empty:
        print("No closest keyword data available.")
        return

    print("\n========== CLOSEST KEYWORD ANALYSIS ==========")

    # --------------------------------------------------------
    # 1 Frequency distribution
    # --------------------------------------------------------

    keyword_counts = working_df["closest_keyword"].value_counts()
    total = len(working_df)

    print("\nTop Keywords by Frequency:")
    for kw, count in keyword_counts.head(20).items():
        pct = (count / total) * 100
        print(f"{kw}: {count} papers ({pct:.2f}%)")

    # --------------------------------------------------------
    # 2 Citation impact per keyword
    # --------------------------------------------------------

    citation_stats = (
        working_df
        .groupby("closest_keyword")["citation_count"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nTop Keywords by Mean Citation Count:")
    for kw, mean_cites in citation_stats.head(15).items():
        print(f"{kw}: {mean_cites:.2f} mean citations")

    # --------------------------------------------------------
    # 3 Year-normalized impact
    # --------------------------------------------------------

    if "year_zscore" in working_df.columns:
        zscore_stats = (
            working_df
            .groupby("closest_keyword")["year_zscore"]
            .mean()
            .sort_values(ascending=False)
        )

        print("\nTop Keywords by Mean Year-Normalized Z-Score:")
        for kw, mean_z in zscore_stats.head(15).items():
            print(f"{kw}: {mean_z:.2f} mean year-z")

    # --------------------------------------------------------
    # 4 ANOVA for keyword → year_zscore
    # --------------------------------------------------------

    print("\n[INFO] Running ANOVA: keyword → year_zscore")

    groups = [
        group["year_zscore"].dropna().values
        for name, group in working_df.groupby("closest_keyword")
        if len(group) >= 5   # minimum paper threshold
    ]

    if len(groups) > 1:
        F, p = f_oneway(*groups)
        print(f"ANOVA F = {F:.4f}")
        print(f"p-value = {p:.6f}")
    else:
        print("Not enough keyword groups for ANOVA.")

    # --------------------------------------------------------
    # 5 REGRESSION: keyword → year_zscore (min 5 papers)
    # --------------------------------------------------------

    print("\n[INFO] Running OLS regression: year_zscore ~ closest_keyword")

    # Drop missing
    reg_df = working_df.dropna(subset=["year_zscore", "closest_keyword"]).copy()

    # Remove keywords with < 5 papers
    keyword_counts = reg_df["closest_keyword"].value_counts()
    valid_keywords = keyword_counts[keyword_counts >= 5].index

    reg_df = reg_df[reg_df["closest_keyword"].isin(valid_keywords)]

    print(f"[INFO] Keywords retained (>=5 papers): {len(valid_keywords)}")

    # Convert to categorical
    reg_df["closest_keyword"] = reg_df["closest_keyword"].astype("category")

    # Run regression
    import statsmodels.formula.api as smf
    model = smf.ols("year_zscore ~ C(closest_keyword)", data=reg_df).fit()

    print("\nRegression Summary:")
    print(model.summary())

    # ============================================================
    # CITATION PERFORMANCE BY KEYWORD
    # ============================================================

    print("\n[INFO] Computing citation statistics by keyword...")

    keyword_stats = (
        df.groupby("closest_keyword")["citation_count"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )

    print("\nTop keywords by average citation count:")
    print(keyword_stats.head(15))

     # verify correct data loading
    print("\nTotal unique closest keywords:")
    print(len(keyword_stats))

    print("\nFull keyword count distribution:")
    print(keyword_stats["count"].sort_values(ascending=False))


    # Ensure output directory exists
    os.makedirs("./graphs_and_buckets", exist_ok=True)

    # ------------------------------------------------------------
    # GRAPH 1: Average Citation Count per Keyword
    # ------------------------------------------------------------
    plt.figure()
    keyword_stats["mean"].plot(kind="bar")
    plt.ylabel("Average Citation Count")
    plt.title("Average Citations per Closest Keyword")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./graphs_and_buckets/avg_citations_per_keyword.png")
    plt.close()

    # ------------------------------------------------------------
    # GRAPH 2: Citation Standard Deviation per Keyword
    # ------------------------------------------------------------
    plt.figure()
    keyword_stats["std"].plot(kind="bar")
    plt.ylabel("Citation Standard Deviation")
    plt.title("Citation Variability per Closest Keyword")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("./graphs_and_buckets/std_citations_per_keyword.png")
    plt.close()

    print("[DONE] Saved keyword citation analysis graphs to ./graphs_and_buckets")

    print("==============================================\n")



if __name__ == "__main__":
    analyze_closest_keywords()


# What have we learned?
# The mean citation count is 34.59
# The standard deviation is 104.74
# Bucket counts:
# 0–1σ    501
# 1–2σ     15
# 2–3σ      9
# 3+σ       5
#


''''
========== CLOSEST KEYWORD ANALYSIS ==========

Top Keywords by Frequency:
artificial intelligence fairness: 154 papers (29.06%)
accountable artificial intelligence: 52 papers (9.81%)
artificial intelligence equitable: 37 papers (6.98%)
artificial intelligence algorithmic accountability: 36 papers (6.79%)
artificial intelligence explainability: 36 papers (6.79%)
artificial intelligence compliance: 36 papers (6.79%)
artificial intelligence governance: 31 papers (5.85%)
explainable artificial intelligence: 28 papers (5.28%)
artificial intelligence equality: 26 papers (4.91%)
artificial intelligence privacy: 20 papers (3.77%)
artificial intelligence transparency: 17 papers (3.21%)
artificial intelligence accountability: 13 papers (2.45%)
artificial intelligence confidentiality: 9 papers (1.70%)
artificial intelligence auditability: 8 papers (1.51%)
artificial intelligence equity: 6 papers (1.13%)
artificial intelligence accountability mechanisms: 6 papers (1.13%)
artificial intelligence environmental impact: 4 papers (0.75%)
artificial intelligence confidential: 4 papers (0.75%)
artificial intelligence energy consumption: 2 papers (0.38%)
eco-friendly artificial intelligence: 2 papers (0.38%)

Top Keywords by Mean Citation Count:
explainable artificial intelligence: 85.96 mean citations
artificial intelligence algorithmic accountability: 46.42 mean citations
artificial intelligence fairness: 46.19 mean citations
artificial intelligence auditability: 43.00 mean citations
artificial intelligence equity: 35.00 mean citations
artificial intelligence explainability: 33.47 mean citations
artificial intelligence equality: 32.50 mean citations
artificial intelligence governance: 31.87 mean citations
artificial intelligence confidentiality: 26.78 mean citations
artificial intelligence compliance: 25.97 mean citations
artificial intelligence environmental impact: 24.25 mean citations
artificial intelligence transparency: 23.59 mean citations
artificial intelligence privacy: 20.15 mean citations
artificial intelligence energy consumption: 20.00 mean citations
artificial intelligence equitable: 17.84 mean citations

Top Keywords by Mean Year-Normalized Z-Score:
artificial intelligence algorithmic accountability: 0.66 mean year-z
artificial intelligence equity: 0.20 mean year-z
artificial intelligence environmental impact: 0.15 mean year-z
artificial intelligence explainability: 0.14 mean year-z
explainable artificial intelligence: 0.14 mean year-z
artificial intelligence equality: 0.12 mean year-z
artificial intelligence auditability: 0.09 mean year-z
artificial intelligence compliance: 0.02 mean year-z
artificial intelligence privacy: -0.02 mean year-z
artificial intelligence transparency: -0.02 mean year-z
artificial intelligence confidentiality: -0.03 mean year-z
artificial intelligence fairness: -0.09 mean year-z
artificial intelligence governance: -0.09 mean year-z
artificial intelligence equitable: -0.15 mean year-z
accountable artificial intelligence: -0.17 mean year-z

Correlation between similarity score and citation count:
Pearson r = 0.0591

[INFO] Computing citation statistics by keyword...

Top keywords by average citation count:
                                                         mean         std  count
closest_keyword                                                                 
explainable artificial intelligence                 85.964286  196.200102     28
artificial intelligence algorithmic accountability  46.416667  125.444438     36
artificial intelligence fairness                    46.194805  152.716709    154
artificial intelligence auditability                43.000000   71.021124      8
artificial intelligence equity                      35.000000   46.121578      6
artificial intelligence explainability              33.472222   64.363915     36
artificial intelligence equality                    32.500000   55.216121     26
artificial intelligence governance                  31.870968   52.511422     31
artificial intelligence confidentiality             26.777778   45.622850      9
artificial intelligence compliance                  25.972222   49.317332     36
artificial intelligence environmental impact        24.250000   17.385339      4
artificial intelligence transparency                23.588235   46.066608     17
artificial intelligence privacy                     20.150000   44.590859     20
artificial intelligence energy consumption          20.000000    8.485281      2
artificial intelligence equitable                   17.837838   25.098156     37

'''''
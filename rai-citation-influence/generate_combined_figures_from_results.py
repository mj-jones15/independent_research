import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import adjustText

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from adjustText import adjust_text

from policy_labels import apply_label, POLICY_LABELS

CHUNKED_POLICY_CSV = "./data/policy_documents/policy_keyword_scores_all.csv"
ABSTRACT_SIMILARITY_CSV = "./data/final_papers_with_keyword_similarities.csv"

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

# Build reverse map: keyword → category
keyword_to_category = {
    kw: cat
    for cat, kws in categories.items()
    for kw in kws
}

def compute_category_scores(chunk_df, keyword_to_category):
    """
    For each (document, category), compute the mean similarity across
    all chunks whose keyword belongs to that category.
    Returns a DataFrame: index=document, columns=categories
    """
    df = chunk_df.copy()
    df["category"] = df["keyword"].map(keyword_to_category)
    df = df.dropna(subset=["category"])  # drop keywords not in any category
    
    cat_scores = (
        df.groupby(["document", "category"])["similarity"]
        .mean()
        .unstack(fill_value=0.0)
    )
    return cat_scores


# 1. LOAD DATA
OUTPUT_ALIGNMENT_CSV = "./data/results/policy_research_alignment.csv"
# Assuming alignment_df is your (Policies x Abstracts) matrix from the previous step
alignment_df = pd.read_csv(OUTPUT_ALIGNMENT_CSV, index_col=0)

# Also load chunk-level keyword scores and abstract-level keyword similarities for the deep dive
chunk_df = pd.read_csv(CHUNKED_POLICY_CSV)
abstract_df = pd.read_csv(ABSTRACT_SIMILARITY_CSV)

# 2. CALCULATE GROUNDING (Strength of Connection)
# We take the average of the top 20 papers for each policy. 
# This represents the "Technical Foundation" available for that policy.
K = 20 
grounding_scores = alignment_df.apply(lambda x: x.nlargest(K).mean(), axis=1)

# 3. CALCULATE KEYWORD BREADTH
# How many different research papers are "hitting" this policy?
# (Counting papers that pass a minimum threshold of alignment)
threshold = alignment_df.values.mean() + (alignment_df.values.std() * 2) # Strict threshold
coverage_count = (alignment_df > threshold).sum(axis=1)

# 4. COMBINE INTO A "CONNECTION REPORT"
connection_df = pd.DataFrame({
    "Research_Foundation_Strength": grounding_scores,
    "Research_Consensus_Count": coverage_count
}).sort_values(by="Research_Foundation_Strength", ascending=False)
# Optionally, filter out outlier
connection_df = connection_df[~connection_df.index.str.contains("general-data-protection-regulation", case=False, na=False)]

category_score_df = compute_category_scores(chunk_df, keyword_to_category)
connection_df["Top_Category"] = category_score_df.idxmax(axis=1)

# 5. VISUALIZATION: Connection Strength vs. Breadth
from adjustText import adjust_text

plt.figure(figsize=(16, 10))

# Color points by their top RAI category
palette = {
    "fairness": "#4C72B0",
    "privacy": "#DD8452",
    "explainability": "#55A868",
    "accountability": "#C44E52",
    "sustainability": "#8172B2",
}
colors = connection_df["Top_Category"].map(palette).fillna("#999999")

# Normalize bubble sizes to a fixed range (min=50, max=600) regardless of raw score magnitude
raw_sizes = connection_df["Research_Foundation_Strength"]
size_scaled = 50 + (raw_sizes - raw_sizes.min()) / (raw_sizes.max() - raw_sizes.min()) * 550

scatter = plt.scatter(
    connection_df["Research_Consensus_Count"],
    connection_df["Research_Foundation_Strength"],
    s=size_scaled,
    c=colors,
    alpha=0.7,
    edgecolors="white",
    linewidths=0.5
)

# Label ALL policies — adjustText will prevent overlap
texts = []
for i, txt in enumerate(connection_df.index):
        short_name = apply_label(txt)
        texts.append(plt.text(
        connection_df["Research_Consensus_Count"].iloc[i],
        connection_df["Research_Foundation_Strength"].iloc[i],
        short_name,
        fontsize=17,
        ha="center"
    ))

adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

# Legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=cat) for cat, c in palette.items()]
plt.legend(handles=legend_elements, title="Dominant RAI Category", loc="lower right")

plt.title(
    "Policy Research Grounding\n"
    "Bubble size & Y-axis = Strength of research foundation  |  X-axis = Width of research consensus",
    fontsize=13
)
plt.xlabel("Research Consensus  (number of papers with strong alignment)", fontsize=11)
plt.ylabel("Research Foundation Strength  (avg. alignment of top-20 papers)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./data/results/policy_grounding_plot.png", dpi=150)
plt.show()

print(connection_df.head(10))

# ============================================================
# 6. DEEP DIVE: TOP POLICIES → CHUNKS → ABSTRACTS
# ============================================================

# Build per-category abstract scores once, reused for every policy
category_abstract_scores = {}
for cat, kws in categories.items():
    # Find matching kw_sim columns
    matching_cols = [
        col for col in abstract_df.columns
        if col.startswith("kw_sim_") and
        col.replace("kw_sim_", "").replace("_", " ") in kws
    ]
    if matching_cols:
        category_abstract_scores[cat] = abstract_df[matching_cols].mean(axis=1)

TOP_CATEGORIES_PER_POLICY = 3  # how many categories to show per policy
TOP_CHUNKS_PER_CATEGORY = 2    # representative chunks per category
TOP_N_ABSTRACTS = 3            # supporting papers per category

doc = SimpleDocTemplate(
    "./data/results/policy_deep_dive.pdf",
    rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50
)
styles = getSampleStyleSheet()
elements = []

# ALL policies, sorted best to worst
all_policies = connection_df.index.tolist()

for policy in all_policies:
    short_name = apply_label(policy)
    foundation = connection_df.loc[policy, "Research_Foundation_Strength"]
    consensus = connection_df.loc[policy, "Research_Consensus_Count"]

    elements.append(Paragraph(
        f"<b>{short_name}</b>",
        styles["Heading2"]
    ))
    elements.append(Paragraph(
        f"Research Foundation Strength: <b>{foundation:.4f}</b> &nbsp;|&nbsp; "
        f"Research Consensus: <b>{int(consensus)} papers</b>",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 8))

    # Get this policy's category scores
    if policy not in category_score_df.index:
        elements.append(Paragraph("No category data available.", styles["Normal"]))
        elements.append(Spacer(1, 20))
        continue

    policy_cat_scores = category_score_df.loc[policy].sort_values(ascending=False)
    top_cats = policy_cat_scores.head(TOP_CATEGORIES_PER_POLICY)

    policy_chunks = chunk_df[chunk_df["document"] == policy]

    for cat, cat_score in top_cats.items():
        elements.append(Paragraph(
            f"<b>Category: {cat.upper()}</b>  (avg. keyword score: {cat_score:.3f})",
            styles["Heading3"]
        ))

        # Find top chunks in this category
        cat_keywords = categories[cat]
        cat_chunks = policy_chunks[policy_chunks["keyword"].isin(cat_keywords)]
        top_chunks = cat_chunks.nlargest(TOP_CHUNKS_PER_CATEGORY, "similarity")

        for _, row in top_chunks.iterrows():
            chunk_text = str(row.get("chunk_preview", ""))[:400]
            elements.append(Paragraph(
                f"<i>Keyword matched: \"{row['keyword']}\" "
                f"(score: {row['similarity']:.3f})</i>",
                styles["Normal"]
            ))
            elements.append(Paragraph(chunk_text, styles["Normal"]))
            elements.append(Spacer(1, 5))

        # Find top supporting abstracts for this category
        if cat in category_abstract_scores:
            cat_abs_scores = category_abstract_scores[cat].copy()
            cat_abs_scores.index = abstract_df.index
            top_abs_indices = cat_abs_scores.nlargest(TOP_N_ABSTRACTS).index

            elements.append(Paragraph(
                f"Supporting Research ({cat}):",
                styles["Normal"]
            ))
            for idx in top_abs_indices:
                row = abstract_df.loc[idx]
                title = row["title"]
                score = cat_abs_scores[idx]
                abstract_text = str(row.get("abstract", ""))[:400]

                elements.append(Paragraph(
                    f"→ <b>{title}</b> (category alignment: {score:.3f})",
                    styles["Normal"]
                ))
                elements.append(Paragraph(abstract_text, styles["Normal"]))
                elements.append(Spacer(1, 5))

        elements.append(Spacer(1, 12))

    elements.append(Spacer(1, 24))

doc.build(elements)
print("[DONE] Deep-dive PDF generated.")


# ============================================================
# 7. HEATMAP: Policy × Category (derived purely from alignment_df)
# ============================================================

# category_abstract_scores already maps category → per-abstract scores (from PDF section above)
# Use it to assign each abstract title its dominant category — no raw CSV re-processing needed

# Build a title-indexed DataFrame from already-computed category scores
cat_score_matrix = pd.DataFrame(category_abstract_scores)  # rows = abstract_df integer index
cat_score_matrix.index = abstract_df["title"]              # swap to title strings

# Drop any abstracts with all-zero scores (no category matched)
cat_score_matrix = cat_score_matrix.loc[(cat_score_matrix > 0).any(axis=1)]

# Each abstract gets assigned to its highest-scoring category
abstract_top_category = cat_score_matrix.idxmax(axis=1)  # Series: title → category

# Filter alignment_df columns to only abstracts we have a category for
known_titles = [t for t in alignment_df.columns if t in abstract_top_category.index]
M_df = alignment_df[known_titles].copy()

print(f"[SANITY] Abstracts with category assignment: {len(known_titles)} / {alignment_df.shape[1]}")

# Rename columns to their category, then average within each category group
M_df.columns = [abstract_top_category[t] for t in M_df.columns]
policy_category_matrix = M_df.T.groupby(level=0).mean().T  # (policies × categories)

# Sort policies by their mean alignment score across all categories (descending)
policy_category_matrix = policy_category_matrix.loc[
    policy_category_matrix.mean(axis=1).sort_values(ascending=False).index
]

# Use the same labeling logic as Figure 1 for consistency
policy_category_matrix.index = [apply_label(idx) for idx in policy_category_matrix.index]

# Plot heatmap
plt.figure(figsize=(10, max(6, len(policy_category_matrix) * 0.35)))
sns.heatmap(
    policy_category_matrix,
    annot=True,
    fmt=".3f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "Mean Research Alignment Score"},
    yticklabels=True
)
plt.title(
    "Policy × RAI Research Category Alignment\n"
    "(each cell = mean alignment score across abstracts in that category)",
    fontsize=12
)
plt.xlabel("RAI Category", fontsize=11)
plt.ylabel("Policy Document", fontsize=11)
plt.xticks(fontsize=8)   # shrink x-axis labels (categories)
plt.tight_layout()
plt.savefig("./data/results/policy_cross_research_category_heatmap.png", dpi=150)
plt.show()

print("[DONE] Deep-dive PDF generated at ./data/results/policy_deep_dive.pdf")


# ============================================================
# 8. FIND MID-RANGE (≈0.7) EXAMPLES FOR CONTRAST
# ============================================================

LOWER_BOUND = 0.65
UPPER_BOUND = 0.75

contrast_examples = []

for policy in connection_df.index:
    policy_chunks = chunk_df[chunk_df["document"] == policy]

    # Filter for mid-range similarity scores
    mid_chunks = policy_chunks[
        (policy_chunks["similarity"] >= LOWER_BOUND) &
        (policy_chunks["similarity"] <= UPPER_BOUND)
    ]

    if len(mid_chunks) == 0:
        continue

    # Take best of the mid-range
    top_mid = mid_chunks.nlargest(2, "similarity")

    for _, row in top_mid.iterrows():
        contrast_examples.append({
            "policy": policy,
            "keyword": row["keyword"],
            "score": row["similarity"],
            "chunk": row.get("chunk_preview", "")[:300]
        })

# Convert to DataFrame
contrast_df = pd.DataFrame(contrast_examples)

# Save for inspection
contrast_df.to_csv("./data/results/midrange_similarity_examples.csv", index=False)

print("[DONE] Mid-range similarity examples saved.")


# ============================================================
# 9. RANKED ALIGNMENT CURVE: High vs Low Scorer
# ============================================================
from reportlab.platypus import Image as RLImage
import io

high_policy = connection_df.index[0]  # best grounded policy
low_policy = connection_df.index[-1]  # worst grounded policy

CURVE_OUTPUT = "./data/results/alignment_curve_comparison.png"
CURVE_PDF_OUTPUT = "./data/results/alignment_curve_report.pdf"
TOP_K_CURVE = 50  # how many top abstracts to show on the curve

high_label = apply_label(high_policy)
low_label = apply_label(low_policy)

# Pull ranked alignment scores directly from the matrix
high_scores = alignment_df.loc[high_policy].sort_values(ascending=False).reset_index(drop=True)
low_scores  = alignment_df.loc[low_policy].sort_values(ascending=False).reset_index(drop=True)

high_top = high_scores.iloc[:TOP_K_CURVE]
low_top  = low_scores.iloc[:TOP_K_CURVE]

# ---- Figure: Ranked Alignment Curve ----
fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(range(1, TOP_K_CURVE + 1), high_top.values,
        color="#55A868", linewidth=2.5, marker="o", markersize=4,
        label=high_label)

ax.plot(range(1, TOP_K_CURVE + 1), low_top.values,
        color="#C44E52", linewidth=2.5, marker="o", markersize=4,
        linestyle="--", label=low_label)

# Mark the threshold line
ax.axhline(threshold, color="black", linewidth=1, linestyle=":",
           label=f"Consensus threshold ({threshold:.2f})")

# Shade the area above threshold for the high scorer
ax.fill_between(range(1, TOP_K_CURVE + 1), high_top.values, threshold,
                where=(high_top.values > threshold),
                alpha=0.12, color="#55A868", label="Egypt above threshold")

# Annotate paper counts above threshold
high_above = (high_top.values > threshold).sum()
low_above  = (low_top.values > threshold).sum()

ax.annotate(f"{high_above} papers above threshold",
            xy=(high_above, threshold),
            xytext=(high_above + 2, threshold + 0.3),
            fontsize=9, color="#55A868",
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=1))

if low_above > 0:
    ax.annotate(f"{low_above} papers above threshold",
                xy=(low_above, threshold),
                xytext=(low_above + 2, threshold + 0.15),
                fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))
else:
    ax.annotate("0 papers above threshold",
                xy=(1, low_top.values[0]),
                xytext=(8, low_top.values[0] + 0.1),
                fontsize=9, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1))

ax.set_xlabel("Abstract Rank (1 = strongest match)", fontsize=11)
ax.set_ylabel("Alignment Score", fontsize=11)
ax.set_title(
    "Research Grounding: Top-50 Abstract Alignment Scores\n"
    "How quickly does research support drop off?",
    fontsize=13
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(CURVE_OUTPUT, dpi=150)
plt.show()
print(f"[DONE] Curve figure saved to {CURVE_OUTPUT}")

# ---- PDF Report: Curve + 20th-place comparison table ----
# Get the actual abstract title and score at rank 20 for each
rank_n = 20

high_ranked = alignment_df.loc[high_policy].sort_values(ascending=False)
low_ranked  = alignment_df.loc[low_policy].sort_values(ascending=False)

def get_rank_n(ranked_series, n):
    if len(ranked_series) >= n:
        return ranked_series.index[n-1], ranked_series.iloc[n-1]
    return "N/A", 0.0

high_20_title, high_20_score = get_rank_n(high_ranked, rank_n)
low_20_title,  low_20_score  = get_rank_n(low_ranked,  rank_n)

# Also get abstract text for each if available
def get_abstract(title):
    matches = abstract_df[abstract_df["title"] == title]
    if matches.empty:
        return ""
    text = str(matches.iloc[0].get("abstract", ""))
    return text[:300] + "…" if len(text) > 300 else text

high_20_abstract = get_abstract(high_20_title)
low_20_abstract  = get_abstract(low_20_title)

# Build PDF
curve_doc = SimpleDocTemplate(
    CURVE_PDF_OUTPUT,
    rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
)
curve_elements = []

curve_elements.append(Paragraph(
    "Research Grounding: Where the Difference Actually Lives",
    styles["Heading1"]
))
curve_elements.append(Paragraph(
    "Both documents use strong AI governance language. The difference is not in "
    "how well their text matches RAI keywords — it is in how many research abstracts "
    "are genuinely aligned with their content through the semantic matrix.",
    styles["Normal"]
))
curve_elements.append(Spacer(1, 12))

# Embed the curve figure
curve_elements.append(RLImage(CURVE_OUTPUT, width=6.5*inch, height=3.8*inch))
curve_elements.append(Spacer(1, 14))

# Rank-20 comparison table
curve_elements.append(Paragraph(
    f"<b>What does the {rank_n}th-best abstract match look like for each policy?</b>",
    styles["Heading2"]
))
curve_elements.append(Spacer(1, 6))

rank_table_data = [
    ["", f"✦ {high_label}", f"✧ {low_label}"],
    ["Rank-20 alignment score",
     f"{high_20_score:.4f}",
     f"{low_20_score:.4f}"],
    ["Abstract title",
     high_20_title[:80] + ("…" if len(high_20_title) > 80 else ""),
     low_20_title[:80]  + ("…" if len(low_20_title)  > 80 else "")],
    ["Abstract excerpt",
     high_20_abstract,
     low_20_abstract],
]

rank_table = Table(rank_table_data, colWidths=[1.5*inch, 2.7*inch, 2.7*inch])
rank_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2C3E50")),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 8),
    ("BACKGROUND",    (0, 1), (0, -1),  colors.HexColor("#EEEEEE")),
    ("FONTNAME",      (0, 1), (0, -1),  "Helvetica-Bold"),
    ("ROWBACKGROUNDS",(1, 1), (-1, -1), [colors.white, colors.HexColor("#F9F9F9")]),
    ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    # Highlight the score row
    ("BACKGROUND",    (1, 1), (1, 1),   colors.HexColor("#D5E8D4")),  # green for high
    ("BACKGROUND",    (2, 1), (2, 1),   colors.HexColor("#F8CECC")),  # red for low
]))
curve_elements.append(rank_table)
curve_elements.append(Spacer(1, 10))

curve_elements.append(Paragraph(
    f"The rank-20 abstract for {high_label} scores {high_20_score:.4f} — "
    f"still above the consensus threshold of {threshold:.4f}. "
    f"For {low_label}, the rank-20 abstract scores {low_20_score:.4f}, "
    f"well below that threshold. This is the grounding gap.",
    styles["Normal"]
))

curve_doc.build(curve_elements)
print(f"[DONE] Alignment curve report saved to {CURVE_PDF_OUTPUT}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/policy_documents/policy_keyword_scores_all.csv")

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

results = []

for doc, doc_df in df.groupby("document"):

    row = {"document": doc}

    for category, kws in categories.items():

        cat_scores = doc_df[doc_df["keyword"].isin(kws)]

        if len(cat_scores) == 0:
            row[category] = 0.0
        else:
            # average over ALL chunks AND ALL keywords
            row[category] = cat_scores["similarity"].mean()

    results.append(row)

category_df = pd.DataFrame(results)
category_df = category_df.set_index("document")
category_df['total'] = category_df.sum(axis=1)
category_df = category_df.sort_values('total', ascending=False).drop(columns=['total'])

plt.figure(figsize=(12, 14)) # Slightly wider for labels

# Use Seaborn for a cleaner look
# cmap="YlOrRd" is usually better for visibility than "Reds"
ax = sns.heatmap(
    category_df, 
    cmap="YlOrRd", 
    annot=False,       # Set to True if you want to see the actual numbers in the boxes
    linewidths=.1, 
    cbar_kws={'label': 'Mean Similarity Score'}
)

plt.title("AI Governance Dimensions: Policy Alignment Intensity", fontsize=16)
plt.xlabel("Governance Category", fontsize=12)
plt.ylabel("Policy Document", fontsize=12)

# Rotate labels for readability
plt.xticks(rotation=45, ha="right")
plt.yticks(fontsize=7)

plt.tight_layout()
plt.savefig("./data/results/policy_category_heatmap.png", dpi=150)
plt.show()



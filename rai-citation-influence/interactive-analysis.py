import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================

RAW_DATA_PATH = "./data/aies_papers.csv"
ENRICHED_DATA_PATH = "./data/aies_papers_enriched.csv"

# Load enriched file if it exists (so labels persist)
if os.path.exists(ENRICHED_DATA_PATH):
    df = pd.read_csv(ENRICHED_DATA_PATH)
else:
    df = pd.read_csv(RAW_DATA_PATH)

df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)
df["year"] = df["year"].astype(int)

# Ensure label column exists
if "label" not in df.columns:
    df["label"] = ""

df = df.sort_values(by="citation_count", ascending=False).reset_index(drop=True)

# ============================================================
# Precompute year-normalized statistics
# ============================================================

df["year_mean"] = df.groupby("year")["citation_count"].transform("mean")
df["year_std"] = df.groupby("year")["citation_count"].transform("std").replace(0, np.nan)
df["year_z"] = (df["citation_count"] - df["year_mean"]) / df["year_std"]

# ============================================================
# Helper Functions
# ============================================================

def print_paper(row):
    print("-" * 90)
    print(f"Index: {row.name}")
    print(f"Title: {row['title']}")
    print(f"Authors: {row['authors']}")
    print(f"Year: {row['year']}")
    print(f"Citations: {row['citation_count']}")
    print(f"Year-Z: {row['year_z']:.2f}")
    print(f"Label: {row.get('label', '')}")
    print(f"DOI: {row['doi']}")
    print(f"URL: {row['url']}")

def print_papers(subdf, n=10):
    for _, row in subdf.head(n).iterrows():
        print_paper(row)

def top_percentile(df, p):
    threshold = np.percentile(df["citation_count"], 100 - p)
    return df[df["citation_count"] >= threshold]

def top_year_normalized(df, z_threshold):
    return df[df["year_z"] >= z_threshold].sort_values(by="year_z", ascending=False)

def save_labels():
    df.to_csv(ENRICHED_DATA_PATH, index=False)
    print(f"\n[SAVED] Labels written to {ENRICHED_DATA_PATH}\n")

def label_paper(index):
    if index not in df.index:
        print("Invalid index.")
        return
    print_paper(df.loc[index])
    label = input("Enter label (or leave blank to cancel): ").strip()
    if label:
        df.at[index, "label"] = label
        save_labels()

# ============================================================
# Interactive Loop
# ============================================================

def main():
    print("\n========== RAI INTERACTIVE ANALYSIS ==========\n")
    print(f"Loaded {len(df)} papers.\n")

    while True:
        print("\nChoose an option:")
        print("1 → Show top N cited papers (raw)")
        print("2 → Show top 1%, 5%, or 10% papers (raw citations)")
        print("3 → Show year-breakout papers (by z-score)")
        print("4 → Show top papers from a specific year")
        print("5 → Search by keyword in title")
        print("6 → Show papers above year-normalized threshold")
        print("7 → Label a paper by index")
        print("8 → Exit")

        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            n = int(input("How many top papers? "))
            print_papers(df, n)

        elif choice == "2":
            p = int(input("Enter percentile (1, 5, or 10): "))
            subset = top_percentile(df, p)
            print(f"\nTop {p}% papers ({len(subset)} total):\n")
            print_papers(subset, min(20, len(subset)))

        elif choice == "3":
            z = float(input("Enter z-score threshold (default 2.0): ") or 2.0)
            subset = top_year_normalized(df, z)
            print(f"\nYear-normalized breakout papers (z ≥ {z}): {len(subset)} found\n")
            print_papers(subset, min(20, len(subset)))

        elif choice == "4":
            year = int(input("Enter year: "))
            subset = df[df["year"] == year].sort_values(by="citation_count", ascending=False)
            if subset.empty:
                print("No papers found for that year.")
            else:
                print_papers(subset, min(20, len(subset)))

        elif choice == "5":
            keyword = input("Enter keyword: ").lower()
            subset = df[df["title"].str.lower().str.contains(keyword, na=False)]
            print(f"\nFound {len(subset)} papers matching '{keyword}'\n")
            print_papers(subset, min(20, len(subset)))

        elif choice == "6":
            z = float(input("Enter year-normalized z threshold: "))
            subset = top_year_normalized(df, z)
            print(f"\nPapers with year_z ≥ {z}: {len(subset)} found\n")
            print_papers(subset, min(20, len(subset)))

        elif choice == "7":
            index = int(input("Enter paper index to label: "))
            label_paper(index)

        elif choice == "8":
            print("Exiting.")
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
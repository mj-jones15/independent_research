import pandas as pd
import numpy as np
import os

DATA_PATH = "./data/aies_papers.csv"

df = pd.read_csv(DATA_PATH)
df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0)
df["year"] = df["year"].astype(int)

df = df.sort_values(by="citation_count", ascending=False)

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def print_papers(subdf, n=10):
    for i, row in subdf.head(n).iterrows():
        print("-" * 80)
        print(f"Title: {row['title']}")
        print(f"Authors: {row['authors']}")
        print(f"Year: {row['year']}")
        print(f"Citations: {row['citation_count']}")
        print(f"DOI: {row['doi']}")
        print(f"URL: {row['url']}")

def top_percentile(df, p):
    threshold = np.percentile(df["citation_count"], 100 - p)
    return df[df["citation_count"] >= threshold]

def year_breakouts(df, z_threshold=2.0):
    df = df.copy()
    df["year_mean"] = df.groupby("year")["citation_count"].transform("mean")
    df["year_std"] = df.groupby("year")["citation_count"].transform("std").replace(0, np.nan)
    df["year_z"] = (df["citation_count"] - df["year_mean"]) / df["year_std"]
    return df[df["year_z"] >= z_threshold].sort_values(by="year_z", ascending=False)

# ------------------------------------------------------------
# Interactive loop
# ------------------------------------------------------------

def main():
    print("\n========== AIES INTERACTIVE ANALYSIS ==========\n")
    print(f"Loaded {len(df)} papers.\n")

    while True:
        print("\nChoose an option:")
        print("1 → Show top N cited papers")
        print("2 → Show top 1%, 5%, or 10% papers")
        print("3 → Show year-breakout papers (z ≥ 2)")
        print("4 → Show top papers from a specific year")
        print("5 → Search by keyword in title")
        print("6 → Exit")

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
            subset = year_breakouts(df)
            print(f"\nYear-breakout papers (z ≥ 2): {len(subset)} found\n")
            print_papers(subset, min(20, len(subset)))

        elif choice == "4":
            year = int(input("Enter year: "))
            subset = df[df["year"] == year]
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
            print("Exiting.")
            break

        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
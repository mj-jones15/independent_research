# Responsible AI Citation Influence Mapping

This project builds a literature review and analysis pipeline for analyzing the extent of alignment between regulation and RAI research literature.

The current implementation uses a literature review of the Artificial Intelligence, Ethics, and Society (AIES) conference and Neural Information Processing Systems (NeurIPS). The corpus of analyzed policy documents involves 18 national AI policies and ethical frameworks.

## Key Findings
- Egypt and UNESCO lead in research foundation as comprehensive ethics frameworks.
- China scores highest on RAI keyword density but has zero papers above the research consensus threshold — the largest gap between topical ambition and research grounding in the dataset.
- US policies cluster in the middle: broad enough to appear grounded, but with meaningful room for improvement.
- There is an evident three-way misalignment: fairness dominates RAI paper production, explainability dominates paper citations, but accountability dominates policy language.
- Sustainability is near-absent in both research production and citation impact, despite appearances in policy documents.

## Final Output
**The best entry point to this project is the poster.** It summarizes the full pipeline, findings, and visualizations in one place:

📄 [`poster/rai-translation-research-poster.pdf`](./poster/rai-translation-research-poster.pdf)

If you want to go deeper, the sections below describe the full codebase and the order in which everything runs.

---

## Repository Structure

```
.
├── poster/                          # Final deliverable — start here
│   └── rai-translation-research-poster.pdf
│
├── data/                                          #  Raw and processed corpora
│   ├── final_papers_with_keyword_similarities.csv # All research papers with computed similarity scores
│   └── policy_documents/                          # National AI policy documents (18 countries + UNESCO)
│   └── results/                                   # Final figures and CSV files for calculations post policy research alignment
│
├── bibtex/                          # BibTeX references for the literature review
├── graphs_and_buckets/              # Deprecated graphs and figures
├── semantic-scholar-fos/            # Field of Study (FoS) metadata from Semantic Scholar API
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## How to Navigate the Code

The scripts fall into four stages. If you want to understand the project quickly, **read the pipeline in order** — but you can jump straight to Stage 3 or 4 using the pre-processed data already in `data/`.

### Stage 1 — Data Collection
Pulls raw paper metadata and policy documents.

| Script | What it does |
|---|---|
| `data-ingestion.py` | Queries the Semantic Scholar API and OpenAlex API to retrieve AIES and NeurIPS paper records, respectively |
| `data-filtering.py` | Filters the AIES corpus by relevance, year range, and minimum citation thresholds. Built to analyze early findings of AIES RAI papers |
| `pdf_cleaning.py` | Extracts and cleans raw text from policy document PDFs |

### Stage 2 — Embedding
Generates vector representations for similarity comparisons.

| Script | What it does |
|---|---|
| `generate_embeddings.py` | Embeds research paper abstracts using `intfloat/e5-large-v2` |
| `embed-policy.py` | Embeds policy document chunks using the same model for apples-to-apples comparison |

### Stage 3 — Analysis
Core alignment and influence scoring.

| Script | What it does |
|---|---|
| `analyze_aies_similarity.py` | Identify similarity statistics to RAI keywords for AIES papers |
| `compute_neurips_enrichment.py` | Adds in enrichment statistics to NeurIPS papers |
| `policy-research-alignment.py` | Produces the overall alignment matrix across all policy documents and abstracts |

### Stage 4 — Visualization
Generates all figures found in the poster and the `data/results` folder.

| Script | What it does |
|---|---|
| `policy_labels.py` | Defines and loads human-readable labels for each policy document |
| `generate_combined_figures_from_results.py` | Generates figures for policy x research alignment |
| `generate_policy_heatmap.py` | Produces the policy × RAI-topic heatmap |
| `generate_research_figures.py` | Generates citation and embedding distribution plots |
| `preliminary-graph-creation.py` | Earlier exploratory graphs (kept for reproducibility) |

---

## Quickstart

```bash
pip install -r requirements.txt

# If starting from scratch:
python data-ingestion.py
python generate_embeddings.py
python embed-policy.py
python policy-research-alignment.py

# To reproduce the poster figures:
python generate_policy_heatmap.py
python generate_combined_figures_from_results.py

# For interactive exploration:
python interactive-analysis.py
```

> **Note:** Most intermediate outputs are already saved in `data/` and `graphs_and_buckets/`, so you can skip directly to Stage 4 without re-running the full pipeline.

---

## Research Motivation
This work supports investigation into how Responsible AI knowledge flows into governance and policy frameworks, including analysis of citation concentration and policy uptake (e.g., EU AI Act references).

---

## Primary References
1. Phoebe Koundouri et al. (2025). "Do SDGs Support Human Security? A Machine Learning Analysis with Policy Recommendations," DEOS Working Papers 2538, Athens University of Economics and Business. https://ideas.repec.org/p/aue/wpaper/2538.html

2. Ali Akbar Septiandri, Marios Constantinides & Daniele Quercia (2024). "The Impact of Responsible AI Research on Innovation and Development." https://arxiv.org/abs/2407.15647
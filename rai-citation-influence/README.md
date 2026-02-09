# Responsible AI Citation Influence Mapping

This project builds a reproducible bibliometric pipeline for analyzing which Responsible AI research papers exert disproportionate influence on academic and policy ecosystems.

The current implementation focuses on the AAAI/ACM Artificial Intelligence, Ethics, and Society (AIES) conference (2018–2025), with planned extensions to NeurIPS, ICML, FAccT, and related venues.

## Key Features
- Ground-truth DOI corpus construction from conference proceedings
- Semantic Scholar enrichment for citation metadata
- Year-normalized citation analysis
- Heavy-tail and inequality modeling (Lorenz curves, Gini coefficient)
- Breakout paper detection
- Interactive analysis tooling

## Research Motivation
This work supports investigation into how Responsible AI knowledge flows into governance and policy frameworks, including analysis of citation concentration and policy uptake (e.g., EU AI Act references).

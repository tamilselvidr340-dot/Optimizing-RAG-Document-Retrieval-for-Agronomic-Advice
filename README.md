# Optimizing RAG Document Retrieval for Agronomic Advice

Retrieval engine for a Retrieval-Augmented Generation (RAG) system that
connects smallholder farmers' plain-language questions to the right
agricultural extension documents — covering crop diseases, pests, nutrient
deficiencies, soil management, fertilizers, and climate adaptation.

Built for the Kaggle competition
[[agricultural-extension-rag-smart-retrieval-for-farmers]](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)
(host: TRI AI), as part of AI Saturdays Lagos Cohort 10.

## Overview

Farmers describe problems in everyday language ("Why are my maize leaves
turning yellow?") while extension documents use technical terminology
("nitrogen chlorosis"). This vocabulary mismatch causes keyword-based
retrieval to miss or misrank relevant documents. This project implements a
hybrid retrieval pipeline — BM25 fused with sentence-transformer dense
embeddings, followed by cross-encoder reranking  and evaluates it against
the competition's provided TF-IDF baseline to close that gap and improve
ranking quality.

## Dataset

695 agricultural extension factsheets (`documents.csv`) covering
Sub-Saharan Africa, paired with farmer queries (`train_queries.csv`,
`test_queries.csv`) and expert graded relevance judgments (`qrels_train.csv`,
0–3 scale, including hard negatives). Supplied in full by the competition
host — not scraped or collected by the team. Full details, labeling process,
and bias/representation limitations (English-only, uneven topic coverage,
fixed ground truth) are documented in `docs/Challenge_2_Data_Card.pdf`.

The raw data is not redistributed here per competition terms — see
[`data/README.md`](data/README.md) for download instructions.

## Training / Retrieval Pipeline

**Baseline (provided by competition):** TF-IDF vector retrieval, ~0.55 nDCG@5.

**Implemented method** (`src/pipeline.py`, `HybridRetriever`) — two stages:

1. **Stage 1 — Hybrid candidate retrieval.** BM25 (`k1=1.5, b=0.75`) scores
   the corpus lexically; `sentence-transformers/all-MiniLM-L6-v2` scores it
   semantically via cosine similarity. Both score vectors are min-max
   normalized and fused: `hybrid_score = 0.55 * dense + 0.45 * BM25`,
   weighted toward dense embeddings to bridge everyday-language queries and
   technical document phrasing. Top 25 candidates advance.
2. **Stage 2 — Cross-encoder reranking.** `cross-encoder/ms-marco-MiniLM-L-6-v2`
   scores each (query, candidate) pair directly and reorders the 25
   candidates down to the final top-5.

Document text is enriched with crop/country metadata before indexing
(`[crop | country] title. body`) to help disambiguate similar-sounding
issues across crops/regions — including hard-negative cases like potassium
vs. nitrogen deficiency.

## Evaluation

**Metric:** nDCG@5, computed against graded relevance judgments.
**Method:** each retrieval method compared directly against the fixed
TF-IDF baseline on the same query set.

**Validation Mean nDCG@5: 0.8236** (baseline ≈ 0.55) — see
[`results/`](results/) for the ranked prediction file this score was
computed from.

This is a retrieval-quality benchmark only; the system retrieves candidate
documents and does not generate agronomic advice (out of scope — see
`docs/Challenge_1_Problem_Statement.docx`, Section 6).

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download competition data into data/ (see data/README.md)
kaggle competitions download -c agricultural-extension-rag-smart-retrieval-for-farmers
unzip agricultural-extension-rag-smart-retrieval-for-farmers.zip -d data/

# 3. Evaluate on the labeled training split (prints Validation Mean nDCG@5)
python src/evaluate.py --data_dir data/

# 4. Generate the final ranked submission for the hidden test queries
python src/generate_submission.py --data_dir data/ --out results/submission.csv
```

GPU recommended (auto-falls back to CPU). See [`src/README.md`](src/README.md)
for script details, and `notebooks/` for the original single-cell Kaggle
version this was refactored from.

## Repository Structure

```
├── README.md
├── requirements.txt
├── docs/           # Four Cohort Challenges (see below)
├── data/           # Download instructions (raw data not redistributed)
├── notebooks/       # Exploratory / development notebooks
├── src/            # Reproduction pipeline scripts
└── results/        # Validation predictions + evaluation output
```

## Cohort Challenges

- [Challenge 1 — Problem Statement](Structured%20Research%20Problem%20Statement%20(TANGANYIKA%20GROUP).docx)
- [Challenge 2 — Data Card](Tanganyika%20Datacard%20for%20RAG%20optimization.pdf)
- [Challenge 3 — Impact Statement](Challenge_3_Impact_Statement_Optimising_RAG_Agronomic_Advice.pdf)
- [Challenge 4 — Stakeholder Engagement Plan](Challenge_4_Stakeholder_Engagement_Plan_Agronomic_RAG.pdf)

## Values & Responsible AI

Guided by accessibility, fairness, accuracy, transparency, inclusivity,
reliability, and sustainability. Known limitations — English-only content,
uneven topic coverage, and a fixed non-community-validated dataset — are
documented rather than treated as resolved. Results here reflect
retrieval-method benchmarking on a fixed competition dataset, **not**
evidence of readiness for real-world farmer deployment; see
`docs/Challenge_3_Impact_Statement.pdf` and
`docs/Challenge_4_Stakeholder_Engagement_Plan.pdf` for full discussion.

## Appendix: Contributors & Mentors

| Name | Role |
|---|---|
| _Add name_ | Team Lead |
| _Add name_ | Team Member |
| _Add name_ | Team Member |
| _Add name_ | Mentor |

Program: Tri AI Saturdays Lagos — .

## References

1. Kaggle competition: [agricultural-extension-rag-smart-retrieval-for-farmers](https://www.kaggle.com/competitions/agricultural-extension-rag-smart-retrieval-for-farmers)
2. Development notebook: https://www.kaggle.com/code/cosmaskungu/notebookf496f3bb89

# RAG Failure-Mode Diagnostic Suite

**Adversarial stress-testing framework for Retrieval-Augmented Generation systems**

> Status: 🚧 Actively under development. This README will be expanded with full results, figures, and usage instructions as the project progresses.

## What this is

Most RAG evaluations report a single aggregate accuracy score. That number doesn't say *why* a pipeline failed — whether the retriever pulled the wrong context, or the generator ignored the right context anyway.

This project builds a diagnostic suite that answers that question directly: given a RAG pipeline and a controlled taxonomy of failure-inducing perturbations (contradictory passages, lexical-similarity distractors, temporally stale documents, multi-hop reasoning gaps, and topically similar noise), it measures **where** each pipeline breaks, not just **whether** it breaks.

## Project structure

```
RAG-Diagnostic-Suite/
├── configs/                # Experiment configuration files (models, k, perturbation rates)
├── data/
│   └── datasets/           # Cached dataset downloads (gitignored — regenerated from notebooks)
├── retrievers/              # Retrieval pipeline implementations (naive, reranked, HyDE, reflection)
├── evaluation/
│   └── metrics/             # EM, F1, retrieval recall, hallucination rate, failure attribution
├── notebooks/                # Exploratory and orchestration notebooks (one per phase)
├── experiments/               # Experiment run scripts / saved configs per run
├── results/
│   └── visualizations/        # Generated figures (gitignored except curated final outputs)
├── utils/                      # Shared helper functions
├── tests/                      # Unit tests for perturbation generators and metrics
├── requirements.txt
├── environment.yml
└── LICENSE
```

## Datasets

- [HotpotQA](https://hotpotqa.github.io/) — multi-hop QA with gold supporting facts
- [Natural Questions](https://ai.google.com/research/NaturalQuestions) — single-hop, real search queries grounded in Wikipedia

## Installation

### Option A: pip
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Option B: conda
```bash
conda env create -f environment.yml
conda activate rag-diagnostic-suite
```

Then launch Jupyter:
```bash
jupyter notebook
```

## Roadmap

- [x] Repository scaffolding
- [ ] Dataset loading and schema exploration
- [ ] Perturbation / stress-test suite
- [ ] FAISS vector index
- [ ] Baseline RAG pipeline
- [ ] Pipeline variants (reranking, HyDE, reflection)
- [ ] Evaluation and failure attribution
- [ ] Results and visualizations
- [ ] Full write-up

## License

MIT — see [LICENSE](LICENSE).

# RAG Failure-Mode Diagnostic Suite

A stress-test harness for Retrieval-Augmented Generation pipelines, built to answer one question precisely: **when RAG fails, where does it fail, and does a given architectural fix actually address the failure it claims to?**

Rather than reporting a single aggregate accuracy number, this project injects five distinct, controlled failure modes into a HotpotQA-derived corpus and measures how four different RAG pipeline variants — baseline dense retrieval, cross-encoder reranking, HyDE, and self-critique reflection — hold up against each one individually.

---

## Motivation

Most RAG demos report one number on one dataset under ideal conditions. That number says almost nothing about how the system behaves when retrieval is contested, when documents are stale, when reasoning requires two hops, or when evidence conflicts — which is where production RAG systems actually break. This project treats those failure conditions as the thing being measured, not noise to average away, and treats "does technique X fix problem Y" as a testable claim rather than an assumption.

## Methodology

**Dataset.** 1,000 examples sampled from HotpotQA (a multi-hop QA dataset with gold "supporting fact" passages), giving every question a ground-truth answer and a known set of passages required to answer it correctly.

**Perturbation suite.** Each example is deterministically transformed by one of five generators (`perturbations/generators.py`), simulating a distinct retrieval/generation failure mode:

| Perturbation | Mechanism | Tests whether the pipeline... |
|---|---|---|
| `contradiction` | Appends a sentence to a gold passage that contradicts the true answer | ...can arbitrate between conflicting evidence rather than pattern-matching the first mention |
| `lexical_distractor` | Replaces the least-relevant non-gold passage with one sharing high keyword overlap with the question but answering a different one | ...retrieves on genuine semantic relevance vs. surface keyword overlap |
| `stale_document` | Marks a gold passage as an outdated record with a superseded claim appended | ...is vulnerable to temporally stale-but-confident information |
| `multihop_gap` | Deletes one of two-or-more required gold passages outright | ...correctly fails/abstains on a broken reasoning chain rather than hallucinating |
| `topical_noise` | Replaces the least-relevant non-gold passage with a random, topically-unrelated passage from elsewhere in the corpus | ...is robust to plausible-looking but irrelevant clutter |

**Retrieval.** Dense retrieval via `all-MiniLM-L6-v2` sentence embeddings + FAISS, top-k=3, over each example's local ~10-passage candidate pool.

**Generation.** `flan-t5-small`, given the top-k retrieved passages as context.

**Metrics.** Exact Match (EM), token-level F1, and retrieval recall (fraction of gold supporting-fact titles present in the top-k retrieved set), computed per-example and aggregated by perturbation type.

### Design iteration: perturbation injection (append → replace)

The `lexical_distractor` and `topical_noise` generators originally **appended** the injected passage as an extra (11th) candidate onto the retrieval pool. A diagnostic check on the resulting stress-test suite found that the injected passage entered the top-3 retrieved set in only **1.2% of cases (5/400)** — the perturbation was, empirically, close to a no-op, because a single extra passage rarely displaces any of the ~10 already in the pool.

**Fix:** both generators now **replace** the least-relevant existing non-gold passage instead of appending, guaranteeing the injected passage a competitive slot in the candidate pool while never evicting a gold passage (keeping the question solvable in principle). Re-running the same diagnostic post-fix confirmed the injected passage now genuinely contests a slot (see Finding 2 below) — the fix worked as a mechanism, even though it changed the mechanism without dramatically changing the retrieval-recall outcome (see Finding 1).

This iteration — catching an instrument that wasn't measuring what it claimed to, and confirming the fix with a second diagnostic — is treated here as a first-class part of the project, not a bug footnote.

---

## Pipelines tested

| # | Pipeline | Mechanism |
|---|---|---|
| 1 | **Baseline** | Dense bi-encoder retrieval → generation, no additional processing |
| 2 | **Reranking** | Bi-encoder retrieves top-10 → cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks to top-3 → generation |
| 3 | **HyDE** | Generator produces a hypothetical answer passage → that hypothetical document (not the raw question) is embedded and used for retrieval → generation |
| 4 | **Self-critique reflection** | Generate an answer → ask the generator to judge whether its own answer is supported by the retrieved context → abstain if unsupported |

---

## Results

Exact Match / F1 / Retrieval Recall by perturbation type, full 1,000-example runs (baseline, reranking, HyDE):

| Perturbation | Baseline EM/F1/Recall | Reranked EM/F1/Recall | HyDE EM/F1/Recall |
|---|---|---|---|
| contradiction | .190 / .265 / .618 | .230 / .314 / **.695** | .190 / .262 / .530 |
| lexical_distractor | .245 / .323 / .614 | **.260** / **.352** / **.695** | .210 / .282 / .511 |
| multihop_gap | .175 / .213 / .326 | .180 / .245 / .366 | .155 / .212 / .264 |
| stale_document | .225 / .303 / .604 | **.265** / **.343** / **.693** | .215 / .283 / .515 |
| topical_noise | .255 / .342 / .621 | **.270** / **.366** / **.704** | .195 / .269 / .512 |

*Self-critique reflection (pipeline 4) was evaluated via targeted spot-check rather than a full 1,000-example run — see Finding 4 for why.*

![Pipeline comparison](results/final_comparison.png)

---

## Key findings

**Finding 1 — Reranking is the clear winner, and its gains are concentrated where the base retriever was weakest, not uniform.**
Cross-encoder reranking improved retrieval recall by +4 to +9 points across every single perturbation type, and EM/F1 improved similarly, though more modestly (generation quality, not just retrieval, is a limiting factor). Notably, recall gains were largest on `contradiction` and `stale_document` (+7-9pp) — perturbations that modify an *existing* gold passage rather than inject a competing one — suggesting the cross-encoder is a stronger relevance model overall (trained on query-passage relevance judgments via MS MARCO), not merely a distractor-suppressor.

**Finding 2 — Dense retrieval is intrinsically robust to lexical-overlap distractors; reranking sharpens this further.**
Post-fix rank diagnostics showed the bi-encoder buries an injected lexical distractor at median rank 9/10 even when it shares a mean of ~4 keywords with the question (81% land in the bottom 3 of ~10 candidates); pure topical noise fares even worse for the distractor (98.5% bottom-3). Reranking improves on this specifically for lexical distractors — cutting top-3 intrusion from ~19% to 2% — while barely moving the already-near-ceiling topical_noise case (98.5% → 99%). This is a clean, gradient result: reranking helps in direct proportion to where the base retriever had a correctable weakness, not uniformly.

**Finding 3 — HyDE underperforms both baseline and reranking on every metric, across every perturbation type, and the reason is a real, citable limitation, not a bug.**
Despite resolving one specific ambiguous-entity retrieval confusion in a qualitative spot-check (embedding a declarative hallucinated answer landed closer to the correct biography passage than the raw interrogative question did), HyDE's aggregate retrieval recall (0.51-0.53) trails baseline (0.60-0.62) and reranking (0.69-0.70) across the board. `flan-t5-small` is small and produces frequently unreliable hypothetical documents when generating without context — the noise from bad hallucinations outweighs the benefit of declarative-style query reformulation. This is consistent with HyDE's original evaluation setting, which used substantially larger generators than the one used here.

**Finding 4 — Self-critique reflection is unreliable with a small generator, for the same underlying reason as HyDE, and was not scaled to a full run because of it.**
A 5-example spot-check of the critique step's raw (unparsed) output showed the model judging its own `'unanswerable'` response as "unsupported" (a category error — there's nothing to support or refute), and in one case ignoring the yes/no instruction format entirely, generating an unrelated hallucinated string instead of a verdict. The critique step does not reliably discriminate supported from unsupported answers at this model scale. Rather than run the full 1,000-example evaluation on a mechanism already visibly broken at the example level, the suite was stopped at the diagnostic stage — a deliberate methodological choice, not an oversight.

**Finding 5 — `multihop_gap` is a structural ceiling that no tested technique meaningfully closes.**
Retrieval recall for `multihop_gap` sits at 0.33-0.37 across baseline and reranking — the worst of all five perturbation types by a wide margin, and the *least*-improved by reranking (+4.0pp, smallest gain of any perturbation) or HyDE (which makes it worse, 0.264). This is expected and instructive: reranking and HyDE both operate on ranking/embedding *existing* candidates, and neither can reconstruct a passage that was deleted from the pool outright. Closing this gap would require a fundamentally different mechanism — e.g. query decomposition into sub-questions with independent retrieval per hop — which is out of scope for this suite but a natural next research direction.

---

## Limitations

- **Small generator (`flan-t5-small`, ~80M params)** is a bottleneck for any technique requiring generation-quality reasoning (HyDE's hypothetical documents, self-critique's verdicts). Results for those two pipelines specifically should be read as "how these techniques perform with a small generator," not as a general verdict on the techniques themselves.
- **Unseeded perturbation sampling** (`topical_noise`'s donor-passage selection) means results are stable within roughly ±1-2 percentage points across reruns from a fresh kernel, rather than bit-for-bit reproducible.
- **Small local candidate pools (~10 passages/example)** mean k=3 retrieval operates in a easier regime than a production corpus with thousands of candidates; absolute recall numbers likely don't generalize to larger-scale retrieval, though the *relative* comparison between pipelines should.
- **Self-critique reflection** has no full-suite quantitative results, by design (see Finding 4) — only a qualitative spot-check.

## Possible future extensions

- Query decomposition / iterative multi-hop retrieval, targeting the `multihop_gap` ceiling directly.
- Re-running HyDE and self-critique with a larger generator, to test whether the negative results are generator-scale artifacts or more fundamental.
- Hybrid sparse (BM25) + dense retrieval, since lexical distractors are hypothesized to matter more for sparse retrievers than the dense retriever used here.

---

## Repo structure

```
├── notebooks/
│   └── 01_dataset_exploration.ipynb   # full pipeline: data loading, perturbation generation,
│                                       # retrieval, generation, evaluation, all 4 pipeline variants
├── perturbations/
│   └── generators.py                  # the 5 perturbation generator functions
├── results/
│   ├── baseline_results.csv
│   ├── reranked_results.csv
│   ├── hyde_results.csv
│   ├── hyde_sample_docs.csv           # sample hypothetical documents (qualitative evidence)
│   ├── baseline_by_perturbation.png
│   ├── reranking_delta.png
│   └── final_comparison.png
├── requirements.txt
└── README.md
```

## Setup & running

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/01_dataset_exploration.ipynb` and run all cells top to bottom. Full baseline/reranked/HyDE evaluation runs take roughly 10-20 minutes each on CPU.

"""
Perturbation generators for the RAG Failure-Mode Diagnostic Suite.

Each function takes a HotpotQA-style example (dict with 'question', 'answer',
'supporting_facts', 'context') plus a pool of other examples to draw noise
from, and returns a NEW example dict with one failure mode injected into
its 'context' field. The original example is never mutated in place.
"""

from __future__ import annotations
import random
import re
from typing import Any


def _flatten_context(example: dict) -> list[tuple[str, str]]:
    """Convert HotpotQA's parallel-list context format into (title, text) pairs."""
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    return [(t, "".join(s)) for t, s in zip(titles, sentences)]


def _gold_titles(example: dict) -> set[str]:
    """Return the set of passage titles required to answer the question."""
    return set(example["supporting_facts"]["title"])


def _non_gold_indices(example: dict, gold_titles: set[str]) -> list[int]:
    """Indices of passages in this example's context that are NOT gold-required."""
    return [i for i, t in enumerate(example["context"]["title"]) if t not in gold_titles]


def _least_relevant_non_gold_index(example: dict, gold_titles: set[str], question_keywords: set[str]) -> int | None:
    """
    Among non-gold passages, return the index with the LOWEST keyword overlap
    with the question — i.e. the passage that's already contributing least,
    and therefore the most defensible slot to evict when injecting a
    perturbation. Returns None if there are no non-gold passages to evict.
    """
    candidates = _non_gold_indices(example, gold_titles)
    if not candidates:
        return None

    context = example["context"]
    scored = []
    for i in candidates:
        text = "".join(context["sentences"][i])
        passage_keywords = set(re.findall(r"\w+", text.lower()))
        overlap = len(question_keywords & passage_keywords)
        scored.append((overlap, i))

    scored.sort(key=lambda pair: pair[0])  # lowest overlap first
    return scored[0][1]


def inject_contradiction(example: dict, **_) -> dict:
    """
    Failure mode: CONTRADICTION.
    Takes a gold passage and appends a sentence directly contradicting the
    answer (e.g. flips a yes/no claim, or negates a stated fact). Tests
    whether the generator can arbitrate between conflicting evidence
    instead of just pattern-matching to the first mention.
    """
    new_example = _deep_copy_example(example)
    gold_titles = _gold_titles(new_example)
    context = new_example["context"]

    for i, title in enumerate(context["title"]):
        if title in gold_titles:
            original_text = "".join(context["sentences"][i])
            contradiction = _make_contradiction_sentence(new_example["question"], new_example["answer"])
            context["sentences"][i] = context["sentences"][i] + [" " + contradiction]
            break  # only contradict one gold passage, to keep the task solvable in principle

    new_example["perturbation_type"] = "contradiction"
    return new_example


def _make_contradiction_sentence(question: str, answer: str) -> str:
    """Very simple templated contradiction — good enough to stress-test retrieval/generation."""
    if answer.strip().lower() in ("yes", "no"):
        flipped = "no" if answer.strip().lower() == "yes" else "yes"
        return f"Contrary to some accounts, the correct answer to '{question}' is actually {flipped}."
    return f"Note: some sources dispute this, and claim the answer is not '{answer}' but something else entirely."


def inject_lexical_distractor(example: dict, corpus_pool: list[dict], **_) -> dict:
    """
    Failure mode: LEXICAL SIMILARITY DISTRACTOR.
    Finds a passage from elsewhere in the corpus that shares many keywords
    with the question but answers a DIFFERENT question, and REPLACES the
    least-relevant existing non-gold passage in the pool with it.

    NOTE (design iteration): this used to *append* the distractor as an
    extra candidate. Empirically, that intrusion rate into top-3 retrieval
    was ~1.2% at k=3 with a ~10-passage pool — the appended passage almost
    never displaced anything, making the perturbation a near no-op. We now
    replace the weakest-relevance non-gold passage instead, guaranteeing
    the distractor actually competes for a retrieval slot, while still
    never evicting a gold passage (so the task stays solvable in principle).
    """
    new_example = _deep_copy_example(example)
    gold_titles = _gold_titles(new_example)
    question_keywords = set(re.findall(r"\w+", new_example["question"].lower())) - _STOPWORDS

    best_passage = None
    best_overlap = -1
    for candidate in corpus_pool:
        if candidate["id"] == new_example["id"]:
            continue
        for title, text in _flatten_context(candidate):
            passage_keywords = set(re.findall(r"\w+", text.lower()))
            overlap = len(question_keywords & passage_keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_passage = (title, text)

    if best_passage:
        title, text = best_passage
        context = new_example["context"]
        evict_idx = _least_relevant_non_gold_index(new_example, gold_titles, question_keywords)

        if evict_idx is not None:
            # Replace the weakest non-gold passage so the distractor is
            # guaranteed a competitive slot in the candidate pool.
            context["title"][evict_idx] = title + " (lexical distractor)"
            context["sentences"][evict_idx] = [text]
        else:
            # Edge case: every passage in this example is gold-required
            # (nothing safe to evict). Fall back to appending rather than
            # risk breaking the question.
            context["title"].append(title + " (lexical distractor)")
            context["sentences"].append([text])

    new_example["perturbation_type"] = "lexical_distractor"
    new_example["lexical_distractor_overlap_score"] = best_overlap
    return new_example


def inject_stale_document(example: dict, **_) -> dict:
    """
    Failure mode: TEMPORAL STALENESS.
    Prepends an outdated-sounding framing to a gold passage and appends a
    stale/superseded claim. We approximate real temporal drift (we don't
    have historical Wikipedia snapshots) by explicitly marking a passage as
    an older revision with a contradicting older fact.
    """
    new_example = _deep_copy_example(example)
    gold_titles = _gold_titles(new_example)
    context = new_example["context"]

    for i, title in enumerate(context["title"]):
        if title in gold_titles:
            stale_note = (
                f" [Note: as of an earlier record, this information was reported differently "
                f"and has since been updated or corrected.]"
            )
            context["sentences"][i] = context["sentences"][i] + [stale_note]
            context["title"][i] = title + " (2010 archive)"
            break

    new_example["perturbation_type"] = "stale_document"
    return new_example


def inject_multihop_gap(example: dict, **_) -> dict:
    """
    Failure mode: MULTI-HOP REASONING GAP.
    Removes ONE of the gold supporting passages required for multi-hop
    questions, leaving the other(s) intact. Tests whether the pipeline
    correctly fails to answer (or flags uncertainty) rather than
    hallucinating a confident answer from a broken reasoning chain.
    Only meaningful for questions with 2+ gold passages.
    """
    new_example = _deep_copy_example(example)
    gold_titles = list(_gold_titles(new_example))

    if len(gold_titles) < 2:
        new_example["perturbation_type"] = "multihop_gap_not_applicable"
        return new_example

    removed_title = gold_titles[0]
    context = new_example["context"]
    keep_indices = [i for i, t in enumerate(context["title"]) if t != removed_title]

    new_example["context"] = {
        "title": [context["title"][i] for i in keep_indices],
        "sentences": [context["sentences"][i] for i in keep_indices],
    }
    new_example["perturbation_type"] = "multihop_gap"
    new_example["removed_supporting_title"] = removed_title
    return new_example


def inject_topical_noise(example: dict, corpus_pool: list[dict], rng: random.Random, **_) -> dict:
    """
    Failure mode: IRRELEVANT BUT TOPICALLY SIMILAR NOISE.
    Pulls a random passage from a DIFFERENT example (same overall domain —
    Wikipedia — but genuinely unrelated to this question) and REPLACES the
    least-relevant existing non-gold passage with it.

    NOTE (design iteration): same fix as inject_lexical_distractor — this
    used to append the noise passage as an extra (11th) candidate, which
    measured empirically at ~1.2% top-3 intrusion rate and made the
    perturbation a near no-op at k=3. We now replace a non-gold passage
    instead, so the noise passage actually competes for a retrieval slot.
    We still pick the evicted passage randomly (via rng) rather than by
    relevance, since topical noise is meant to simulate arbitrary clutter,
    not a targeted keyword attack.
    """
    new_example = _deep_copy_example(example)
    gold_titles = _gold_titles(new_example)

    other_examples = [e for e in corpus_pool if e["id"] != new_example["id"]]
    donor = rng.choice(other_examples)
    donor_passages = _flatten_context(donor)
    title, text = rng.choice(donor_passages)

    context = new_example["context"]
    non_gold = _non_gold_indices(new_example, gold_titles)

    if non_gold:
        evict_idx = rng.choice(non_gold)
        context["title"][evict_idx] = title + " (noise)"
        context["sentences"][evict_idx] = [text]
    else:
        # Edge case: nothing safe to evict — fall back to appending.
        context["title"].append(title + " (noise)")
        context["sentences"].append([text])

    new_example["perturbation_type"] = "topical_noise"
    return new_example


def _deep_copy_example(example: dict) -> dict:
    """Shallow-safe deep copy for our nested dict/list structure (avoids mutating the source dataset)."""
    import copy
    return copy.deepcopy(example)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "and",
    "to", "for", "did", "do", "does", "what", "who", "when", "where", "which",
    "how", "with", "by", "as", "it", "that", "this", "be", "been",
}

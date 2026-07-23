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
    with the question but answers a DIFFERENT question, and inserts it.
    Tests whether the retriever is doing genuine semantic matching or is
    fooled by surface-level keyword overlap.
    """
    new_example = _deep_copy_example(example)
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
        new_example["context"]["title"].append(title + " (lexical distractor)")
        new_example["context"]["sentences"].append([text])

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
    Wikipedia — but genuinely unrelated to this question) and inserts it.
    Tests robustness to plausible-looking but useless clutter, as opposed
    to lexical distractors which specifically target keyword overlap.
    """
    new_example = _deep_copy_example(example)
    other_examples = [e for e in corpus_pool if e["id"] != new_example["id"]]
    donor = rng.choice(other_examples)
    donor_passages = _flatten_context(donor)
    title, text = rng.choice(donor_passages)

    new_example["context"]["title"].append(title + " (noise)")
    new_example["context"]["sentences"].append([text])

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

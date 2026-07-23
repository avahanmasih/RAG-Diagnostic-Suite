"""Standard extractive-QA metrics: normalized exact match and token-level F1."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace — standard SQuAD-style normalization."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> int:
    """1 if normalized strings match exactly, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer (standard SQuAD metric)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_recall(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Fraction of gold supporting-fact titles that appear among the retrieved passage titles."""
    if not gold_titles:
        return 1.0
    retrieved_set = set(t.split(" (")[0] for t in retrieved_titles)  # strip our " (noise)"/" (2010 archive)" suffixes
    hits = sum(1 for t in gold_titles if t in retrieved_set)
    return hits / len(gold_titles)
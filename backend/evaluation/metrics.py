"""ASR and classification metrics with no third-party dependencies.

WER and CER use Levenshtein alignment. Normalisation lower-cases and strips
punctuation so that "Ngiyabonga." and "ngiyabonga" match; it does NOT
convert digits to words, so a digits-versus-words convention mismatch (as
seen in the AST baseline) still counts as errors and should be reported
separately.
"""

import math
import re
from collections import Counter


def normalise_transcript(text):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_ops(reference, hypothesis):
    """Return (substitutions, deletions, insertions) between two sequences."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    # dp[i][j] = (cost, subs, dels, ins)
    previous = [(j, 0, 0, j) for j in range(cols)]
    for i in range(1, rows):
        current = [(i, 0, i, 0)]
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            dele = previous[j]
            ins = current[j - 1]
            best = min(
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
            )
            current.append(best)
        previous = current
    _, subs, dels, ins = previous[-1]
    return subs, dels, ins


def _rate(ref_units, hyp_units):
    subs, dels, ins = edit_ops(ref_units, hyp_units)
    errors = subs + dels + ins
    n = len(ref_units)
    return {
        "rate": (errors / n) if n else (0.0 if not hyp_units else math.inf),
        "errors": errors,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "reference_length": n,
    }


def wer(reference, hypothesis, normalise=True):
    if normalise:
        reference, hypothesis = normalise_transcript(reference), normalise_transcript(hypothesis)
    return _rate(reference.split(), hypothesis.split())


def cer(reference, hypothesis, normalise=True):
    if normalise:
        reference, hypothesis = normalise_transcript(reference), normalise_transcript(hypothesis)
    return _rate(list(reference.replace(" ", "")), list(hypothesis.replace(" ", "")))


def corpus_error_rate(pairs, unit="word"):
    """Corpus-level WER/CER: total edits over total reference length."""
    func = wer if unit == "word" else cer
    errors = length = 0
    for reference, hypothesis in pairs:
        result = func(reference, hypothesis)
        errors += result["errors"]
        length += result["reference_length"]
    return errors / length if length else None


def classification_report(y_true, y_pred, labels=None):
    """Accuracy, macro-F1 and per-class precision/recall/F1.

    Macro-F1 averages over `labels` if given, else over every label seen in
    either the gold or the predicted column (sklearn's default behaviour).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    labels = list(labels) if labels else sorted(set(y_true) | set(y_pred))
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, pred in zip(y_true, y_pred):
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    per_class = {}
    for label in labels:
        p_den, r_den = tp[label] + fp[label], tp[label] + fn[label]
        precision = tp[label] / p_den if p_den else 0.0
        recall = tp[label] / r_den if r_den else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": r_den}
    n = len(y_true)
    return {
        "n": n,
        "accuracy": (sum(tp.values()) / n) if n else None,
        "macro_f1": (sum(c["f1"] for c in per_class.values()) / len(per_class)) if per_class else None,
        "per_class": per_class,
    }


def mcnemar_exact(a_correct, b_correct):
    """Exact two-sided McNemar test on paired correctness vectors.

    Returns the discordant counts (b01 = only B right, b10 = only A right) and
    the binomial p-value. Use for RQ4 with identical inputs to both branches.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("paired vectors differ in length")
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = only_a + only_b
    if n == 0:
        return {"only_a_correct": 0, "only_b_correct": 0, "p_value": 1.0, "n_pairs": len(a_correct)}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {
        "only_a_correct": only_a,
        "only_b_correct": only_b,
        "p_value": min(1.0, 2 * tail),
        "n_pairs": len(a_correct),
    }


def latency_summary(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None

    def pct(q):
        index = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
        return values[index]

    return {
        "n": len(values),
        "mean_ms": sum(values) / len(values),
        "median_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "max_ms": values[-1],
    }

"""Summarise a research log into the dissertation's measures.

    python -m backend.evaluation.report data/logs/research_log.csv [--include-dev] [--json]

Rows produced by stand-in engines (dev-rules, nllb-rules) are excluded
unless --include-dev is passed, so stand-in output cannot leak into results.
Every figure printed is computed from the log; nothing is estimated.
"""

import argparse
import json
from collections import defaultdict

from backend.evaluation.logger import read_rows
from backend.evaluation.metrics import (
    classification_report,
    corpus_error_rate,
    latency_summary,
    mcnemar_exact,
)
from backend.nlu.dev_rules import STAND_IN_ENGINES
from backend.nlu.intents import intent_labels


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_dev(row):
    return row.get("branch_engine") in STAND_IN_ENGINES


def summarise(rows, include_dev=False):
    excluded = 0 if include_dev else sum(1 for r in rows if _is_dev(r))
    if not include_dev:
        rows = [r for r in rows if not _is_dev(r)]

    report = {"rows": len(rows), "excluded_dev_rows": excluded, "branches": {}, "asr": None, "paired": None}

    # --- ASR (one row per trial; take branch rows deduplicated by trial) ---
    trials = {}
    for row in rows:
        trials.setdefault((row["session_id"], row["trial_id"]), row)
    asr_pairs = [
        (r["reference_transcript"], r["asr_transcript"])
        for r in trials.values()
        if r.get("reference_transcript") and r.get("source") == "audio"
    ]
    if asr_pairs:
        report["asr"] = {
            "n_utterances": len(asr_pairs),
            "corpus_wer": corpus_error_rate(asr_pairs, "word"),
            "corpus_cer": corpus_error_rate(asr_pairs, "char"),
            "asr_latency": latency_summary(_float(r["asr_latency_ms"]) for r in trials.values()),
        }

    # --- Per branch --------------------------------------------------------
    by_branch = defaultdict(list)
    for row in rows:
        by_branch[row["branch"]].append(row)
    labels = intent_labels()
    for branch, branch_rows in sorted(by_branch.items()):
        gold = [r for r in branch_rows if r.get("true_intent")]
        entry = {
            "engines": sorted({r["branch_engine"] for r in branch_rows}),
            "n_rows": len(branch_rows),
            "n_labelled": len(gold),
            "classification_latency": latency_summary(_float(r["classification_latency_ms"]) for r in branch_rows),
            "translation_latency": latency_summary(_float(r["translation_latency_ms"]) for r in branch_rows),
            "total_latency": latency_summary(_float(r["total_latency_ms"]) for r in branch_rows),
        }
        if gold:
            seen = sorted({r["true_intent"] for r in gold} | {r["predicted_intent"] for r in gold})
            entry["intent"] = classification_report(
                [r["true_intent"] for r in gold],
                [r["predicted_intent"] for r in gold],
                labels=[label for label in labels if label in seen],
            )
            entry["task_completion_rate"] = sum(r["task_correct"] == "1" for r in gold) / len(gold)
            entry["asr_error_effect"] = _asr_effect(gold)
        report["branches"][branch] = entry

    # --- Paired A vs B on identical inputs (RQ4) ---------------------------
    pairs = defaultdict(dict)
    for row in rows:
        if row.get("true_intent"):
            pairs[(row["session_id"], row["trial_id"])][row["branch"]] = row
    matched = [p for p in pairs.values() if "A" in p and "B" in p]
    if matched:
        a_ok = [p["A"]["correct"] == "1" for p in matched]
        b_ok = [p["B"]["correct"] == "1" for p in matched]
        report["paired"] = mcnemar_exact(a_ok, b_ok)
        report["paired"]["agreement_rate"] = sum(
            p["A"]["predicted_intent"] == p["B"]["predicted_intent"] for p in matched
        ) / len(matched)
    return report


def _asr_effect(gold_rows):
    """Intent accuracy split by whether the ASR transcript was error-free."""
    buckets = {"wer_0": [], "wer_gt_0": []}
    for row in gold_rows:
        value = _float(row.get("wer"))
        if value is None:
            continue
        buckets["wer_0" if value == 0 else "wer_gt_0"].append(row["correct"] == "1")
    return {
        name: ({"n": len(v), "accuracy": sum(v) / len(v)} if v else {"n": 0, "accuracy": None})
        for name, v in buckets.items()
    }


def _fmt(value, pct=False):
    if value is None:
        return "n/a"
    return ("%.1f%%" % (value * 100)) if pct else ("%.3f" % value)


def print_report(report):
    print("Rows analysed: %d (excluded %d development-engine rows)" % (report["rows"], report["excluded_dev_rows"]))
    if not report["rows"] and report["excluded_dev_rows"]:
        print("Only stand-in (dev-rules / nllb-rules) output was logged. Configure the real engines, or pass --include-dev to inspect plumbing.")
    if report["asr"]:
        asr = report["asr"]
        print("\nASR on %d utterances: WER %s  CER %s" % (
            asr["n_utterances"], _fmt(asr["corpus_wer"], True), _fmt(asr["corpus_cer"], True)))
    for branch, entry in report["branches"].items():
        print("\nBranch %s  engines=%s  rows=%d  labelled=%d" % (
            branch, ",".join(entry["engines"]), entry["n_rows"], entry["n_labelled"]))
        if "intent" in entry:
            print("  accuracy %s  macro-F1 %s  task completion %s" % (
                _fmt(entry["intent"]["accuracy"]), _fmt(entry["intent"]["macro_f1"]),
                _fmt(entry["task_completion_rate"], True)))
        if entry["total_latency"]:
            lat = entry["total_latency"]
            print("  total latency: median %.0f ms, p95 %.0f ms" % (lat["median_ms"], lat["p95_ms"]))
    if report["paired"]:
        p = report["paired"]
        print("\nPaired A vs B on %d trials: only A right %d, only B right %d, McNemar exact p = %.4f, agreement %s" % (
            p["n_pairs"], p["only_a_correct"], p["only_b_correct"], p["p_value"], _fmt(p["agreement_rate"], True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="research log CSV")
    parser.add_argument("--include-dev", action="store_true", help="include dev-rules rows (never report these)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()
    report = summarise(read_rows(args.log), include_dev=args.include_dev)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)


if __name__ == "__main__":
    main()

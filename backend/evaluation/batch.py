"""Run a labelled dataset through both branches with identical inputs.

    python -m backend.evaluation.batch data/intent_dataset.csv --out data/logs/batch.csv
    python -m backend.evaluation.batch data/intent_dataset.csv --use-audio   # ASR first

The dataset uses the columns of data/intent_dataset_template.csv. Rows whose
`verified` column is not yes/true are skipped unless --allow-unverified is
given, so only checked rows enter the reported results.
Without --use-audio the `text` column is classified directly, which answers
RQ3/RQ4 on verified text even if no command recordings exist.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

from backend.config import settings
from backend.engines import Engines
from backend.evaluation.logger import ResearchLog, read_rows
from backend.evaluation.report import print_report, summarise
from backend.pipeline import run_trial


def _verified(row):
    return (row.get("verified") or "").strip().lower() in ("yes", "y", "true", "1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--out", default=None, help="log CSV to write (default data/logs/batch-<time>.csv)")
    parser.add_argument("--branches", default="both", choices=["A", "B", "both"])
    parser.add_argument("--use-audio", action="store_true", help="transcribe audio_path with Whisper first")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)

    with open(args.dataset, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if args.allow_unverified or _verified(r)]
    if len(kept) < len(rows):
        print("Skipped %d unverified rows." % (len(rows) - len(kept)), file=sys.stderr)
    if not kept:
        print("No rows to evaluate.", file=sys.stderr)
        return 1

    out = Path(args.out or settings.log_dir / ("batch-%s.csv" % time.strftime("%Y%m%dT%H%M%S")))
    engines = Engines(settings)
    engines.preload()
    if engines.load_errors:
        print("Engine load errors: %s" % engines.load_errors, file=sys.stderr)
        return 1
    log = ResearchLog(out)
    session = "batch-" + time.strftime("%Y%m%dT%H%M%S")

    failures = 0
    for index, row in enumerate(kept):
        audio = None
        if args.use_audio:
            if not row.get("audio_path"):
                failures += 1
                continue
            audio = Path(row["audio_path"]).read_bytes()
        events = list(run_trial(
            engines, log, settings.recordings_dir,
            audio=audio, text=row.get("text"), branch=args.branches,
            session_id=session, trial_id=row.get("id") or str(index),
            reference_transcript=row.get("reference_transcript") or (row.get("text") if args.use_audio else None),
            true_intent=row.get("intent"), true_course=row.get("course"), true_weekday=row.get("weekday"),
            log=True,
        ))
        if events[-1]["stage"] == "error":
            failures += 1
            print("Row %s failed: %s" % (row.get("id", index), events[-1]["error"]["message"]), file=sys.stderr)

    print("Wrote %s (%d rows evaluated, %d failed)\n" % (out, len(kept) - failures, failures))
    print_report(summarise(read_rows(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Pre-fill the labelling sheet with Whisper transcripts, to correct by hand.

    python -m backend.evaluation.draft_sheet data/recordings --out data/intent_dataset.csv

Adds a row for every audio file in the folder (top level only, so
data/recordings/unsplit/ is skipped) that is not yet in the sheet, then
transcribes every row whose draft_transcript is empty with the configured
Whisper decoding (beam 5 by default). For those rows it fills:
    draft_transcript      Whisper's output (keep as is: it is the ASR result)
    reference_transcript  only if empty; correct it to exactly what was said
    text                  only if empty; set it to the corrected transcript
Columns you have already filled are never overwritten, so it is safe to rerun.
Leave verified = no until you have corrected and checked a row: an unedited
reference equals the draft and would score a false 0% WER.

Needs the Whisper engine (requirements-ml.txt, weights in WhisperModel/).
The sheet and recordings stay off GitHub (see .gitignore).
"""

import argparse
import csv
import sys
from pathlib import Path

from backend.config import settings

AUDIO = {".wav", ".m4a", ".mp3", ".ogg", ".webm", ".flac", ".mp4"}
COLUMNS = ["id", "audio_path", "draft_transcript", "reference_transcript", "text", "intent",
           "course", "weekday", "location", "contact", "verified", "verified_by", "notes"]


def read_sheet(path):
    if not path.exists() or path.stat().st_size == 0:
        return [], list(COLUMNS)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    columns += [c for c in COLUMNS if c not in columns]
    return rows, columns


def write_sheet(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def add_new_files(rows, folder):
    known = {Path(r.get("audio_path", "")).as_posix() for r in rows}
    added = 0
    for audio in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO):
        if audio.as_posix() in known:
            continue
        rows.append({"id": audio.stem, "audio_path": audio.as_posix(), "verified": "no"})
        added += 1
    return added


def fill_drafts(rows, transcribe):
    """transcribe(path) -> text. Fills only empty draft/reference/text cells."""
    done = failed = 0
    for row in rows:
        if (row.get("draft_transcript") or "").strip():
            continue
        try:
            draft = transcribe(Path(row["audio_path"]))
        except Exception as error:  # keep going; note the problem on the row
            failed += 1
            row["notes"] = ((row.get("notes") or "") + " transcription failed: %s" % error).strip()
            continue
        row["draft_transcript"] = draft
        for column in ("reference_transcript", "text"):
            if not (row.get(column) or "").strip():
                row[column] = draft
        done += 1
        print("%s -> %s" % (row["id"], draft[:80]))
    return done, failed


def whisper_transcriber():
    from backend.engines import Engines
    from backend.pipeline import transcribe_audio

    engines = Engines(settings)
    if engines.asr.name == "none":
        raise SystemExit("No ASR model: install backend/requirements-ml.txt and put the Whisper weights "
                         "(model.safetensors) in WhisperModel/.")
    engines.asr.load()
    print("ASR:", engines.asr.describe())

    def transcribe(path):
        text, _ = transcribe_audio(engines, path.read_bytes())
        return text
    return transcribe


def main(argv=None, transcriber=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--out", default="data/intent_dataset.csv")
    parser.add_argument("--no-asr", action="store_true", help="only add rows for new files")
    args = parser.parse_args(argv)

    out = Path(args.out)
    rows, columns = read_sheet(out)
    added = add_new_files(rows, Path(args.folder))
    print("%d rows in %s (%d new)" % (len(rows), out, added))
    if not args.no_asr:
        done, failed = fill_drafts(rows, transcriber or whisper_transcriber())
        print("Transcribed %d, failed %d" % (done, failed))
    write_sheet(out, rows, columns)
    print("Next: correct reference_transcript and text, fill intent and slots, then set verified = yes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

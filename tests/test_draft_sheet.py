import csv

from backend.evaluation import draft_sheet


def read(path):
    with open(path, newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def test_adds_rows_and_fills_only_empty_cells(tmp_path):
    rec = tmp_path / "rec"
    (rec / "unsplit").mkdir(parents=True)
    for name in ("seg-001.wav", "seg-002.wav", "unsplit/seg-099.wav", "notes.txt"):
        (rec / name).write_bytes(b"x")
    sheet = tmp_path / "sheet.csv"
    assert draft_sheet.main([str(rec), "--out", str(sheet)], transcriber=lambda p: "draft " + p.stem) == 0
    rows = read(sheet)
    assert [r["id"] for r in rows] == ["seg-001", "seg-002"]          # unsplit/ and non-audio skipped
    assert rows[0]["draft_transcript"] == rows[0]["reference_transcript"] == rows[0]["text"] == "draft seg-001"
    assert rows[0]["verified"] == "no"

    # The user corrects a row; rerunning must not touch it.
    rows[0].update(reference_transcript="corrected", text="corrected", intent="greeting", verified="yes")
    rows[1].update(draft_transcript="")                                # forces a re-transcription
    rows[1].update(reference_transcript="typed by hand")
    draft_sheet.write_sheet(sheet, rows, list(rows[0].keys()))
    draft_sheet.main([str(rec), "--out", str(sheet)], transcriber=lambda p: "new " + p.stem)
    again = read(sheet)
    assert again[0]["reference_transcript"] == "corrected" and again[0]["verified"] == "yes"
    assert again[1]["draft_transcript"] == "new seg-002" and again[1]["reference_transcript"] == "typed by hand"


def test_failed_transcription_is_noted_not_fatal(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "seg-001.wav").write_bytes(b"x")
    def boom(path):
        raise RuntimeError("bad audio")
    sheet = tmp_path / "s.csv"
    draft_sheet.main([str(rec), "--out", str(sheet)], transcriber=boom)
    assert "transcription failed" in read(sheet)[0]["notes"]

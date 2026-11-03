from backend.branch_a.branch import BranchA
from backend.config import Settings
from backend.engines import _resolve_branch_a
from backend.evaluation.report import summarise


class FakeTranslator:
    model_id = "fake-nllb"
    loaded = True

    def __init__(self, mapping):
        self.mapping = mapping

    def load(self):
        pass

    def translate(self, text):
        return self.mapping[text]


def test_dev_rules_cannot_read_isizulu():
    result = BranchA(Settings(), "dev-rules").classify("ngubani ofundisa i-computer vision")
    assert result["intent"] == "OOS"
    assert "untranslated" in result["translation_note"]


def test_nllb_rules_classifies_the_english_translation():
    branch = BranchA(Settings(), "nllb-rules")
    branch.translator = FakeTranslator({"ngubani ofundisa i-computer vision": "Who teaches computer vision?"})
    result = branch.classify("ngubani ofundisa i-computer vision")
    assert result["engine"] == "nllb-rules"
    assert result["translation"] == "Who teaches computer vision?"
    assert result["intent"] == "ask_lecturer"
    assert result["slots"] == {"course": "COMS4036A"}
    assert branch.loaded


def test_auto_picks_nllb_rules_when_ml_is_installed(monkeypatch):
    monkeypatch.setattr("backend.engines.ml_available", lambda: True)
    settings = Settings()
    settings.branch_a_engine = "auto"
    settings.branch_a_classifier = None
    assert _resolve_branch_a(settings).engine == "nllb-rules"
    settings.branch_a_classifier = "models/clf"
    assert _resolve_branch_a(settings).engine == "hf"


def test_auto_falls_back_to_dev_rules_without_ml(monkeypatch):
    monkeypatch.setattr("backend.engines.ml_available", lambda: False)
    settings = Settings()
    settings.branch_a_engine = "auto"
    assert _resolve_branch_a(settings).engine == "dev-rules"


def test_nllb_rules_rows_are_excluded_from_reports():
    rows = [{"session_id": "s", "trial_id": "1", "branch": "A", "branch_engine": "nllb-rules"}]
    assert summarise(rows)["excluded_dev_rows"] == 1

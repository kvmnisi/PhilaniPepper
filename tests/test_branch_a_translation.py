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


def test_fine_tuned_classifier_folder_is_found_automatically(tmp_path, monkeypatch):
    import backend.config as config
    from backend.engines import _resolve_branch_a
    folder = tmp_path / "EnglishIntentClassifier"
    folder.mkdir()
    monkeypatch.setattr(config, "LOCAL_CLASSIFIER_DIR", folder)
    monkeypatch.delenv("BRANCH_A_CLASSIFIER", raising=False)
    assert Settings().branch_a_classifier is None                     # empty folder: not used
    (folder / "config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"")
    settings = Settings()
    assert settings.branch_a_classifier == str(folder)
    settings.branch_a_engine = "auto"
    monkeypatch.setattr("backend.engines.ml_available", lambda: True)
    assert _resolve_branch_a(settings).engine == "hf"
    monkeypatch.setenv("BRANCH_A_CLASSIFIER", "none")
    assert Settings().branch_a_classifier is None


def test_classifier_is_downloaded_from_classifier_repo(monkeypatch, tmp_path):
    import sys
    import types

    from backend import config

    calls = []

    def fake_download(repo, local_dir, token=None):
        calls.append(repo)
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_download))
    monkeypatch.setattr(config, "LOCAL_CLASSIFIER_DIR", tmp_path)
    monkeypatch.setenv("CLASSIFIER_REPO", "someone/classifier")
    monkeypatch.delenv("BRANCH_A_CLASSIFIER", raising=False)
    assert config._classifier_setting() == str(tmp_path)
    assert calls == ["someone/classifier"]
    # Already present: no second download.
    assert config._classifier_setting() == str(tmp_path)
    assert calls == ["someone/classifier"]

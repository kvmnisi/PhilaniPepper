import importlib.util
import json
import subprocess
import sys
import xml.dom.minidom
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PEPPER = ROOT / "pepper_app"


def load_executor():
    spec = importlib.util.spec_from_file_location("task_executor", PEPPER / "scripts" / "task_executor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pepper_html_is_in_sync_with_frontend():
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_pepper_html.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout


def test_pepper_copy_targets_pepper_runtime_without_css_variables():
    assert '"pepper"' in (PEPPER / "html" / "js" / "env.js").read_text()
    assert "var(" not in (PEPPER / "html" / "css" / "styles.css").read_text()


def test_package_xml_is_well_formed():
    for name in ("manifest.xml", "isizulu_wits_assistant.pml", "behavior.xar"):
        xml.dom.minidom.parse(str(PEPPER / name))


def test_event_names_match_tablet_bridge():
    executor = load_executor()
    js = (ROOT / "frontend" / "js" / "pepper.js").read_text()
    for name in executor.EVENTS.values():
        assert name in js


def test_multipart_body_contains_fields_and_audio():
    executor = load_executor()
    boundary, body = executor.encode_multipart({"branch": "B", "session_id": None}, "audio", "x.wav", b"RIFFdata")
    assert boundary.encode() in body
    assert b'name="branch"\r\n\r\nB' in body
    assert b"session_id" not in body
    assert b'filename="x.wav"' in body and b"RIFFdata" in body


def test_config_defaults_cover_config_file():
    executor = load_executor()
    config = json.loads((PEPPER / "scripts" / "config.json").read_text())
    assert set(config) == set(executor.DEFAULT_CONFIG)

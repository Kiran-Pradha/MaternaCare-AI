import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.voice.stt_vosk_integration import _resolve_model_dir


def test_resolve_model_dir_handles_nested_vosk_download(tmp_path):
    root = tmp_path / "models" / "vosk-model-small-en-us-0.15"
    nested = root / "vosk-model-small-en-us-0.15"
    (nested / "conf").mkdir(parents=True)
    (nested / "conf" / "model.conf").write_text("test", encoding="utf-8")

    resolved = _resolve_model_dir(str(root))

    assert resolved == str(nested), f"Expected {nested}, got {resolved}"
    assert os.path.isdir(resolved)

    print("[PASS] nested Vosk model directory resolves correctly")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_resolve_model_dir_handles_nested_vosk_download(__import__("pathlib").Path(d))

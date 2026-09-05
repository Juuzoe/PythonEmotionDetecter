from __future__ import annotations

from pathlib import Path

import pytest

from affectlab.cli import build_parser, main


def test_version_and_help() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    parser = build_parser()
    for command in ("live", "video", "image", "report", "doctor", "models"):
        assert command in parser.format_help()


def test_models_list_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "face_landmarker" in out and "ferplus" in out and "yunet" in out


def test_models_path_respects_models_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--models-dir", str(tmp_path), "models", "path", "yunet"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "face_detection_yunet_2023mar.onnx")


def test_report_on_missing_file_fails_cleanly(tmp_path: Path) -> None:
    assert main(["report", str(tmp_path / "missing.csv")]) != 0

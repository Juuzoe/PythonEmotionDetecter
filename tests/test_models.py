from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from affectlab import models


def test_models_dir_honours_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AFFECTLAB_MODELS_DIR", raising=False)
    monkeypatch.setenv("AFFECTLAB_HOME", str(tmp_path))
    assert models.models_dir() == tmp_path / "models"
    monkeypatch.setenv("AFFECTLAB_MODELS_DIR", str(tmp_path / "exact"))
    assert models.models_dir() == tmp_path / "exact"
    assert models.model_path("ferplus").name == "emotion-ferplus-8.onnx"


def test_unknown_model() -> None:
    with pytest.raises(models.ModelError):
        models.spec("nope")


def test_download_verifies_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    payload = b"affectlab" * 1000
    source.write_bytes(payload)
    url = source.as_uri()
    good = hashlib.sha256(payload).hexdigest()

    dest = tmp_path / "models" / "copy.bin"
    models.download_file(url, dest, good, progress=None)
    assert dest.read_bytes() == payload

    bad_dest = tmp_path / "models" / "bad.bin"
    with pytest.raises(models.ModelError):
        models.download_file(url, bad_dest, "0" * 64, progress=None)
    assert not bad_dest.exists()
    assert not list((tmp_path / "models").glob("*.part"))


def test_registry_is_consistent() -> None:
    for name, spec in models.MODELS.items():
        assert spec.name == name
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://")
        assert spec.size_bytes > 0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import httpx
import pytest

import oneload_tts.download_guard as download_guard
from oneload_tts.download_guard import download_locked_model, validate_download_target


def _lock(path: Path, payload: bytes = b"x") -> Path:
    path.write_text(
        json.dumps(
            {
                "model_id": "example/public-model",
                "revision": "1" * 40,
                "license": "Apache-2.0",
                "required_files": {
                    "weight.bin": {
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_download_guard_accepts_private_expected_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir(mode=0o700)
    (target / "weight.bin").write_bytes(b"partial")

    validate_download_target(target, _lock(tmp_path / "lock.json"))


def test_download_guard_rejects_symlinked_target(tmp_path: Path) -> None:
    real_target = tmp_path / "real-model"
    real_target.mkdir(mode=0o700)
    linked_target = tmp_path / "linked-model"
    linked_target.symlink_to(real_target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="not protected"):
        validate_download_target(linked_target, _lock(tmp_path / "lock.json"))


def test_download_guard_rejects_unexpected_entry(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir(mode=0o700)
    (target / "unexpected.bin").write_bytes(b"attacker")

    with pytest.raises(RuntimeError, match="unexpected entry"):
        validate_download_target(target, _lock(tmp_path / "lock.json"))


def test_download_guard_rejects_unprotected_cache_root(tmp_path: Path) -> None:
    target = tmp_path / "model"
    cache = target / ".cache"
    cache.mkdir(parents=True, mode=0o700)
    cache.chmod(0o777)

    with pytest.raises(RuntimeError, match="unprotected"):
        validate_download_target(target, _lock(tmp_path / "lock.json"))


def test_download_guard_does_not_traverse_legacy_cache(tmp_path: Path) -> None:
    target = tmp_path / "model"
    cache = target / ".cache"
    cache.mkdir(parents=True, mode=0o700)
    (cache / "legacy-link").symlink_to(tmp_path / "outside")

    validate_download_target(target, _lock(tmp_path / "lock.json"))


def test_locked_downloader_caps_and_verifies_public_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"locked model bytes"
    target = tmp_path / "model"
    lock = _lock(tmp_path / "lock.json", payload)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"location": "https://us.aws.cdn.hf.co/model"})
        return httpx.Response(
            200,
            headers={"content-length": str(len(payload)), "content-encoding": "identity"},
            content=payload,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    options: dict[str, object] = {}

    def client_factory(**kwargs: object) -> httpx.Client:
        options.update(kwargs)
        return client

    monkeypatch.setattr(download_guard.httpx, "Client", client_factory)

    download_locked_model(target, lock)

    assert (target / "weight.bin").read_bytes() == payload
    assert options["trust_env"] is False
    assert [request.url.host for request in requests] == ["huggingface.co", "us.aws.cdn.hf.co"]
    assert all("authorization" not in request.headers for request in requests)


def test_locked_downloader_rejects_oversize_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"locked"
    target = tmp_path / "model"
    lock = _lock(tmp_path / "lock.json", payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(len(payload))},
            content=payload + b"attacker",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(download_guard.httpx, "Client", lambda **_: client)

    with pytest.raises(RuntimeError, match="byte budget"):
        download_locked_model(target, lock)

    assert not (target / "weight.bin").exists()
    assert os.listdir(target) == []


def test_locked_downloader_rejects_redirect_outside_hugging_face(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"locked"
    target = tmp_path / "model"
    lock = _lock(tmp_path / "lock.json", payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(download_guard.httpx, "Client", lambda **_: client)

    with pytest.raises(RuntimeError, match="unsafe redirect"):
        download_locked_model(target, lock)

    assert not (target / "weight.bin").exists()

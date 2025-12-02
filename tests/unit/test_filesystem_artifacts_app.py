from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sparkle_motion.filesystem_artifacts import FilesystemArtifactsConfig, create_app


@pytest.fixture()
def shim_config(tmp_path: Path) -> FilesystemArtifactsConfig:
    root = tmp_path / "root"
    index = tmp_path / "index.db"
    return FilesystemArtifactsConfig(
        root=root,
        index_path=index,
        base_url="http://127.0.0.1:7077",
        token="secret-token",
        allow_insecure=False,
        max_payload_bytes=8 * 1024 * 1024,
    )


@pytest.fixture()
def shim_client(shim_config: FilesystemArtifactsConfig) -> TestClient:
    app = create_app(config=shim_config)
    return TestClient(app)


def _auth_headers(token: str = "secret-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _sample_manifest() -> dict[str, object]:
    return {
        "run_id": "run_local_1",
        "stage": "plan_intake",
        "artifact_type": "movie_plan",
        "mime_type": "application/json",
        "metadata": {
            "schema_uri": "schemas/movie_plan.json",
        },
    }


def test_healthcheck_does_not_require_auth(shim_client: TestClient) -> None:
    response = shim_client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_requires_auth(shim_client: TestClient) -> None:
    response = shim_client.post("/artifacts", json={})
    assert response.status_code == 401


def test_json_upload_get_and_list(shim_client: TestClient) -> None:
    manifest = _sample_manifest()
    payload = base64.b64encode(b"hello world").decode("ascii")
    response = shim_client.post(
        "/artifacts",
        json={
            "manifest": manifest,
            "payload_b64": payload,
            "filename_hint": "movie_plan.json",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 201
    artifact = response.json()

    artifact_id = artifact["artifact_id"]
    get_response = shim_client.get(f"/artifacts/{artifact_id}", headers=_auth_headers())
    assert get_response.status_code == 200
    assert get_response.json()["artifact_id"] == artifact_id

    list_response = shim_client.get(
        "/artifacts",
        params={"run_id": manifest["run_id"]},
        headers=_auth_headers(),
    )
    assert list_response.status_code == 200
    data = list_response.json()
    assert data["items"], "Expected at least one artifact"
    assert data["items"][0]["artifact_id"] == artifact_id


def test_include_payload_streams_file(shim_client: TestClient) -> None:
    payload_bytes = b"payload-bytes"
    response = shim_client.post(
        "/artifacts",
        json={
            "manifest": _sample_manifest(),
            "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
            "filename_hint": "artifact.bin",
        },
        headers=_auth_headers(),
    )
    assert response.status_code == 201
    artifact_id = response.json()["artifact_id"]

    payload_response = shim_client.get(
        f"/artifacts/{artifact_id}",
        params={"include_payload": "true"},
        headers=_auth_headers(),
    )
    assert payload_response.status_code == 200
    assert payload_response.content == payload_bytes
    assert payload_response.headers["content-type"] == "application/json"

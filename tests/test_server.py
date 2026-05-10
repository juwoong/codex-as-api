from __future__ import annotations

import json
import logging

import pytest

from codex_as_api.auth import ChatGPTOAuthError, ChatGPTOAuthMissingError


@pytest.fixture()
def client():
    from codex_as_api.server import app
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_root_returns_auth_status(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "codex-as-api"
    assert "auth_available" in body
    assert "auth_status" in body


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "auth_available" in body
    assert "model" in body


def test_request_timing_log_includes_duration_ms(client, caplog):
    caplog.set_level(logging.INFO, logger="gunicorn.error")
    resp = client.get("/health")
    assert resp.status_code == 200

    messages = [record.getMessage() for record in caplog.records if record.name == "gunicorn.error"]
    assert any("method=GET" in message and "path=/health" in message for message in messages)
    assert any("duration_ms=" in message for message in messages)


# ---------------------------------------------------------------------------
# API auth
# ---------------------------------------------------------------------------


def test_authorization_token_accepts_bearer_and_raw():
    from codex_as_api.server import _authorization_token
    assert _authorization_token("Bearer test-key") == "test-key"
    assert _authorization_token("test-key") == "test-key"
    assert _authorization_token(None) == ""


def test_chat_completions_missing_api_key_returns_401(tmp_path):
    import codex_as_api.server as server_mod
    old_api_key = server_mod.API_KEY
    old_auth_path = server_mod.AUTH_PATH
    server_mod.API_KEY = "secret-key"
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    try:
        from codex_as_api.server import app
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = c.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"
        assert resp.json()["error"]["type"] == "api_auth_error"
    finally:
        server_mod.API_KEY = old_api_key
        server_mod.AUTH_PATH = old_auth_path
        server_mod._provider = None


def test_chat_completions_valid_api_key_continues_to_oauth_check(tmp_path):
    import codex_as_api.server as server_mod
    old_api_key = server_mod.API_KEY
    old_auth_path = server_mod.AUTH_PATH
    server_mod.API_KEY = "secret-key"
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    try:
        from codex_as_api.server import app
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = c.post("/v1/chat/completions", headers={"Authorization": "Bearer secret-key"}, json=payload)
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "chatgpt_oauth_error"
    finally:
        server_mod.API_KEY = old_api_key
        server_mod.AUTH_PATH = old_auth_path
        server_mod._provider = None


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — schema validation
# ---------------------------------------------------------------------------


def test_chat_completions_invalid_body_returns_422(client):
    resp = client.post("/v1/chat/completions", json={})
    assert resp.status_code == 422


def test_chat_completions_valid_schema_reaches_provider(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code in (200, 401, 500)
    if resp.status_code == 422:
        pytest.fail(f"Schema validation rejected a valid request: {resp.json()}")


def test_chat_completions_auth_error_not_422(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_subagent_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "subagent": "my-subagent",
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_memgen_request_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "memgen_request": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_previous_response_id_field_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ],
        "previous_response_id": "resp-abc123",
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_all_extended_fields_accepted(client):
    payload = {
        "model": "gpt-5.5",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        "subagent": "agent-1",
        "memgen_request": False,
        "previous_response_id": "resp-xyz",
        "reasoning_effort": "high",
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code != 422


def test_chat_completions_missing_auth_returns_auth_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "nonexistent.json"))
    import codex_as_api.server as server_mod
    old_auth_path = server_mod.AUTH_PATH
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    try:
        from codex_as_api.server import app
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = c.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "chatgpt_oauth_error"
    finally:
        server_mod.AUTH_PATH = old_auth_path
        server_mod._provider = None


def test_chat_completions_stream_missing_auth_returns_401_before_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "nonexistent.json"))
    import codex_as_api.server as server_mod
    old_auth_path = server_mod.AUTH_PATH
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    try:
        from codex_as_api.server import app
        from fastapi.testclient import TestClient
        c = TestClient(app, raise_server_exceptions=False)
        payload = {
            "model": "gpt-5.5",
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        resp = c.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 401
        assert resp.headers["content-type"].startswith("application/json")
    finally:
        server_mod.AUTH_PATH = old_auth_path
        server_mod._provider = None

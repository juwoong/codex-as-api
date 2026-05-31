from __future__ import annotations

import logging

import pytest


class RecordingSummaryProvider:
    def __init__(self, summary: str = "integrated summary") -> None:
        self.summary = summary
        self.calls: list[dict] = []

    def summarize_files(self, content_items, *, model=None, reasoning_effort=None):
        self.calls.append({
            "content_items": content_items,
            "model": model,
            "reasoning_effort": reasoning_effort,
        })
        return self.summary


class RecordingChatProvider:
    def __init__(self, content: str = "webhook answer") -> None:
        self.content = content
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        from codex_as_api.messages import AssistantResponse, Usage

        self.calls.append({"messages": messages, **kwargs})
        return AssistantResponse(
            content=self.content,
            finish_reason="stop",
            usage=Usage(prompt_tokens=4, completion_tokens=3),
        )


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from codex_as_api.server import app
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
        from fastapi.testclient import TestClient

        from codex_as_api.server import app
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
        from fastapi.testclient import TestClient

        from codex_as_api.server import app
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
# POST /v1/files/summarize
# ---------------------------------------------------------------------------


def test_files_summarize_multipart_parses_files_and_context(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    provider = RecordingSummaryProvider("summary text")
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)
    monkeypatch.setattr(server_mod, "_provider", provider)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        data={
            "model": "gpt-5.5",
            "context": "May 2026 ABC vendor ledger",
            "reasoning_effort": "high",
        },
        files=[
            ("files", ("ledger.pdf", b"%PDF-1.4\nfake", "application/pdf")),
            ("files", ("receipt.png", b"\x89PNG\r\n\x1a\nfake", "image/png")),
            ("files", ("memo.txt", b"Payment memo", "text/plain")),
        ],
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["summary"] == "summary text"
    assert body["model"] == "codex-oauth:gpt-5.5"
    assert body["files"] == [
        {"filename": "ledger.pdf", "content_type": "application/pdf", "size_bytes": 13, "kind": "pdf"},
        {"filename": "receipt.png", "content_type": "image/png", "size_bytes": 12, "kind": "image"},
        {"filename": "memo.txt", "content_type": "text/plain", "size_bytes": 12, "kind": "text"},
    ]

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["reasoning_effort"] == "high"
    content_items = call["content_items"]
    text_payload = "\n".join(item["text"] for item in content_items if item["type"] == "input_text")
    assert "Context metadata:\nMay 2026 ABC vendor ledger" in text_payload
    assert "not a summary instruction" in text_payload
    assert "Payment memo" in text_payload

    pdf_items = [item for item in content_items if item["type"] == "input_file"]
    assert pdf_items == [{
        "type": "input_file",
        "filename": "ledger.pdf",
        "file_data": pdf_items[0]["file_data"],
    }]
    assert pdf_items[0]["file_data"].startswith("data:application/pdf;base64,")

    image_items = [item for item in content_items if item["type"] == "input_image"]
    assert len(image_items) == 1
    assert image_items[0]["image_url"].startswith("data:image/png;base64,")


def test_files_summarize_missing_files_returns_4xx(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    monkeypatch.setattr(server_mod, "API_KEY", None)
    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post("/v1/files/summarize", data={"model": "gpt-5.5"})
    assert 400 <= resp.status_code < 500


def test_files_summarize_too_many_files_returns_400(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    provider = RecordingSummaryProvider()
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)
    monkeypatch.setattr(server_mod, "_provider", provider)
    monkeypatch.setattr(server_mod, "MAX_SUMMARY_FILES", 2)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        data={"model": "gpt-5.5"},
        files=[
            ("files", ("one.txt", b"one", "text/plain")),
            ("files", ("two.txt", b"two", "text/plain")),
            ("files", ("three.txt", b"three", "text/plain")),
        ],
    )
    assert resp.status_code == 400
    assert provider.calls == []


def test_files_summarize_file_too_large_returns_413(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    provider = RecordingSummaryProvider()
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)
    monkeypatch.setattr(server_mod, "_provider", provider)
    monkeypatch.setattr(server_mod, "MAX_SUMMARY_FILE_BYTES", 5)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        data={"model": "gpt-5.5"},
        files=[("files", ("large.txt", b"123456", "text/plain"))],
    )
    assert resp.status_code == 413
    assert provider.calls == []


def test_files_summarize_unsupported_file_type_returns_415(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    provider = RecordingSummaryProvider()
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)
    monkeypatch.setattr(server_mod, "_provider", provider)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        data={"model": "gpt-5.5"},
        files=[("files", ("payload.bin", b"\x00\x01", "application/octet-stream"))],
    )
    assert resp.status_code == 415
    assert provider.calls == []


def test_files_summarize_missing_api_key_returns_401(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    monkeypatch.setattr(server_mod, "API_KEY", "secret-key")
    monkeypatch.setattr(server_mod, "_require_auth", lambda: pytest.fail("_require_auth should not run"))

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        data={"model": "gpt-5.5"},
        files=[("files", ("memo.txt", b"hello", "text/plain"))],
    )
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json()["error"]["type"] == "api_auth_error"


def test_files_summarize_valid_api_key_continues_to_oauth_check(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    monkeypatch.setattr(server_mod, "API_KEY", "secret-key")
    monkeypatch.setattr(server_mod, "AUTH_PATH", str(tmp_path / "nonexistent.json"))
    monkeypatch.setattr(server_mod, "_provider", None)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/files/summarize",
        headers={"Authorization": "Bearer secret-key"},
        data={"model": "gpt-5.5"},
        files=[("files", ("memo.txt", b"hello", "text/plain"))],
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "chatgpt_oauth_error"


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
        from fastapi.testclient import TestClient

        from codex_as_api.server import app
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


# ---------------------------------------------------------------------------
# POST /v1/jobs/chat/completions
# ---------------------------------------------------------------------------


def test_chat_completions_webhook_queues_and_posts_response(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    caplog.set_level(logging.INFO, logger="gunicorn.error")
    provider = RecordingChatProvider("async answer")
    deliveries: list[dict] = []
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)
    monkeypatch.setattr(server_mod, "_provider", provider)
    monkeypatch.setattr(
        server_mod,
        "_post_job_webhook",
        lambda webhook_url, payload: deliveries.append({"webhook_url": webhook_url, "payload": payload}),
    )

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/jobs/chat/completions",
        headers={
            "x-openai-subagent": "header-agent",
            "x-openai-memgen-request": "true",
        },
        json={
            "model": "gpt-5.5",
            "webhook_url": "https://example.test/webhook?token=secret-token",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert resp.status_code == 202
    body = resp.json()
    assert body["object"] == "chat.completion.webhook.job"
    assert body["status"] == "queued"
    assert body["id"].startswith("job-")

    assert len(deliveries) == 1
    assert deliveries[0]["webhook_url"] == "https://example.test/webhook?token=secret-token"
    payload = deliveries[0]["payload"]
    assert payload["id"] == body["id"]
    assert payload["object"] == "chat.completion.webhook"
    assert payload["status"] == "completed"
    assert payload["response"]["object"] == "chat.completion"
    assert payload["response"]["choices"][0]["message"]["content"] == "async answer"
    assert payload["response"]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
    }

    assert len(provider.calls) == 1
    assert provider.calls[0]["subagent"] == "header-agent"
    assert provider.calls[0]["memgen_request"] is True

    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "gunicorn.error")
    assert "webhook job request" in logs
    assert '"message_count": 2' in logs
    assert '"message_roles": {"system": 1, "user": 1}' in logs
    assert "webhook job queued response" in logs
    assert '"status": "queued"' in logs
    assert "webhook job response" in logs
    assert '"content_chars": 12' in logs
    assert "webhook delivery request" in logs
    assert "https://example.test/webhook" in logs
    assert "secret-token" not in logs


def test_chat_completions_webhook_rejects_stream(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/jobs/chat/completions",
        json={
            "model": "gpt-5.5",
            "webhook_url": "https://example.test/webhook",
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert resp.status_code == 400
    assert "stream is not supported" in resp.json()["detail"]


def test_chat_completions_webhook_rejects_invalid_webhook_url(monkeypatch):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: None)

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/jobs/chat/completions",
        json={
            "model": "gpt-5.5",
            "webhook_url": "/relative-webhook",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        },
    )

    assert resp.status_code == 400
    assert "webhook_url" in resp.json()["detail"]


def test_chat_completions_webhook_validation_error_logs_request_and_response(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    import codex_as_api.server as server_mod
    from codex_as_api.server import app

    caplog.set_level(logging.INFO, logger="gunicorn.error")
    monkeypatch.setattr(server_mod, "API_KEY", None)
    monkeypatch.setattr(server_mod, "_require_auth", lambda: pytest.fail("_require_auth should not run"))

    c = TestClient(app, raise_server_exceptions=False)
    resp = c.post(
        "/v1/jobs/chat/completions",
        json={
            "model": "gpt-5.5",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello 422 validation body"},
            ],
            "client_metadata": {
                "token": "secret-client-token",
            },
        },
    )

    assert resp.status_code == 422
    logs = "\n".join(record.getMessage() for record in caplog.records if record.name == "gunicorn.error")
    assert "request validation error" in logs
    assert '"path": "/v1/jobs/chat/completions"' in logs
    assert '"status_code": 422' in logs
    assert "Hello 422 validation body" in logs
    assert '"webhook_url"' in logs
    assert "Field required" in logs
    assert "secret-client-token" not in logs
    assert "<redacted>" in logs
    assert "webhook job request" not in logs


def test_webhook_delivery_headers_include_job_webhook_credential(monkeypatch):
    import codex_as_api.server as server_mod

    monkeypatch.setattr(server_mod, "JOB_WEBHOOK_CREDENTIAL", "secret-webhook-token")

    headers = server_mod._webhook_delivery_headers()

    assert headers["Authorization"] == "Bearer secret-webhook-token"


def test_chat_completions_stream_missing_auth_returns_401_before_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "nonexistent.json"))
    import codex_as_api.server as server_mod
    old_auth_path = server_mod.AUTH_PATH
    server_mod.AUTH_PATH = str(tmp_path / "nonexistent.json")
    server_mod._provider = None
    try:
        from fastapi.testclient import TestClient

        from codex_as_api.server import app
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

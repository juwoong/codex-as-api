from __future__ import annotations

import base64
import hmac
import json
import logging
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any

from .auth import (
    ChatGPTOAuthError,
    ChatGPTOAuthMissingError,
    is_auth_locally_available,
    load_token_data,
    resolve_auth_path,
)
from .messages import Message, MessageRole, ToolSchema
from .provider import ChatGPTOAuthProvider


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is not None and value.isdigit():
        return int(value)
    return default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


HOST = _env_str("CODEX_AS_API_HOST", "127.0.0.1")
PORT = _env_int("CODEX_AS_API_PORT", 18080)
MODEL = _env_str("CODEX_AS_API_MODEL", "gpt-5.5")
AUTH_PATH = os.getenv("CODEX_AS_API_AUTH_PATH")
API_KEY = os.getenv("CODEX_AS_API_API_KEY")
JOB_WEBHOOK_CREDENTIAL = os.getenv("JOB_WEBHOOK_CREDENTIAL")
JOB_WEBHOOK_TIMEOUT_SECONDS = _env_int("JOB_WEBHOOK_TIMEOUT_SECONDS", 30)
MAX_SUMMARY_FILES = 10
MAX_SUMMARY_FILE_BYTES = 10 * 1024 * 1024
SUMMARY_TEXT_MIME_TYPES = frozenset({
    "application/csv",
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/sql",
    "application/toml",
    "application/typescript",
    "application/x-ndjson",
    "application/x-sh",
    "application/x-yaml",
    "application/xml",
    "application/yaml",
})
SUMMARY_TEXT_EXTENSIONS = frozenset({
    ".cfg",
    ".conf",
    ".css",
    ".csv",
    ".env",
    ".htm",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".log",
    ".md",
    ".markdown",
    ".py",
    ".rst",
    ".sql",
    ".text",
    ".toml",
    ".ts",
    ".tsx",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
})

_provider: ChatGPTOAuthProvider | None = None
_request_logger = logging.getLogger("gunicorn.error")


class APIAuthError(RuntimeError):
    pass


def _get_provider() -> ChatGPTOAuthProvider:
    global _provider
    if _provider is None:
        _provider = ChatGPTOAuthProvider(
            model=MODEL,
            auth_json_path=AUTH_PATH,
        )
    return _provider


def _auth_status() -> dict[str, Any]:
    auth_available = is_auth_locally_available(AUTH_PATH)
    result: dict[str, Any] = {
        "auth_available": auth_available,
        "auth_path": str(resolve_auth_path(AUTH_PATH)),
    }
    if not auth_available:
        result["auth_status"] = "required"
        result["auth_hint"] = "Run `codex login` with CODEX_HOME pointing at the persistent auth volume."
    else:
        result["auth_status"] = "ready"
    return result


def _api_auth_status() -> dict[str, Any]:
    return {"api_auth_enabled": bool(API_KEY)}


def _authorization_token(value: str | None) -> str:
    if value is None:
        return ""
    stripped = value.strip()
    if stripped.lower().startswith("bearer "):
        return stripped[7:].strip()
    return stripped


def _require_api_key(request: Any) -> None:
    expected = API_KEY.strip() if API_KEY else ""
    if not expected:
        return
    provided = _authorization_token(request.headers.get("authorization"))
    if not provided or not hmac.compare_digest(provided, expected):
        raise APIAuthError("Invalid or missing API key")


def _require_auth() -> None:
    load_token_data(AUTH_PATH)


# FastAPI is an optional dependency; fail gracefully if missing.
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    from starlette.datastructures import FormData
    from starlette.datastructures import UploadFile as StarletteUploadFile

    app = FastAPI(
        title="codex-as-api",
        description="Local OpenAI-compatible API server backed by ChatGPT/Codex OAuth.",
        version="0.1.0",
    )

    class RequestTimingMiddleware:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await self.app(scope, receive, send)
                return

            started_at = time.perf_counter()
            method = str(scope.get("method") or "")
            path = str(scope.get("path") or "")
            client = scope.get("client")
            client_host = client[0] if isinstance(client, tuple) and client else "-"
            status_code: int | None = None
            logged = False

            def log_request(status: int | str, *, error: bool = False) -> None:
                nonlocal logged
                if logged:
                    return
                logged = True
                duration_ms = (time.perf_counter() - started_at) * 1000
                _request_logger.info(
                    "request method=%s path=%s status=%s duration_ms=%.2f client=%s error=%s",
                    method,
                    path,
                    status,
                    duration_ms,
                    client_host,
                    str(error).lower(),
                )

            async def send_with_timing(message: dict[str, Any]) -> None:
                nonlocal status_code
                if message.get("type") == "http.response.start":
                    status_code = int(message.get("status") or 0)
                await send(message)
                if message.get("type") == "http.response.body" and not message.get("more_body", False):
                    log_request(status_code or "-")

            try:
                await self.app(scope, receive, send_with_timing)
            except Exception:
                log_request(status_code or 500, error=True)
                raise

    app.add_middleware(RequestTimingMiddleware)

    @app.exception_handler(ChatGPTOAuthError)
    async def _chatgpt_oauth_error_handler(_request: Request, exc: ChatGPTOAuthError) -> JSONResponse:
        status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
        return JSONResponse(status_code=status, content={"error": {"message": str(exc), "type": "chatgpt_oauth_error"}})

    @app.exception_handler(APIAuthError)
    async def _api_auth_error_handler(_request: Request, exc: APIAuthError) -> JSONResponse:
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={"error": {"message": str(exc), "type": "api_auth_error"}},
        )

    # ------------------------------------------------------------------
    # Request/response schemas
    # ------------------------------------------------------------------

    class ChatMessage(BaseModel):
        role: str
        content: str | list[dict[str, Any]] | None = None
        name: str | None = None
        tool_calls: list[dict[str, Any]] | None = None
        tool_call_id: str | None = None

    class ChatCompletionRequest(BaseModel):
        model: str
        messages: list[ChatMessage]
        stream: bool = False
        temperature: float | None = None
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        stop: str | list[str] | None = None
        tools: list[dict[str, Any]] | None = None
        tool_choice: str | dict[str, Any] | None = None
        reasoning_effort: str | None = None
        prompt_cache_key: str | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        user: str | None = None
        subagent: str | None = None
        memgen_request: bool | None = None
        previous_response_id: str | None = None
        service_tier: str | None = None
        text: dict[str, Any] | None = None
        client_metadata: dict[str, str] | None = None

    class WebhookChatCompletionRequest(ChatCompletionRequest):
        webhook_url: str

    class ImageGenerationRequest(BaseModel):
        model: str
        prompt: str
        size: str | None = "auto"
        reasoning_effort: str | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _openai_model_id(request_model: str | None = None) -> str:
        return f"codex-oauth:{request_model or MODEL}"

    def _request_messages_to_internal(messages: list[ChatMessage]) -> list[Message]:
        result: list[Message] = []
        for msg in messages:
            role = _map_role(msg.role)
            content = _normalize_content(msg.content)
            tool_calls = _parse_tool_calls(msg.tool_calls) if msg.tool_calls else ()
            result.append(
                Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                )
            )
        return result

    def _map_role(role: str) -> MessageRole:
        mapping = {
            "system": MessageRole.SYSTEM,
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "tool": MessageRole.TOOL,
        }
        return mapping.get(role.lower(), MessageRole.USER)

    def _normalize_content(content: str | list[dict[str, Any]] | None) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content)

    def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> tuple[Any, ...]:
        from .messages import ToolCall
        if not raw:
            return ()
        calls = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            call_id = item.get("id") or item.get("call_id") or str(uuid.uuid4().hex)
            func = item.get("function") or {}
            name = func.get("name") if isinstance(func, dict) else item.get("name")
            args = func.get("arguments") if isinstance(func, dict) else item.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    parsed = {"input": args}
            elif isinstance(args, dict):
                parsed = args
            else:
                parsed = {}
            if name:
                calls.append(ToolCall(id=str(call_id), name=str(name), arguments=parsed))
        return tuple(calls)

    def _parse_tools(raw: list[dict[str, Any]] | None) -> list[ToolSchema] | None:
        if not raw:
            return None
        schemas = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            func = item.get("function") or item
            name = func.get("name")
            desc = func.get("description") or ""
            params = func.get("parameters") or {}
            if name:
                schemas.append(
                    ToolSchema(
                        name=str(name),
                        description=str(desc),
                        parameters=params if isinstance(params, dict) else {},
                    )
                )
        return schemas if schemas else None

    def _normalize_stop(stop: str | list[str] | None) -> list[str] | None:
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        return list(stop)

    def _max_tokens_from_request(req: ChatCompletionRequest) -> int | None:
        if req.max_completion_tokens is not None:
            return req.max_completion_tokens
        return req.max_tokens

    def _normalized_headers(headers: Mapping[str, str]) -> dict[str, str]:
        return {str(key).lower(): str(value) for key, value in headers.items()}

    def _json_log(fields: dict[str, Any]) -> str:
        return json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)

    def _safe_webhook_url(webhook_url: str) -> str:
        parsed = urllib.parse.urlparse(webhook_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "<invalid>"
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    def _message_role_counts(messages: list[ChatMessage]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in messages:
            role = (message.role or "unknown").lower()
            counts[role] = counts.get(role, 0) + 1
        return counts

    def _request_header_options(
        request: ChatCompletionRequest,
        headers: Mapping[str, str],
    ) -> tuple[str | None, bool | None]:
        normalized_headers = _normalized_headers(headers)
        subagent = request.subagent or normalized_headers.get("x-openai-subagent")
        memgen_request_header = normalized_headers.get("x-openai-memgen-request")
        memgen_request: bool | None = request.memgen_request
        if memgen_request is None and memgen_request_header is not None:
            memgen_request = memgen_request_header.lower() not in ("false", "0", "")
        return subagent, memgen_request

    def _chat_request_log_fields(
        *,
        job_id: str,
        request: WebhookChatCompletionRequest,
        headers: Mapping[str, str],
    ) -> dict[str, Any]:
        subagent, memgen_request = _request_header_options(request, headers)
        return {
            "job_id": job_id,
            "model": request.model,
            "message_count": len(request.messages),
            "message_roles": _message_role_counts(request.messages),
            "stream": request.stream,
            "tool_count": len(request.tools or []),
            "tool_choice_present": request.tool_choice is not None,
            "reasoning_effort": request.reasoning_effort,
            "max_tokens": _max_tokens_from_request(request),
            "stop_count": len(_normalize_stop(request.stop) or []),
            "prompt_cache_key_present": request.prompt_cache_key is not None,
            "previous_response_id_present": request.previous_response_id is not None,
            "service_tier": request.service_tier,
            "text_present": request.text is not None,
            "client_metadata_count": len(request.client_metadata or {}),
            "subagent": subagent,
            "memgen_request": memgen_request,
            "webhook_url": _safe_webhook_url(request.webhook_url),
            "webhook_credential_configured": bool(JOB_WEBHOOK_CREDENTIAL),
            "webhook_timeout_seconds": JOB_WEBHOOK_TIMEOUT_SECONDS,
        }

    def _chat_response_log_fields(job_id: str, payload: dict[str, Any], duration_ms: float) -> dict[str, Any]:
        response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
        choices = response.get("choices") if isinstance(response, dict) else None
        first_choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        return {
            "job_id": job_id,
            "status": payload.get("status"),
            "duration_ms": round(duration_ms, 2),
            "completion_id": response.get("id") if isinstance(response, dict) else None,
            "model": response.get("model") if isinstance(response, dict) else None,
            "choice_count": len(choices) if isinstance(choices, list) else 0,
            "finish_reason": first_choice.get("finish_reason"),
            "usage": response.get("usage") if isinstance(response, dict) else None,
            "content_chars": len(content) if isinstance(content, str) else 0,
            "reasoning_present": bool(message.get("reasoning_content")) if isinstance(message, dict) else False,
            "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        }

    def _chat_completion_response_payload(
        request: ChatCompletionRequest,
        headers: Mapping[str, str],
    ) -> dict[str, Any]:
        provider = _get_provider()
        messages = _request_messages_to_internal(request.messages)
        tools = _parse_tools(request.tools)
        stop = _normalize_stop(request.stop)
        max_tokens = _max_tokens_from_request(request)
        subagent, memgen_request = _request_header_options(request, headers)

        response = provider.chat(
            messages,
            model=request.model,
            tools=tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            reasoning_effort=request.reasoning_effort,
            max_tokens=max_tokens,
            stop=stop,
            prompt_cache_key=request.prompt_cache_key,
            subagent=subagent,
            memgen_request=memgen_request,
            previous_response_id=request.previous_response_id,
            service_tier=request.service_tier,
            text=request.text,
            client_metadata=request.client_metadata,
        )

        choices: list[dict[str, Any]] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content,
            },
            "finish_reason": response.finish_reason,
        }]

        if response.tool_calls:
            choices[0]["message"]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]

        if response.reasoning_content:
            choices[0]["message"]["reasoning_content"] = response.reasoning_content

        result: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _openai_model_id(request.model),
            "choices": choices,
        }

        if response.usage:
            total_tokens = response.usage.total_tokens or (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": total_tokens,
            }

        return result

    def _validate_webhook_url(webhook_url: str) -> None:
        parsed = urllib.parse.urlparse(webhook_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="webhook_url must be an absolute http(s) URL")

    def _webhook_delivery_headers() -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "codex-as-api",
        }
        credential = JOB_WEBHOOK_CREDENTIAL.strip() if JOB_WEBHOOK_CREDENTIAL else ""
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        return headers

    def _webhook_error_type(exc: Exception) -> str:
        if isinstance(exc, ChatGPTOAuthError):
            return "chatgpt_oauth_error"
        if isinstance(exc, APIAuthError):
            return "api_auth_error"
        return "server_error"

    def _post_job_webhook(webhook_url: str, payload: dict[str, Any]) -> int:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            webhook_url,
            data=data,
            headers=_webhook_delivery_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=JOB_WEBHOOK_TIMEOUT_SECONDS) as response:
            response.read()
            return int(response.getcode())

    def _run_chat_completion_webhook_job(
        job_id: str,
        request: ChatCompletionRequest,
        headers: Mapping[str, str],
        webhook_url: str,
    ) -> None:
        started_at = time.perf_counter()
        try:
            response_payload = _chat_completion_response_payload(request, headers)
            payload = {
                "id": job_id,
                "object": "chat.completion.webhook",
                "status": "completed",
                "created": int(time.time()),
                "response": response_payload,
            }
            _request_logger.info(
                "webhook job response %s",
                _json_log(_chat_response_log_fields(job_id, payload, (time.perf_counter() - started_at) * 1000)),
            )
        except Exception as exc:
            payload = {
                "id": job_id,
                "object": "chat.completion.webhook",
                "status": "failed",
                "created": int(time.time()),
                "error": {
                    "message": str(exc),
                    "type": _webhook_error_type(exc),
                },
            }
            _request_logger.exception(
                "webhook job response %s",
                _json_log({
                    "job_id": job_id,
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                    "error_type": _webhook_error_type(exc),
                    "error_message": str(exc),
                }),
            )

        delivery_started_at = time.perf_counter()
        payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        _request_logger.info(
            "webhook delivery request %s",
            _json_log({
                "job_id": job_id,
                "webhook_url": _safe_webhook_url(webhook_url),
                "payload_status": payload.get("status"),
                "payload_bytes": payload_bytes,
                "credential_configured": bool(JOB_WEBHOOK_CREDENTIAL),
                "timeout_seconds": JOB_WEBHOOK_TIMEOUT_SECONDS,
            }),
        )
        try:
            status_code = _post_job_webhook(webhook_url, payload)
            _request_logger.info(
                "webhook delivery response %s",
                _json_log({
                    "job_id": job_id,
                    "webhook_url": _safe_webhook_url(webhook_url),
                    "status_code": status_code,
                    "duration_ms": round((time.perf_counter() - delivery_started_at) * 1000, 2),
                }),
            )
        except Exception:
            _request_logger.exception(
                "webhook delivery response %s",
                _json_log({
                    "job_id": job_id,
                    "webhook_url": _safe_webhook_url(webhook_url),
                    "status": "failed",
                    "duration_ms": round((time.perf_counter() - delivery_started_at) * 1000, 2),
                }),
            )

    def _safe_upload_filename(upload: UploadFile, index: int) -> str:
        raw = (upload.filename or "").strip().replace("\\", "/")
        name = raw.rsplit("/", 1)[-1].strip()
        return name or f"file-{index}"

    def _summary_context_content_item(context: str | None) -> dict[str, str]:
        metadata = context.strip() if isinstance(context, str) and context.strip() else "(none provided)"
        return {
            "type": "input_text",
            "text": (
                "Create one integrated summary of the attached files.\n"
                "The context below is user-provided metadata for the entire file bundle, not a summary instruction.\n\n"
                f"Context metadata:\n{metadata}"
            ),
        }

    def _summary_file_kind(filename: str, content_type: str | None) -> tuple[str | None, str]:
        guessed_type = mimetypes.guess_type(filename)[0]
        normalized = (content_type or guessed_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        if normalized == "application/octet-stream" and guessed_type:
            normalized = guessed_type.lower()
        ext = os.path.splitext(filename.lower())[1]
        if normalized == "application/pdf" or ext == ".pdf":
            return "pdf", "application/pdf"
        if normalized.startswith("image/"):
            return "image", normalized
        is_text = (
            normalized.startswith("text/")
            or normalized in SUMMARY_TEXT_MIME_TYPES
            or ext in SUMMARY_TEXT_EXTENSIONS
        )
        if is_text:
            if normalized == "application/octet-stream":
                normalized = "text/plain"
            return "text", normalized
        return None, normalized or "application/octet-stream"

    async def _read_summary_upload(upload: UploadFile, filename: str) -> bytes:
        try:
            data = await upload.read(MAX_SUMMARY_FILE_BYTES + 1)
        finally:
            await upload.close()
        if len(data) > MAX_SUMMARY_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"file '{filename}' exceeds the {MAX_SUMMARY_FILE_BYTES} byte limit",
            )
        return data

    async def _summary_content_from_uploads(
        files: list[UploadFile | StarletteUploadFile],
        context: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not files:
            raise HTTPException(status_code=400, detail="at least one file is required")
        if len(files) > MAX_SUMMARY_FILES:
            raise HTTPException(status_code=400, detail=f"at most {MAX_SUMMARY_FILES} files are allowed")

        content_items: list[dict[str, Any]] = [_summary_context_content_item(context)]
        file_metadata: list[dict[str, Any]] = []

        for index, upload in enumerate(files, start=1):
            filename = _safe_upload_filename(upload, index)
            kind, content_type = _summary_file_kind(filename, upload.content_type)
            if kind is None:
                raise HTTPException(
                    status_code=415,
                    detail=f"unsupported file type for '{filename}': {content_type}",
                )

            data = await _read_summary_upload(upload, filename)
            size_bytes = len(data)
            file_metadata.append({
                "filename": filename,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "kind": kind,
            })

            file_label = (
                f"File {index}: {filename}\n"
                f"Kind: {kind}\n"
                f"Content-Type: {content_type}\n"
                f"Size: {size_bytes} bytes"
            )

            if kind == "text":
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"text file '{filename}' must be valid UTF-8",
                    ) from exc
                content_items.append({
                    "type": "input_text",
                    "text": f"{file_label}\n\nText content:\n{text}",
                })
            elif kind == "image":
                encoded = base64.b64encode(data).decode("ascii")
                content_items.append({"type": "input_text", "text": file_label})
                content_items.append({"type": "input_image", "image_url": f"data:{content_type};base64,{encoded}"})
            elif kind == "pdf":
                encoded = base64.b64encode(data).decode("ascii")
                content_items.append({"type": "input_text", "text": file_label})
                content_items.append({
                    "type": "input_file",
                    "filename": filename,
                    "file_data": f"data:application/pdf;base64,{encoded}",
                })

        return content_items, file_metadata

    def _summary_form_string(form: FormData, field: str, *, required: bool) -> str | None:
        value = form.get(field)
        if value is None:
            if required:
                raise HTTPException(status_code=400, detail=f"{field} is required")
            return None
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"{field} must be a string")
        stripped = value.strip()
        if required and stripped == "":
            raise HTTPException(status_code=400, detail=f"{field} is required")
        return stripped if required else value

    def _summary_form_files(form: FormData) -> list[UploadFile | StarletteUploadFile]:
        values = form.getlist("files")
        files = [value for value in values if isinstance(value, (UploadFile, StarletteUploadFile))]
        if len(files) != len(values):
            raise HTTPException(status_code=400, detail="files must be uploaded as multipart file fields")
        return files

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "service": "codex-as-api",
            "status": "ok",
            "model": MODEL,
            **_auth_status(),
            **_api_auth_status(),
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL,
            **_auth_status(),
            **_api_auth_status(),
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        http_request: Request,
    ) -> JSONResponse | StreamingResponse:
        _require_api_key(http_request)
        _require_auth()

        if request.stream:
            provider = _get_provider()
            messages = _request_messages_to_internal(request.messages)
            tools = _parse_tools(request.tools)
            stop = _normalize_stop(request.stop)
            max_tokens = _max_tokens_from_request(request)

            subagent = request.subagent or http_request.headers.get("x-openai-subagent")
            memgen_request_header = http_request.headers.get("x-openai-memgen-request")
            memgen_request: bool | None = request.memgen_request
            if memgen_request is None and memgen_request_header is not None:
                memgen_request = memgen_request_header.lower() not in ("false", "0", "")
            previous_response_id = request.previous_response_id

            async def _stream() -> AsyncIterator[str]:
                request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())
                model = _openai_model_id(request.model)

                # SSE preamble
                preamble = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(preamble)}\n\n"

                reasoning_parts: list[str] = []
                content_parts: list[str] = []
                tool_calls_buffer: list[dict[str, Any]] = []
                usage_dict: dict[str, Any] | None = None

                for event in provider.chat_stream(
                    messages,
                    model=request.model,
                    tools=tools,
                    tool_choice=request.tool_choice,
                    temperature=request.temperature,
                    reasoning_effort=request.reasoning_effort,
                    max_tokens=max_tokens,
                    stop=stop,
                    prompt_cache_key=request.prompt_cache_key,
                    subagent=subagent,
                    memgen_request=memgen_request,
                    previous_response_id=previous_response_id,
                    service_tier=request.service_tier,
                    text=request.text,
                    client_metadata=request.client_metadata,
                ):
                    typ = event.get("type")
                    if typ == "content":
                        text = str(event.get("text", ""))
                        content_parts.append(text)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "reasoning_delta":
                        text = str(event.get("text", ""))
                        reasoning_parts.append(text)
                        # OpenAI-compatible reasoning field
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning_content": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "reasoning_raw_delta":
                        text = str(event.get("text", ""))
                        reasoning_parts.append(text)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"reasoning": text},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "tool_call":
                        tc = {
                            "id": event.get("id"),
                            "type": "function",
                            "function": {
                                "name": event.get("name"),
                                "arguments": json.dumps(event.get("arguments") or {}),
                            },
                        }
                        tool_calls_buffer.append(tc)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [tc]},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif typ == "finish":
                        usage = event.get("usage")
                        if isinstance(usage, dict):
                            usage_dict = usage
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": event.get("finish_reason") or "stop",
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Usage summary chunk if available
                if usage_dict:
                    u = usage_dict
                    finish_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": u.get("prompt_tokens", 0),
                            "completion_tokens": u.get("completion_tokens", 0),
                            "total_tokens": u.get("total_tokens", 0),
                        },
                    }
                    yield f"data: {json.dumps(finish_chunk)}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(_stream(), media_type="text/event-stream")

        # Non-streaming
        result = _chat_completion_response_payload(request, http_request.headers)
        return JSONResponse(content=result)

    @app.post("/v1/jobs/chat/completions")
    async def chat_completions_webhook(
        request: WebhookChatCompletionRequest,
        http_request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        _require_api_key(http_request)
        _require_auth()
        job_id = f"job-{uuid.uuid4().hex[:24]}"
        request_headers = dict(http_request.headers)
        _request_logger.info(
            "webhook job request %s",
            _json_log(_chat_request_log_fields(job_id=job_id, request=request, headers=request_headers)),
        )
        if request.stream:
            _request_logger.info(
                "webhook job queued response %s",
                _json_log({
                    "job_id": job_id,
                    "http_status": 400,
                    "status": "rejected",
                    "reason": "stream_not_supported",
                }),
            )
            raise HTTPException(status_code=400, detail="stream is not supported for webhook jobs")
        try:
            _validate_webhook_url(request.webhook_url)
        except HTTPException:
            _request_logger.info(
                "webhook job queued response %s",
                _json_log({
                    "job_id": job_id,
                    "http_status": 400,
                    "status": "rejected",
                    "reason": "invalid_webhook_url",
                }),
            )
            raise

        background_tasks.add_task(
            _run_chat_completion_webhook_job,
            job_id,
            request,
            request_headers,
            request.webhook_url,
        )
        _request_logger.info(
            "webhook job queued response %s",
            _json_log({
                "job_id": job_id,
                "http_status": 202,
                "status": "queued",
                "object": "chat.completion.webhook.job",
            }),
        )
        return JSONResponse(
            status_code=202,
            content={
                "id": job_id,
                "object": "chat.completion.webhook.job",
                "status": "queued",
            },
        )

    @app.post("/v1/images/generations")
    async def images_generations(request: ImageGenerationRequest, http_request: Request) -> JSONResponse:
        _require_api_key(http_request)
        _require_auth()
        provider = _get_provider()
        images = provider.generate_image(
            request.prompt,
            model=request.model,
            size=request.size,
            reasoning_effort=request.reasoning_effort,
        )
        data = [
            {
                "url": image.get("result"),
                "revised_prompt": image.get("revised_prompt") or request.prompt,
            }
            for image in images
            if image.get("result")
        ]
        return JSONResponse(content={"created": int(time.time()), "data": data})

    @app.post("/v1/files/summarize")
    async def files_summarize(http_request: Request) -> JSONResponse:
        _require_api_key(http_request)
        _require_auth()
        form = await http_request.form()
        try:
            request_model = _summary_form_string(form, "model", required=True)
            context = _summary_form_string(form, "context", required=False)
            reasoning_effort = _summary_form_string(form, "reasoning_effort", required=False)
            files = _summary_form_files(form)
            content_items, file_metadata = await _summary_content_from_uploads(files, context)
        finally:
            await form.close()
        provider = _get_provider()
        summary = provider.summarize_files(
            content_items,
            model=request_model or MODEL,
            reasoning_effort=reasoning_effort,
        )
        return JSONResponse(
            content={
                "summary": summary,
                "model": _openai_model_id(request_model or MODEL),
                "files": file_metadata,
            }
        )

    # ------------------------------------------------------------------
    # Anthropic Messages API compatible endpoint
    # ------------------------------------------------------------------

    from .anthropic_adapter import (
        anthropic_request_to_internal,
        anthropic_stream_adapter,
        format_anthropic_error,
        internal_response_to_anthropic,
    )

    @app.post("/v1/messages", response_model=None)
    async def anthropic_messages(http_request: Request) -> JSONResponse | StreamingResponse:
        _require_api_key(http_request)
        _require_auth()
        provider = _get_provider()
        body = await http_request.json()

        try:
            messages, tools, tool_choice, stop, reasoning_effort = anthropic_request_to_internal(
                model=body.get("model", MODEL),
                messages=body.get("messages", []),
                system=body.get("system"),
                max_tokens=body.get("max_tokens", 4096),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                stop_sequences=body.get("stop_sequences"),
                thinking=body.get("thinking"),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content=format_anthropic_error(400, str(exc)))

        stream = body.get("stream", False)
        request_model = body.get("model") or MODEL

        if stream:
            async def _stream() -> AsyncIterator[str]:
                request_id = f"msg_{uuid.uuid4().hex[:24]}"
                for sse_chunk in anthropic_stream_adapter(
                    provider.chat_stream(
                        messages,
                        model=request_model,
                        tools=tools,
                        tool_choice=tool_choice,
                        reasoning_effort=reasoning_effort,
                        stop=stop,
                    ),
                    model=request_model,
                    request_id=request_id,
                ):
                    yield sse_chunk

            try:
                return StreamingResponse(_stream(), media_type="text/event-stream")
            except ChatGPTOAuthError as exc:
                status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
                return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        # Non-streaming
        try:
            response = provider.chat(
                messages,
                model=request_model,
                tools=tools,
                tool_choice=tool_choice,
                reasoning_effort=reasoning_effort,
                stop=stop,
            )
        except ChatGPTOAuthError as exc:
            status = 401 if isinstance(exc, ChatGPTOAuthMissingError) else 500
            return JSONResponse(status_code=status, content=format_anthropic_error(status, str(exc)))

        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        result = internal_response_to_anthropic(response, request_model, request_id)
        return JSONResponse(content=result)

    # ------------------------------------------------------------------
    # Custom endpoints (not in standard OpenAI API, but exposed for full feature routing)
    # ------------------------------------------------------------------

    @app.post("/v1/inspect")
    async def inspect(request: Request) -> JSONResponse:
        """Inspect images with a text prompt.

        Body: {"prompt": str, "images": [{"image_url": "data:image/..."}, ...], "reasoning_effort": str?}
        """
        _require_api_key(request)
        _require_auth()
        provider = _get_provider()
        body = await request.json()
        prompt = str(body.get("prompt", ""))
        images = body.get("images") or []
        reasoning_effort = body.get("reasoning_effort")
        result = provider.inspect_images(prompt, images=images, reasoning_effort=reasoning_effort)
        return JSONResponse(content={"content": result})

    @app.post("/v1/compact")
    async def compact(request: Request) -> JSONResponse:
        """Compact a conversation into a checkpoint for continuation.

        Body: {"messages": [{"role": "system|user|assistant|tool", "content": str, ...}], "reasoning_effort": str?}
        """
        _require_api_key(request)
        _require_auth()
        provider = _get_provider()
        body = await request.json()
        raw_messages = body.get("messages") or []
        messages = _request_messages_to_internal([ChatMessage.model_validate(m) for m in raw_messages])
        reasoning_effort = body.get("reasoning_effort")
        checkpoint = provider.compact_messages(messages, reasoning_effort=reasoning_effort)
        return JSONResponse(content={"checkpoint": checkpoint})

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def main() -> None:
        import uvicorn
        uvicorn.run("codex_as_api.server:app", host=HOST, port=PORT, log_level="info")

except ImportError as _import_exc:
    # FastAPI / uvicorn not installed
    _fastapi_import_error = _import_exc
    app = None  # type: ignore[assignment]

    def main() -> None:  # type: ignore[misc]
        raise ImportError(
            "FastAPI and uvicorn are required to run the server. "
            "Install with: pip install 'codex-as-api[server]'"
        ) from _fastapi_import_error

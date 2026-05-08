FROM node:22-bookworm-slim AS codex-cli

ARG CODEX_CLI_VERSION=0.129.0

RUN npm install -g "@openai/codex@${CODEX_CLI_VERSION}" \
    && codex --version

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CODEX_HOME=/codex-home \
    CODEX_AS_API_HOST=0.0.0.0 \
    CODEX_AS_API_PORT=18080 \
    CODEX_AS_API_WORKERS=2 \
    CODEX_AS_API_WORKER_TIMEOUT=0 \
    CODEX_AS_API_GRACEFUL_TIMEOUT=30 \
    CODEX_AS_API_KEEP_ALIVE=5

COPY --from=codex-cli /usr/local/bin/node /usr/local/bin/node
COPY --from=codex-cli /usr/local/lib/node_modules/@openai/codex /usr/local/lib/node_modules/@openai/codex

RUN mkdir -p /codex-home \
    && ln -sf /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && codex --version

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install ".[server]"

EXPOSE 18080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('PORT') or os.environ.get('CODEX_AS_API_PORT', '18080'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3).read()" || exit 1

CMD ["sh", "-c", "APP_PORT=\"${PORT:-${CODEX_AS_API_PORT:-18080}}\"; exec gunicorn codex_as_api.server:app --worker-class uvicorn_worker.UvicornWorker --workers ${CODEX_AS_API_WORKERS:-2} --bind 0.0.0.0:${APP_PORT} --timeout ${CODEX_AS_API_WORKER_TIMEOUT:-0} --graceful-timeout ${CODEX_AS_API_GRACEFUL_TIMEOUT:-30} --keep-alive ${CODEX_AS_API_KEEP_ALIVE:-5} --access-logfile - --error-logfile -"]

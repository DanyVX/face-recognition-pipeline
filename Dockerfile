# syntax=docker/dockerfile:1
# NOTE: written on a machine without Docker; first exercised by the CI `docker` job.
# Model weights are NOT baked into the image (non-commercial InsightFace licence, ~280 MB):
# the entrypoint downloads and checksum-verifies them into the /models volume on first start.

FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached until the lockfile changes), then the project itself.
# --no-default-groups: skip the dev and bench groups; --no-editable: the runtime stage only gets
# the virtualenv, so the package must be installed into it rather than linked to /app/src.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

FROM python:3.12-slim-bookworm
# libgomp1: OpenMP runtime used by onnxruntime / faiss on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 frp \
    && useradd --system --uid 10001 --gid frp --home-dir /nonexistent --shell /usr/sbin/nologin frp \
    && mkdir /data /models /keys \
    && chown frp:frp /data /models /keys
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY scripts/download_models.py scripts/docker-entrypoint.sh ./scripts/
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/data \
    MODEL_DIR=/models \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_FORMAT=json
USER frp
VOLUME ["/data", "/models", "/keys"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4).status == 200 else 1)"
ENTRYPOINT ["sh", "/app/scripts/docker-entrypoint.sh"]
CMD ["frp", "serve"]

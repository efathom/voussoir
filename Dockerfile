# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# voussoir — production image
#
# Build:
#   docker build -t voussoir:1.3.0 .
#
# Run (CLI entry point):
#   docker run --rm -e ANTHROPIC_API_KEY voussoir:1.3.0 doctor
#
# voussoir depends on the sibling `ctxforge` repo (a non-PyPI path dependency;
# see [tool.uv.sources] in pyproject.toml). This image clones ctxforge at a
# pinned commit next to voussoir so `uv sync` can resolve it. Override with:
#   docker build --build-arg CTXFORGE_REF=<commit> .
# ---------------------------------------------------------------------------

# ---- builder: install dependencies into a relocatable .venv ----------------
FROM python:3.12-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv — fast, lockfile-driven Python package installer.
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

# git is only needed to fetch ctxforge (see [tool.uv.sources]).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ctxforge — voussoir's only non-PyPI dependency, pinned for reproducibility.
ARG CTXFORGE_REF=59779203e054f8744c9f5a763acb93e74c3254a3
RUN git clone --quiet https://github.com/efathom/ctxforge.git /opt/ctxforge \
    && git -C /opt/ctxforge checkout --quiet "${CTXFORGE_REF}"

# Copy project manifests first (better layer caching), then the package source.
COPY pyproject.toml uv.lock README.md /opt/voussoir/
COPY src /opt/voussoir/src/

WORKDIR /opt/voussoir

# a2a + mcp cover the network-facing deployment modes. Add mem-sqlite /
# mem-postgres / mem-qdrant / tokenizers etc. here for richer backends.
RUN uv sync --frozen --no-dev --extra a2a --extra mcp

# ---- runtime: slim image with just the app + venv ---------------------------
FROM python:3.12-slim

ENV PATH="/opt/voussoir/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Non-root user for defense-in-depth.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

WORKDIR /opt/voussoir

# Copy the built venv + source, and ctxforge (kept as an editable install so
# the runtime import path resolves exactly as it does in development).
COPY --from=builder --chown=appuser:appuser /opt/voussoir /opt/voussoir
COPY --from=builder --chown=appuser:appuser /opt/ctxforge /opt/ctxforge

USER appuser

# `voussoir` is the console script installed in the venv (see [project.scripts]).
# Override CMD to run a specific subcommand, e.g. `voussoir doctor`.
ENTRYPOINT ["voussoir"]
CMD ["--help"]

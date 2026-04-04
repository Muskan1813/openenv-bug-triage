# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL name="bug-triage-env" \
      version="1.0.0" \
      description="OpenEnv Bug Triage and Code Review Environment"

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY server/      ./server/
COPY data/        ./data/
COPY openenv.yaml .

# ── Environment variables ─────────────────────────────────────────────────────
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "server.main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
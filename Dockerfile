FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA libraries (gigabytes)
# that a CPU embedding model never uses.
# --extra-index-url (NOT --index-url) keeps PyPI available: torch's small
# dependencies and build tools are fetched from there.
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY adaptive_rag ./adaptive_rag

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/storage /app/cache /app/.cache/huggingface \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The first start downloads the embedding model, so allow time before checking.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# One worker only: the engine keeps its index, locks and caches in this process.
CMD ["python", "-m", "adaptive_rag.api"]
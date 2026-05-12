# ──────────────────────────────────────────────────────────────────────────────
# PhishSentinel — HuggingFace Spaces Dockerfile
#
# • Python 3.11-slim base (stable, well-tested on HF Spaces)
# • CPU-only PyTorch installed first to prevent the 2 GB CUDA wheel download
# • Runs as non-root user `user` (UID 1000) — required by HF Spaces security
# • Streamlit served on port 7860 (HF Spaces default)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user required by HF Spaces ──────────────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# ── Python dependencies ───────────────────────────────────────────────────────
# Install CPU-only PyTorch BEFORE requirements.txt so pip does not pull the
# full ~2 GB CUDA wheel when it sees `torch>=2.4.0` in requirements.txt.
RUN pip install --no-cache-dir \
        torch \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY --chown=user . .

# ── Runtime ───────────────────────────────────────────────────────────────────
USER user
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

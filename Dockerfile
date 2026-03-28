# =============================================================================
# Clinical Output Evaluation Framework — Dockerfile
# =============================================================================
# Runs the rubric-based LLM-as-judge evaluation harness for clinical AI
# outputs. Provides a Click CLI for evaluating outputs, comparing prompt
# versions, generating reports, and viewing evaluation history.
#
# Build:  docker build -t clinical-eval-framework .
# Run:    docker run -e OPENAI_API_KEY=sk-... clinical-eval-framework eval --input data/sample_outputs/
# =============================================================================

# --- Stage 1: Base image ---
# Use Python 3.11 slim for a smaller image footprint.
FROM python:3.11-slim AS base

# --- Stage 2: Set working directory ---
WORKDIR /app

# --- Stage 3: Install system dependencies ---
# Build tools needed for native Python package compilation.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 4: Install Python dependencies ---
# Copy requirements.txt first to leverage Docker layer caching.
# Dependencies change less frequently than source code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy application source code ---
# Copy source, data, rubrics, and output directories.
COPY src/ src/
COPY data/ data/
COPY rubrics/ rubrics/
COPY outputs/ outputs/

# --- Stage 6: Create non-root user for security ---
# Running as root inside a container is a security risk.
# The appuser needs write access to outputs/ and the SQLite database.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# --- Stage 7: Configure environment ---
# OPENAI_API_KEY is passed at runtime for real (non-simulated) evaluation.
# The demo/simulated mode works without an API key.
ENV PYTHONUNBUFFERED=1

# --- Stage 8: Set the CLI as the entry point ---
# The CLI is invoked via `python src/cli.py <command>`.
# Users pass commands and flags after the image name:
#   docker run clinical-eval-framework eval --input data/sample_outputs/
#   docker run clinical-eval-framework compare run_v1 run_v2
#   docker run clinical-eval-framework history
ENTRYPOINT ["python", "src/cli.py"]

# --- Stage 9: Default command ---
# Show help if no command is provided.
CMD ["--help"]

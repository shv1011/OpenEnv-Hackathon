# School Timetable Scheduling Environment
# =========================================

FROM python:3.11-slim AS base

LABEL maintainer="OpenEnv Hackathon"
LABEL description="School Admin Timetable Scheduling Environment"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# ─────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────
# Application
# ─────────────────────────────────────────
FROM dependencies AS app

# Copy env package and inference script
COPY env/ ./env/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Output directories
RUN mkdir -p timetables logs

# Environment variable defaults (override at runtime)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# ─────────────────────────────────────────
# Entrypoint — runs ALL tasks (easy, medium, hard)
# ─────────────────────────────────────────
ENTRYPOINT ["python", "inference.py"]

# ─────────────────────────────────────────
# Usage:
#   docker build -t school-timetable-env .
#   docker run --rm -e HF_TOKEN=hf_... school-timetable-env
# ─────────────────────────────────────────

# School Timetable Scheduling Environment
# =========================================
# Multi-stage Docker build for a clean, production-grade container

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

# Copy source
COPY env/ ./env/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Output directories
RUN mkdir -p timetables logs

# Environment variable defaults (override at runtime)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o

# ─────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────
ENTRYPOINT ["python", "inference.py"]
CMD ["--task", "easy", "--export-csv"]

# ─────────────────────────────────────────
# Usage Examples:
#
#   Build:
#     docker build -t school-timetable-env .
#
#   Run (easy task):
#     docker run --rm \
#       -e OPENAI_API_KEY=sk-... \
#       -v $(pwd)/timetables:/app/timetables \
#       school-timetable-env --task easy --export-csv
#
#   Run (hard task with debug):
#     docker run --rm \
#       -e OPENAI_API_KEY=sk-... \
#       -e MODEL_NAME=gpt-4o \
#       -v $(pwd)/timetables:/app/timetables \
#       school-timetable-env --task hard --debug --export-csv
#
#   Run with custom API endpoint (e.g. Azure, Groq):
#     docker run --rm \
#       -e OPENAI_API_KEY=... \
#       -e API_BASE_URL=https://api.groq.com/openai/v1 \
#       -e MODEL_NAME=llama-3.1-70b-versatile \
#       school-timetable-env --task medium
# ─────────────────────────────────────────

FROM python:3.11-slim

LABEL maintainer="OpenEnv Hackathon"
LABEL description="School Timetable Scheduling — OpenEnv Environment"
LABEL version="1.0.0"

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY env/        ./env/
COPY server/     ./server/
COPY static/     ./static/
COPY models.py   environment.py client.py openenv.yaml README.md ./

# HF Spaces runs on port 7860
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=64
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

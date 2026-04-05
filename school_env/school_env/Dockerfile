FROM python:3.11-slim

LABEL description="School Timetable Scheduling — OpenEnv Environment"
LABEL version="1.0.0"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy ALL source files needed by both server and inference.py
COPY env/           ./env/
COPY server/        ./server/
COPY static/        ./static/
COPY models.py      ./models.py
COPY environment.py ./environment.py
COPY client.py      ./client.py
COPY inference.py   ./inference.py
COPY openenv.yaml   ./openenv.yaml
COPY README.md      ./README.md

ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV MAX_CONCURRENT_ENVS=64
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

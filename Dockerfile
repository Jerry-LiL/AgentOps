FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AGENTOPS_LLM_MODE=local \
    AGENTOPS_DATA_DIR=/data \
    AGENTOPS_DB_PATH=/data/agentops.sqlite

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /data

EXPOSE 8000

CMD ["python", "-m", "agentops"]

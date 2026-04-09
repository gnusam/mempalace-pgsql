FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# pyproject.toml declares README.md as the package readme — hatchling
# fails metadata generation without it, so we have to copy both.
COPY pyproject.toml README.md ./
COPY mempalace/ ./mempalace/

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir . \
    psycopg2-binary pgvector \
    sentence-transformers

ENV DATABASE_URL=postgresql://mempalace:mempalace@postgres:5432/mempalace

ENTRYPOINT ["python", "-m"]
CMD ["mempalace.mcp_server"]

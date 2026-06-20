FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# StatsBomb cache is pre-bundled so cold starts don't re-fetch 64 match files
# (.dockerignore excludes __pycache__ and .git but keeps .squad_cache)

ENV PORT=8080
EXPOSE 8080

# 1 worker — simulation is CPU-bound; multiple workers would just compete
CMD ["gunicorn", "--workers=1", "--threads=4", "--bind=0.0.0.0:8080", "--timeout=120", "app:app"]

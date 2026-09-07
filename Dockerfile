# Dockerfile for Google Product Search (Production / Render deployment)
# -------------------------------------------------
# 1️⃣ Base image – Python 3.11 slim
# -------------------------------------------------
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PORT=8000

WORKDIR /app

# -------------------------------------------------
# 2️⃣ Install basic system dependencies
# -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 3️⃣ Cache Python dependencies and Playwright Chromium
# -------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

# -------------------------------------------------
# 4️⃣ Create non-root user and grant browser permissions
# -------------------------------------------------
RUN useradd -m -u 1000 appuser && \
    chmod -R 755 /ms-playwright

# -------------------------------------------------
# 5️⃣ Copy application code with non-root ownership
# -------------------------------------------------
COPY . /app
RUN chown -R appuser:appuser /app

# -------------------------------------------------
# 6️⃣ Switch to non-root user and expose port
# -------------------------------------------------
USER appuser
EXPOSE 8000

# -------------------------------------------------
# 7️⃣ Start FastAPI server
# -------------------------------------------------
CMD ["python", "src/main.py", "--serve", "--host", "0.0.0.0", "--port", "8000"]

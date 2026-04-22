# --------------------------------------------------
# Base image
# --------------------------------------------------
FROM python:3.11-slim

# --------------------------------------------------
# Set working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# Environment variables
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# Copy runtime requirements first for caching
# --------------------------------------------------
COPY requirements-docker.txt .

# --------------------------------------------------
# Install runtime dependencies
# --------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# --------------------------------------------------
# Copy app code and artifacts
# --------------------------------------------------
COPY app ./app
COPY src ./src
COPY artifacts ./artifacts
COPY mlruns ./mlruns

# --------------------------------------------------
# Expose FastAPI port
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Start FastAPI app
# --------------------------------------------------
CMD ["uvicorn", "app.fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
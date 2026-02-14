# ─── Dockerfile for Predictive Maintenance Streamlit App ───
# Build: docker build -t predictive-maintenance-app .
# Run:   docker run -p 8501:8501 predictive-maintenance-app
# Optional: mount a local model for offline use:
#   docker run -p 8501:8501 -v $(pwd)/models:/app/models predictive-maintenance-app

FROM python:3.11-slim

WORKDIR /app

# System deps: build-essential for any pip wheels, curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Optional: copy local model so the app can run without Hugging Face at runtime
# Uncomment and ensure models/ exists when building with a pre-downloaded model:
# COPY models/best_random_forest.pkl models/

# HF Spaces Docker default is app_port 7860; proxy forwards to this port
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

# Dockerfile - Turbofan RUL Predictor
# Build:  docker build -t turbofan-rul-app .
# Run:    docker run -p 8501:8501 turbofan-rul-app
# Then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer = faster rebuilds when only
# app code changes, since Docker caches this layer if requirements.txt
# hasn't changed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the app + model files
COPY turbofan_app.py .
COPY rul_model.pkl .
COPY feature_cols.pkl .
COPY sensor_defaults.pkl .

EXPOSE 8501

# --server.address=0.0.0.0 is required inside Docker - without it, Streamlit
# binds to localhost INSIDE the container, which is unreachable from outside
CMD ["streamlit", "run", "turbofan_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

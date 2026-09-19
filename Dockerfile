# Credit Card Customer Intelligence Platform:FastAPI serving image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer, only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY src/ ./src/
COPY fastapi_app/ ./fastapi_app/

# Only the trained model/preprocessing artifacts the app actually loads at startup not the 
# raw/train/test/retention CSVs, which aren't needed to serve predictions and would only bloat the image.
COPY artifacts/kmeans_model.pkl ./artifacts/kmeans_model.pkl
COPY artifacts/scaler.pkl ./artifacts/scaler.pkl
COPY artifacts/classification_preprocessor.pkl ./artifacts/classification_preprocessor.pkl
COPY artifacts/classification_model.pkl ./artifacts/classification_model.pkl

EXPOSE 8000

# Start the FastAPI app with uvicorn, using the PORT environment variable if set, defaulting to 8000
CMD uvicorn fastapi_app.main:app --host 0.0.0.0 --port ${PORT:-8000}
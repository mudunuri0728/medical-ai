FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src \
    API_URL=http://localhost:8000

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY frontend ./frontend
COPY src ./src
COPY uploads/.gitkeep ./uploads/.gitkeep

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.address=0.0.0.0 --server.port=8501 --server.enableCORS=false --server.enableXsrfProtection=false"]

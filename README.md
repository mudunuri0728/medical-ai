# Medical AI

Medical document analysis system with:
- `FastAPI` backend for file upload and analysis
- `Streamlit` frontend for user interaction
- OCR via `LlamaParse`
- LLM-based document understanding via OpenRouter-compatible OpenAI SDK

## Project Structure

```text
medical-ai/
  frontend/
    app.py
  src/
    main.py
    analysis.py
    ocr.py
    pdfconverter.py
    config.py
  uploads/
  requirements.txt
```

## Prerequisites

- Python 3.10+
- Git

## Setup (Git Bash on Windows)

```bash
cd /c/Users/ymudu/medical-ai
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openrouter_or_openai_key
LLAMA_API_KEY=your_llamaparse_key
```

## Run the Backend API

From project root:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`

## Run the Streamlit Frontend

Open a second terminal in project root:

```bash
source .venv/Scripts/activate
streamlit run frontend/app.py
```

Frontend URL: `http://localhost:8501`

## Run with Docker Compose

Build and start the single app container (backend + frontend):

```bash
docker compose up --build
```

Stop services:

```bash
docker compose down
```

Frontend: `http://localhost:8501`  
Backend: `http://localhost:8000`

## API Endpoint

- `POST /analyze`
- Form field: `files` (multiple files supported)
- Supported formats: `.pdf`, `.png`, `.jpg`, `.jpeg`

## Notes

- Uploaded files are stored under `uploads/`.
- `.env` is ignored by git for security.
- `uploads/.gitkeep` keeps the folder tracked while ignoring generated files.

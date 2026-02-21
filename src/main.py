from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import asyncio
import os
from src.analysis import classify_document

app = FastAPI(title="Medical Document Analysis API")

# ✅ CORS CONFIGURATION (IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):

    async def process(file: UploadFile) -> Dict[str, Any]:
        path = os.path.join(UPLOAD_DIR, file.filename)

        contents = await file.read()
        with open(path, "wb") as f:
            f.write(contents)

        try:
            result = await classify_document(path)

            return {
                "file": file.filename,
                "analysis": result
            }

        except Exception as e:
            return {
                "file": file.filename,
                "analysis": {
                    "document_status": "FAILED",
                    "failure_reason": f"System Error: {str(e)}"
                }
            }

    results = await asyncio.gather(*[process(f) for f in files])
    return JSONResponse(content=results)
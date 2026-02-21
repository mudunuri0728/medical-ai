import os
import asyncio
from typing import Dict

from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

from src.config import MAX_CONCURRENT_OCR

from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# CONCURRENCY CONTROL
# --------------------------------------------------
# Limits the number of simultaneous OCR operations
ocr_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    MAX_CONCURRENT_OCR
)

# --------------------------------------------------
# OCR PARSER INITIALIZATION
# --------------------------------------------------
# LlamaParse internally reads the API key from environment
parser: LlamaParse = LlamaParse(
    api_key=os.getenv("LLAMA_API_KEY"),
    result_type="text"
)

# --------------------------------------------------
# FILE EXTENSION HANDLERS
# --------------------------------------------------
file_extractor: Dict[str, LlamaParse] = {
    ".jpg": parser,
    ".jpeg": parser,
    ".png": parser,
    ".pdf": parser,
}


# --------------------------------------------------
# ASYNCHRONOUS OCR EXTRACTION
# --------------------------------------------------
async def extract_text_from_image_async(file_path: str) -> str:
    """
    Extract text asynchronously from an image or PDF using LlamaParse.

    Concurrency is limited using a semaphore to prevent excessive
    parallel OCR requests.

    Parameters
    ----------
    file_path : str
        Path to the image or PDF file.

    Returns
    -------
    str
        Extracted text content, or an empty string on failure.
    """
    async with ocr_semaphore:
        try:
            documents = await asyncio.to_thread(
                lambda: SimpleDirectoryReader(
                    input_files=[file_path],
                    file_extractor=file_extractor
                ).load_data()
            )

            return "\n".join(doc.text for doc in documents).strip()

        except Exception as exc:
            print(f"OCR failed for {file_path}: {exc}")
            return ""

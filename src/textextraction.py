
import os
import gc
from typing import Dict
from pathlib import Path

from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader

from dotenv import load_dotenv
load_dotenv()


# --------------------------------------------------
# FILE EXTENSION HANDLERS
# --------------------------------------------------
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
MIN_FILE_SIZE = 1024  # 1 KB minimum
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB maximum


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------
def _validate_file(file_path: str) -> tuple[bool, str]:
    """Validate file before processing."""
    try:
        if not os.path.exists(file_path):
            return False, f"File not found"
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file type: {file_ext}"
        
        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_SIZE:
            return False, f"File too small"
        if file_size > MAX_FILE_SIZE:
            return False, f"File too large"
        
        if not os.access(file_path, os.R_OK):
            return False, "File not readable"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"


# --------------------------------------------------
# SYNCHRONOUS OCR EXTRACTION
# --------------------------------------------------
def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from image/PDF using LlamaParse (PURE SYNC).
    
    KEY: Creates FRESH parser for each image to avoid event loop conflicts.
    Old parser's event loop closes cleanly, new one starts fresh.
    """
    try:
        is_valid, msg = _validate_file(file_path)
        if not is_valid:
            print(f"File validation failed: {msg}")
            return ""
        
        filename = os.path.basename(file_path)
        print(f"→ OCR extraction: {filename}")
        
        # CRITICAL: Create FRESH parser for this image (not global)
        # This ensures each image gets a clean event loop context
        fresh_parser = LlamaParse(
            api_key=os.getenv("LLAMA_API_KEY"),
            result_type="text"
        )
        
        # Create extractor dict with fresh parser
        fresh_extractor = {
            ".jpg": fresh_parser,
            ".jpeg": fresh_parser,
            ".png": fresh_parser,
            ".pdf": fresh_parser,
        }
        
        # Extract text with fresh parser
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=fresh_extractor
        ).load_data()
        
        if not documents:
            print(f"✗ No text extracted from {filename}")
            return ""
        
        text = "\n".join(doc.text for doc in documents).strip()
        print(f"✓ OCR success: {filename} ({len(text)} chars)")
        
        # Clean up parser (allows its event loop to close properly)
        del fresh_parser
        del fresh_extractor
        gc.collect()  # Force garbage collection to clean up old event loops
        
        return text
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ OCR failed: {error_msg[:150]}")
        
        # On error, still try to clean up
        try:
            gc.collect()
        except:
            pass
        
        return ""



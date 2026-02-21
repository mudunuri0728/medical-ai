# Maximum number of concurrent OCR requests
MAX_CONCURRENT_OCR: int = 5

# Vision LLM model name (OpenRouter)
VISION_MODEL_NAME: str = "nvidia/nemotron-nano-12b-v2-vl:free"


# PDF to image conversion settings
PDF_IMAGE_DPI: int = 302
PDF_IMAGE_BASE_DIR: str = "uploads/images"


# file upload limitations

UPLOAD_DIR: str = "uploads"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

MAX_TOTAL_FILES: int = 5
MAX_PDFS: int = 3
MAX_IMAGES: int = 5

MAX_IMAGE_MB: int = 5
MAX_PDF_MB: int = 10


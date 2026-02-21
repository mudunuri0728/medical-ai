import os
from typing import Optional
import fitz  # PyMuPDF
from src.config import PDF_IMAGE_DPI, PDF_IMAGE_BASE_DIR


# --------------------------------------------------
# PDF TO IMAGE CONVERSION
# --------------------------------------------------
def pdf_to_images(
    pdf_path: str,
    base_dir: Optional[str] = None
) -> str:
    """
    Convert a multi-page PDF into individual PNG images.

    Each page of the PDF is rendered at a fixed DPI and
    saved as a separate image file inside a directory
    named after the PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF file.
    base_dir : str, optional
        Base directory where page images will be stored.
        Defaults to the configured PDF_IMAGE_BASE_DIR.

    Returns
    -------
    str
        Name of the PDF file (without extension), used
        as the output folder name.
    """

    # Resolve base output directory
    output_base: str = base_dir or PDF_IMAGE_BASE_DIR

    # Extract PDF name (without extension)
    pdf_name: str = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create output directory for this PDF
    output_dir: str = os.path.join(output_base, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    # Open PDF document
    document = fitz.open(pdf_path)

    # Render each page as a high-resolution PNG image
    for page_index, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(dpi=PDF_IMAGE_DPI)
        pixmap.save(
            os.path.join(output_dir, f"page_{page_index}.png")
        )

    # Close document to release resources
    document.close()

    return pdf_name

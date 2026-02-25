import os
import base64
import json
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

from src.textextraction import extract_text_from_image
from src.pdfconverter import pdf_to_images # Integrated PDF conversion logic
from src.config import VISION_MODEL_NAME

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# --------------------------------------------------
# MEDICAL_DOC MASTER PROMPT
# --------------------------------------------------
PROMPT = """
You are the MEDICAL_DOC Analysis AI.

Your task is to analyze medical prescriptions, discharge summaries, medical reports, or hospital documents.

STRICTLY extract and return the following:
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. **BE THOROUGH**: Extract ALL medical information from the document.
2. **BE ACCURATE**: Distinguish clearly between facts found in the document and your medical suggestions.
3. **BE COMPLETE**: Identify all diagnoses, symptoms, medications, and treatment plans.
4. **BE CAREFUL**: Flag missing information and inconsistencies.
5. **BE HELPFUL**: Use simple language for patient understanding.

**SECTION 1 - PATIENT DATA EXTRACTION (CRITICAL):**
- patient_name: Extract the EXACT full name. Look near labels like 'Name', 'Patient', 'MR.', or 'MRS.' even if separated by colons or located on the next line.
- patient_age: Extract age in years (e.g., "45").
- patient_sex: Extract as "Male", "Female", or "Other".
- clinical_info: Write 3-4 sentences covering chief complaint, findings, diagnoses, and symptoms.

**SECTION 2 - SUMMARY_FOR_HUMAN (Patient-friendly narrative):**
- MUST be explain briefly in short note and 10-lines.
- Start with: "The patient, [Name], aged [Age], is a [Male/Female]..."
- Include: Reason for visit → Findings → The plan.

**SECTION 3 - DISEASE_EXPLANATION (Educational but simple):**
- MUST be explain briefly in short note and 10-lines.
- Use analogies patients understand. NEVER say "N/A". 
- If diagnosis is missing, explain the likely condition based on symptoms.

**SECTION 4 - MEDICINE_INFO (Medications & Augmentation):**
Format: "Medicine Name: Dosage - Frequency - Purpose/Indication."
Rules for Generation & Extraction:
- RULE A: List EVERY medication explicitly mentioned in the document. 
- RULE B: Return semi-structured and unstructured medicine information.
- Return as a LIST of strings. Never return "N/A".

**SECTION 5 - HOSPITAL_GUIDE (Next steps):**
- MUST be 7-9 sentences written as a SINGLE continuous paragraph.
- Do not use bullet points or numbered lists.
- Recommend specialist type only if names are not present in the document.

**SECTION 6 - VALIDATION RULES (STRICT AUDIT):**
Analyze the document for these 4 specific compliance elements. Return "Found" ONLY if the element is clearly present:

1. **Patient Name**: Look for patient's full name or at least first and last name. Valid indicators: "Name:", "Patient:", "MR.", "Patient Name", "Pt:". If name field is blank or says "N/A", mark as "Missing".

2. **Date**: Look for any date field (appointment date, report date, prescription date, visit date, collected date). Valid indicators: "Date:", "Date of Report:", "Date of Visit:", "Appointment:", specific dates like "12/15/2025". If no date is present, mark as "Missing".

3. **Medication**: Look for ANY medication name with dosage information (e.g., "Aspirin 500mg", "Metformin Twice daily"). Do NOT count lab results (CBC, RBC, Blood Pressure, etc.). If no medications are listed with dosages, mark as "Missing".

4. **Physician Signature**: Look for doctor's name/signature line, stamp, digital signature marker, or authentication. Valid indicators: "Dr.", "Signature:", "/s/", signature block, doctor's name with title. If no signature, stamp, or doctor identifier is present, mark as "Missing".

**VALIDATION LOGIC:**
- Mark as "Found" ONLY if the element is clearly and unmistakably present in the document.
- If ANY of these 4 are missing, the document status is "FAILED".
- **CRITICAL RULE**: If the document fails, list **ONLY** the specific items that are missing in failure_reason.

**OUTPUT FORMAT:**
Return STRICT JSON only.

If FAILED:
{
 "document_status": "FAILED",
 "compliance_summary": {
    "patient_name": "Found/Missing",
    "date": "Found/Missing",
    "medication": "Found/Missing",
    "physician_signature": "Found/Missing"
 },
 "failure_reason": "The document failed validation because the following specific item(s) are missing: [List ONLY the missing items]"
}

If VALID:
{
 "document_status": "VALID",
 "patient_data": {
    "patient_name": "...",
    "patient_age": "...",
    "patient_sex": "...",
    "clinical_info": "..."
 },
 "summary_for_human": "...",
 "disease_explanation": "...",
 "medication_info": [],
 "hospital_guide": "..."
}

Return STRICT JSON only.
"""

def image_to_base64(path: str) -> str:
    """Converts image to base64 string for Vision API."""
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode()

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Parses JSON from raw model output using regex to find the JSON block."""
    try:
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)
    except Exception:
        raise ValueError("Invalid JSON from model")

async def classify_document(file_path: str) -> Dict[str, Any]:
    """
    Main handler for analyzing both Image and PDF files.
    Consolidates text extraction and multimodal visual analysis.
    """
    # Step 1: Extract text asynchronously (All pages of PDF or Image)
    ocr_text = extract_text_from_image(file_path)
    
    # Prepare multimodal message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPT + "\n\nEXTRACTED DOCUMENT TEXT:\n" + ocr_text
                }
            ]
        }
    ]

    # Step 2: Add visual context
    if file_path.lower().endswith(".pdf"):
        # NEW LOGIC: Convert PDF pages to images so Vision LLM can scan multi-page visuals
        folder_name = pdf_to_images(file_path)
        image_dir = os.path.join("uploads/images", folder_name)
        
        # Loop through each page image and add it to the message
        for image_name in sorted(os.listdir(image_dir)):
            image_path = os.path.join(image_dir, image_name)
            base64_image = image_to_base64(image_path)
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
    else:
        # Standard logic for single image uploads
        image_base64 = image_to_base64(file_path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        })

    # Step 3: Call AI Model
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing in environment.")

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL_NAME,
            temperature=0.1,
            messages=messages
        )
    except APIStatusError as exc:
        status = getattr(exc, "status_code", "unknown")
        raise RuntimeError(f"Model API error ({status}): {str(exc)}") from exc

    raw_output = response.choices[0].message.content.strip()

    # Step 4: Parse Result
    try:
        return extract_json_from_text(raw_output)
    except Exception:
        return {
            "document_status": "FAILED",
            "failure_reason": "Model output parsing failed. The document structure could not be verified."
        }
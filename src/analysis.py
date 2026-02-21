import os
import base64
import json
import re
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError

from src.textextraction import extract_text_from_image_async
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
- MUST be 7-9 sentences written as a SINGLE continuous paragraph. [cite: 4]
- Do not use bullet points or numbered lists. [cite: 4]
- Recommend specialist type only if names are not present in the document. [cite: 4]

**SECTION 6 - VALIDATION RULES (STRICT AUDIT):**
Analyze the document for these 4 specific elements:
1. Patient Name
2. Date/reported/collected/appointment
3. Medication without Dosages (Note: Lab results like CBC/RBC are NOT medications)
4. Physician Signature/Stamp Signature/Digital Signature/Doctor Siganture

**VALIDATION LOGIC:**
- If ANY of these 4 are missing, the status is "FAILED".
- **CRITICAL RULE**: If the document fails, the `failure_reason` must list **ONLY** the specific items that are missing. Do not mention items that were found.

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
        # Use a more robust regex to find the first '{' and last '}'
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
    # Step 1: Extract text asynchronously (PDF or Image)
    ocr_text = await extract_text_from_image_async(file_path)
    
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

    # Step 2: Add visual context if it is an image
    if not file_path.lower().endswith(".pdf"):
        image_base64 = image_to_base64(file_path)
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_base64}"
            }
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
        if status == 403:
            raise RuntimeError(
                "OpenRouter 403: key/model permission denied. "
                "Check OPENAI_API_KEY and VISION_MODEL_NAME access."
            ) from exc
        raise RuntimeError(f"Model API error ({status}): {str(exc)}") from exc

    raw_output = response.choices[0].message.content.strip()

    # Step 4: Parse and Return Result
    try:
        return extract_json_from_text(raw_output)
    except Exception:
        return {
            "document_status": "FAILED",
            "failure_reason": "Model output parsing failed. The document structure could not be verified."
        }

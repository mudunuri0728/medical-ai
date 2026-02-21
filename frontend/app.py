import streamlit as st
import requests

# 1. Configuration & Global Styling
API_URL = "http://localhost:8000"
st.set_page_config(page_title="MEDICAL_DOC Analyzer", layout="wide", page_icon="🩺")

# Optimized CSS for clean headers and failure summaries
st.markdown("""
    <style>
    .step-header {
        color: #1E3A8A; font-weight: bold; font-size: 1.4rem;
        border-bottom: 2px solid #3B82F6; padding-bottom: 5px;
        margin-top: 30px; margin-bottom: 15px;
    }
    .clinical-text {
        font-size: 1.1rem; line-height: 1.6; color: #374151; padding-left: 10px;
    }
    .failure-summary {
        background-color: #fcfcfc; border: 1px solid #f5c6cb;
        padding: 20px; border-radius: 5px; margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header
st.title("🩺 MEDICAL_DOC Analysis System")
st.caption("Detailed AI-powered Medical Insights & Compliance Audit")

# 3. File Uploader
uploaded_files = st.file_uploader(
    "Upload Medical Reports (PDF, JPEG, PNG)",
    type=["png", "jpg", "jpeg", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

    with st.spinner("🔍 Running Strict Compliance Audit..."):
        try:
            response = requests.post(f"{API_URL}/analyze", files=files)
            results = response.json()
        except Exception as e:
            st.error(f"❌ Connection Failed: {str(e)}")
            st.stop()

    # 4. Process Results
    for item in results:
        st.markdown("---")
        st.header(f"📄 File: {item['file']}")
        
        analysis = item.get("analysis", {})
        
        # --- IF DOCUMENT FAILED: "WHY IT FAILED" SUMMARY ---
        if analysis.get("document_status") == "FAILED":
            st.error("Document Validation Failed")
            
            st.subheader("Why it failed")
            checks = analysis.get("compliance_summary", {})
            
            # Map for human-friendly reasons
            reasons = {
                "patient_name": "Patient name is missing. A valid medical report must clearly identify the patient.",
                "date": "Report date is missing. The document must clearly specify when it was issued.",
                "medication": "No medications identified. This system requires a prescription list for analysis.",
                "physician_signature": "Doctor signature or authentication details are missing."
            }

            # Generate the numbered list exactly as requested
            fail_count = 1
            for key, msg in reasons.items():
                if checks.get(key) == "Missing":
                    st.markdown(f"{fail_count}. {msg}")
                    fail_count += 1

            # Visual Details
            st.write("---")
            st.subheader("Validation Details")
            cols = st.columns(4)
            labels = {"patient_name": "Name", "date": "Date", "medication": "Meds", "physician_signature": "Signature"}
            for i, (key, label) in enumerate(labels.items()):
                status = checks.get(key, "Missing")
                if status == "Found": cols[i].success(f"✅ {label}")
                else: cols[i].error(f"❌ {label}")
            continue 

        # --- IF DOCUMENT PASSED: STEP-BY-STEP MATTER ---
        st.success("✅ Document Verified & Analyzed Successfully")

        # STEP 1: PATIENT SUMMARY
        st.markdown("<div class='step-header'>🧑 STEP 1: PATIENT SUMMARY</div>", unsafe_allow_html=True)
        p_data = analysis.get("patient_data", {})
        st.markdown(f"""
        <div class='clinical-text'>
        <b>Patient:</b> {p_data.get('patient_name')} | <b>Age:</b> {p_data.get('patient_age')} | <b>Sex:</b> {p_data.get('patient_sex')}<br><br>
        {analysis.get('summary_for_human', 'No summary available.')}
        </div>
        """, unsafe_allow_html=True)

        # STEP 2: DIAGNOSIS EXPLANATION
        st.markdown("<div class='step-header'>🩺 STEP 2: DIAGNOSIS EXPLANATION</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='clinical-text'>{analysis.get('disease_explanation', 'No explanation available.')}</div>", unsafe_allow_html=True)

        # STEP 3: MEDICATION INFO
        st.markdown("<div class='step-header'>💊 STEP 3: MEDICATION INFO</div>", unsafe_allow_html=True)
        meds = analysis.get("medication_info", [])
        if meds:
            for med in meds:
                st.markdown(f"<div class='clinical-text'>• {med}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='clinical-text'>No medications found.</div>", unsafe_allow_html=True)

        # STEP 4: HOSPITAL GUIDE & NEXT STEPS
        st.markdown("<div class='step-header'>🏥 STEP 4: HOSPITAL GUIDE & NEXT STEPS</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='clinical-text'>{analysis.get('hospital_guide', 'Next steps not provided.')}</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Powered by MEDICAL_DOC AI Engine | Ver 2.5 (2026)")
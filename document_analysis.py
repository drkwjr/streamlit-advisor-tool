import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
import re

# ---------- Page & Styles ----------
st.set_page_config(page_title="Document Analysis Tool", layout="centered")

st.markdown(
    """
    <style>
      .meta { color:#5f6368; font-size:0.9rem; }
      .chip { display:inline-block; padding:0.25rem 0.6rem; border-radius:999px; font-weight:600; }
      .chip-admin { background:#e6f4ea; color:#0b8043; border:1px solid #c7e7d1; }
      .chip-criminal { background:#fde7e9; color:#c5221f; border:1px solid #f6c7cc; }
      .chip-unknown { background:#e8f0fe; color:#174ea6; border:1px solid #c7d1fb; }
      .card { border:1px solid #e6e6e6; border-radius:12px; padding:16px; background:#ffffff; }
      .codeblock { background:#f6f8fa; border:1px solid #e6e6e6; padding:12px; border-radius:8px; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
      .filename { font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()

# Load OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variable 'OPENAI_API_KEY'.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Helpers ----------
def _clean_possible_json(text: str) -> str:
    """Strip code fences and try to isolate a JSON object."""
    if not text:
        return ""
    cleaned = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip("` \n\r\t")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    return match.group(0) if match else cleaned

def _normalize_type(raw_type: str) -> str:
    if not raw_type:
        return "unknown"
    t = raw_type.strip().lower()
    if "admin" in t:
        return "administrative"
    if "criminal" in t:
        return "criminal"
    return "unknown"

def render_analysis_ui(result_text: str):
    """
    Renders a clean, structured view.
    Tries to parse JSON: {"document_type": "...", "explanation": "..."}
    Falls back to showing raw text in an expander.
    """
    parsed = None
    try:
        possible_json = _clean_possible_json(result_text)
        parsed = json.loads(possible_json)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        doc_type = _normalize_type(
            parsed.get("document_type") or parsed.get("type") or ""
        )
        explanation = parsed.get("explanation") or parsed.get("reason") or ""

        chip_class = {
            "administrative": "chip-admin",
            "criminal": "chip-criminal",
            "unknown": "chip-unknown",
        }.get(doc_type, "chip-unknown")

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div>
              <div class="meta">Result</div>
              <h3 style="margin-top:4px;">Document Classification</h3>
              <div class="chip {chip_class}" title="Predicted type">{doc_type.title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if explanation:
            st.markdown("#### Explanation")
            st.write(explanation)

        with st.expander("Show raw model output"):
            st.markdown(f'<div class="codeblock">{result_text}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="meta">Result</div>
            <h3 style="margin-top:4px;">Analysis</h3>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="codeblock">{result_text}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def analyze_document(file_bytes, mime_type, filename="uploaded_document"):
    """
    Uses the OpenAI API to analyze the uploaded document.
    The prompt now requests a strict JSON response to support clean rendering.
    """
    prompt = (
        """You are an assistant that analyzes documents. Based on the uploaded document, determine whether it is an administrative document or a criminal document.

Return ONLY valid compact JSON with these exact keys (no extra text, no markdown fences):

{
  "document_type": "administrative" | "criminal" | "unknown",
  "explanation": "A short explanation in simple, non-legal language."
}
"""
    )

    try:
        upload_url = "https://api.openai.com/v1/files"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {"file": (filename, io.BytesIO(file_bytes), mime_type)}
        purpose = "vision" if (mime_type and mime_type.startswith("image/")) else "user_data"
        data = {"purpose": purpose}

        upload_resp = requests.post(upload_url, headers=headers, files=files, data=data)
        if upload_resp.status_code not in (200, 201):
            return f"Error uploading file: {upload_resp.status_code} - {upload_resp.text}"

        upload_json = upload_resp.json()
        file_id = upload_json.get("id")
        if not file_id:
            return f"Error uploading file: no file id returned: {upload_json}"

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful legal‐analysis assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "file", "file": {"file_id": file_id}}
                    ]
                }
            ]
            # Note: no temperature param (not needed / supported for this model)
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {e}"

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file:
    - For image files, returns a PIL Image for preview.
    - For PDFs, no preview is shown.
    Returns (file_b_

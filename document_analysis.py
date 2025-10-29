import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
import base64
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
    # Remove backticked fences if present
    cleaned = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip("` \n\r\t")
    # Try to extract the first {...} block if wrapped in prose
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
            parsed.get("document_type") or parsed.get("type") or parsed.get("document") or ""
        )
        explanation = parsed.get("explanation") or parsed.get("reason") or parsed.get("summary") or ""

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
        # Fallback: show the raw text nicely
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
        # Upload to Files API
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

        # Chat completion referencing the uploaded file
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful legal analysis assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "file", "file": {"file_id": file_id}},
                    ],
                },
            ],
            temperature=0,  # encourage deterministic JSON
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error during analysis: {e}"

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file:
    - For image files, returns a PIL Image for preview.
    - For PDFs, no preview is shown.
    Returns (file_bytes, mime_type, preview_image, filename).
    """
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    filename = getattr(uploaded_file, "name", "uploaded_document")

    preview_image = None
    if mime_type != "application/pdf":
        uploaded_file.seek(0)
        try:
            preview_image = Image.open(uploaded_file)
        except Exception:
            st.warning("Could not open the image for preview.")
    return file_bytes, mime_type, preview_image, filename

def main():
    st.title("Document Analysis Tool")
    st.write("Upload a document (image or PDF) and get a quick classification with a plain-English explanation.")

    input_method = st.radio("Choose input method:", ("Upload File", "Capture Image"), horizontal=True)

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"]) if input_method == "Upload File" else st.camera_input("Capture an image")

    if uploaded_file is not None:
        file_bytes, mime_type, preview_image, filename = process_uploaded_file(uploaded_file)

        # Layout: Preview (left) | Details (right)
        col1, col2 = st.columns([3, 2], vertical_alignment="top")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="meta">Preview</div>', unsafe_allow_html=True)
            if mime_type != "application/pdf" and preview_image:
                st.image(preview_image, caption="Uploaded/Captured Image", use_column_width=True)
            else:
                st.write("PDF uploaded. Preview not shown.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="meta">File</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="filename">{filename}</div>', unsafe_allow_html=True)
            st.caption(f"MIME type: {mime_type or 'unknown'} • Size: {len(file_bytes):,} bytes")
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Analyzing the document..."):
            analysis_result = analyze_document(file_bytes, mime_type, filename=filename)

        if isinstance(analysis_result, str) and analysis_result.startswith("Error"):
            st.error(analysis_result)
        else:
            st.success("Analysis Complete")
            render_analysis_ui(analysis_result)

if __name__ == "__main__":
    main()

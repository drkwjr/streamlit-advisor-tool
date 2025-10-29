import streamlit as st
from PIL import Image
import io
import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
import re
import logging
from datetime import datetime

# ================================
# Page & global styles
# ================================
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
      .small { font-size: 0.85rem; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================================
# Logging setup (Streamlit-safe)
# ================================
if "log_records" not in st.session_state:
    st.session_state.log_records = []

class StreamlitHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        st.session_state.log_records.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "message": msg
        })

logger = logging.getLogger("doc_tool")
logger.setLevel(logging.DEBUG)
if not any(isinstance(h, StreamlitHandler) for h in logger.handlers):
    sh = StreamlitHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)

def log_kv(**kwargs):
    logger.debug(json.dumps(kwargs, ensure_ascii=False))

# Debug toggle in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    debug_mode = st.toggle("Show debug logs", value=False)

# ================================
# Env, client
# ================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variable 'OPENAI_API_KEY'.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ================================
# Helpers
# ================================
def _clean_possible_json(text: str) -> str:
    """Strip fences and isolate a JSON object if present."""
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
    Structured view if JSON present:
      {"document_type": "...", "explanation": "..."}
    Otherwise show raw output.
    """
    parsed = None
    try:
        possible_json = _clean_possible_json(result_text)
        parsed = json.loads(possible_json)
    except Exception as e:
        log_kv(event="json_parse_error", error=str(e))
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
    Calls OpenAI to analyze the uploaded document.
    Returns the raw model text (ideally JSON as instructed).
    """
    # IMPORTANT: keep this triple-quoted string closed and simple
    prompt = (
        "You are an assistant that analyzes documents. Based on the uploaded document, "
        "determine whether it is an administrative document or a criminal document.\n\n"
        "Return ONLY valid compact JSON with these exact keys (no extra text, no markdown fences):\n\n"
        "{\n"
        '  "document_type": "administrative" | "criminal" | "unknown",\n'
        '  "explanation": "A short explanation in simple, non-legal language."\n'
        "}\n"
    )

    try:
        # --- Upload to Files API ---
        upload_url = "https://api.openai.com/v1/files"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {"file": (filename, io.BytesIO(file_bytes), mime_type)}
        purpose = "vision" if (mime_type and mime_type.startswith("image/")) else "user_data"
        data = {"purpose": purpose}

        upload_resp = requests.post(upload_url, headers=headers, files=files, data=data)
        log_kv(stage="file_upload_response", status_code=upload_resp.status_code)

        if upload_resp.status_code not in (200, 201):
            # log body snippet to aid debugging
            body = upload_resp.text[:500]
            log_kv(stage="file_upload_error", body_excerpt=body)
            return f"Error uploading file: {upload_resp.status_code} - {upload_resp.text}"

        upload_json = upload_resp.json()
        file_id = upload_json.get("id")
        log_kv(stage="file_upload_success", file_id=file_id)

        if not file_id:
            return f"Error uploading file: no file id returned: {upload_json}"

        # --- Chat Completions call referencing the uploaded file ---
        # gpt-5-nano works with Chat Completions. Do not pass temperature.
        payload = {
            "model": "gpt-5-nano",
            "messages": [
                {"role": "system", "content": "You are a helpful legal analysis assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "file", "file": {"file_id": file_id}},
                    ],
                },
            ],
        }
        log_kv(stage="chat_request", payload_overview={"model": payload["model"], "messages_len": len(payload["messages"])})

        response = client.chat.completions.create(**payload)

        # Capture small excerpt of response for logs
        text = response.choices[0].message.content
        usage = getattr(response, "usage", None)
        log_kv(stage="chat_response", got_text=bool(text), usage=getattr(usage, "__dict__", str(usage)))
        return text

    except Exception as e:
        log_kv(stage="exception", error=str(e))
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
        except Exception as e:
            log_kv(stage="preview_error", error=str(e))
            st.warning("Could not open the image for preview.")
    return file_bytes, mime_type, preview_image, filename

def render_debug():
    st.subheader("Debug logs")
    if not st.session_state.log_records:
        st.caption("No logs yet.")
        return
    for rec in st.session_state.log_records[-300:]:
        st.write(f"[{rec['time']}] {rec['level']}: {rec['message']}")

# ================================
# Main
# ================================
def main():
    st.title("Document Analysis Tool")
    st.write("Upload a document (image or PDF) for quick classification with a plain-English explanation.")

    input_method = st.radio("Choose input method:", ("Upload File", "Capture Image"), horizontal=True)

    uploaded_file = (
        st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "pdf"])
        if input_method == "Upload File" else
        st.camera_input("Capture an image")
    )

    if uploaded_file is not None:
        file_bytes, mime_type, preview_image, filename = process_uploaded_file(uploaded_file)

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

    if debug_mode:
        st.divider()
        render_debug()
        st.caption("These logs include file upload responses, request payload overviews, and response usage summaries.")

if __name__ == "__main__":
    main()

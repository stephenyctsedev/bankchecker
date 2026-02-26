import json
import os
import re
import shutil
import traceback
from glob import glob

import pandas as pd
import platformdirs
import requests
import streamlit as st

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None

SURYA_IMPORT_ERROR = None
try:
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
except Exception as e:
    SURYA_IMPORT_ERROR = str(e)
    FoundationPredictor = None
    DetectionPredictor = None
    RecognitionPredictor = None

st.set_page_config(page_title="Bank Statement Interest Checker", page_icon="PDF", layout="wide")

KEYWORDS = ["Interest", "INT", "CR", "Interest Credit", "Interest Paid", "Interest Income", "li xi"]


@st.cache_resource(show_spinner="Loading Surya OCR models...")
def load_surya_models():
    if SURYA_IMPORT_ERROR:
        st.error(f"Surya OCR import failed: {SURYA_IMPORT_ERROR}")
        if "torchvision::nms" in SURYA_IMPORT_ERROR or "PreTrainedModel" in SURYA_IMPORT_ERROR:
            st.code(
                "pip uninstall -y torchvision\n"
                "pip install --upgrade torch transformers surya-ocr",
                language="bash",
            )
            st.info(
                "Detected a torch/torchvision mismatch. "
                "torchvision is optional for this app and can be removed."
            )
        else:
            st.info("Install Surya with: pip install surya-ocr")
        return None, None, None

    try:
        foundation = FoundationPredictor()
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor(foundation)
        return foundation, det_predictor, rec_predictor
    except Exception as e:
        err_text = str(e)
        if isinstance(e, OSError) and getattr(e, "errno", None) == 22:
            cache_root = platformdirs.user_cache_dir("datalab")
            models_cache = os.path.join(cache_root, "datalab", "Cache", "models")
            st.warning(
                f"Surya model cache may be corrupted on Windows (Errno 22). "
                f"Clearing cache and retrying once: {models_cache}"
            )
            try:
                if os.path.exists(models_cache):
                    shutil.rmtree(models_cache, ignore_errors=True)

                foundation = FoundationPredictor()
                det_predictor = DetectionPredictor()
                rec_predictor = RecognitionPredictor(foundation)
                st.success("Surya OCR models initialized after cache reset.")
                return foundation, det_predictor, rec_predictor
            except Exception as retry_err:
                st.error(f"Retry after cache reset failed: {retry_err}")
                st.code(traceback.format_exc(), language="text")
                return None, None, None

        st.error(f"Failed to initialize Surya OCR models: {err_text}")
        st.code(traceback.format_exc(), language="text")
        return None, None, None


def get_ollama_models(base_url: str):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return models or ["llama3:8b", "llama3.2"]
    except Exception:
        return ["Could not fetch models - check Ollama URL"]


def select_folder():
    if tk is None or filedialog is None:
        st.error("tkinter is not available in this environment.")
        return ""

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_selected = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_selected


def _pdf_signature(pdf_path: str) -> str:
    stat = os.stat(pdf_path)
    return f"{stat.st_size}-{stat.st_mtime_ns}"


def ocr_pdf_to_lines(pdf_path, models, poppler_path=None, ocr_dpi=150, max_pages=0, math_mode=False):
    foundation, det_predictor, rec_predictor = models
    if not all([foundation, det_predictor, rec_predictor]):
        raise RuntimeError("Surya OCR models are not ready.")

    pages = []
    if convert_from_path is not None:
        try:
            pages = convert_from_path(pdf_path, dpi=ocr_dpi, poppler_path=poppler_path or None)
        except Exception as e:
            # Fall back to pypdfium2 if Poppler is not installed or not in PATH.
            if pdfium is None:
                st.error(
                    "Failed to convert PDF pages. Poppler was not found and pypdfium2 is unavailable. "
                    f"Details: {e}"
                )
                return []
    if not pages:
        if pdfium is None:
            st.error(
                "PDF conversion failed. Install Poppler (pdftoppm) or install pypdfium2."
            )
            return []
        try:
            doc = pdfium.PdfDocument(pdf_path)
            scale = ocr_dpi / 72
            page_count = len(doc)
            if max_pages and max_pages > 0:
                page_count = min(page_count, int(max_pages))
            for i in range(page_count):
                page = doc[i]
                pil_img = page.render(scale=scale).to_pil()
                pages.append(pil_img)
                page.close()
            doc.close()
        except Exception as e:
            st.error(
                "Failed to convert PDF pages with both pdf2image and pypdfium2. "
                f"Details: {e}"
            )
            return []

    if not pages:
        return []
    if max_pages and max_pages > 0:
        pages = pages[: int(max_pages)]

    predictions = rec_predictor(
        pages,
        det_predictor=det_predictor,
        sort_lines=False,
        math_mode=math_mode,
    )

    lines = []
    for page_result in predictions:
        for line_obj in page_result.text_lines:
            text = getattr(line_obj, "text", "")
            if text:
                lines.append(text)
    return lines


@st.cache_data(show_spinner=False)
def ocr_pdf_to_lines_cached(pdf_path, pdf_signature, poppler_path=None, ocr_dpi=150, max_pages=0, math_mode=False):
    _ = pdf_signature
    models = load_surya_models()
    if not all(models):
        return []
    return ocr_pdf_to_lines(
        pdf_path,
        models,
        poppler_path=poppler_path,
        ocr_dpi=ocr_dpi,
        max_pages=max_pages,
        math_mode=math_mode,
    )


def filter_interest_context(lines):
    if not lines:
        return ""

    matched_indices = set()
    for i, line in enumerate(lines):
        lower_line = line.lower()
        for kw in KEYWORDS:
            if kw.lower() in lower_line:
                for j in range(max(0, i - 2), min(len(lines), i + 3)):
                    matched_indices.add(j)
                break

    if not matched_indices:
        return ""

    ordered_indices = sorted(matched_indices)
    filtered_lines = [lines[i] for i in ordered_indices]
    return "\n".join(filtered_lines)


st.title("Bank Statement Interest Checker")
st.markdown("Use pdf2image + Surya OCR to read PDFs and Ollama to extract interest credits.")

with st.sidebar:
    st.header("Settings")
    ollama_ip = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")

    model_list = get_ollama_models(ollama_ip)
    selected_model = st.selectbox("LLM Model", model_list)

    st.divider()
    poppler_path = st.text_input(
        "Poppler Path",
        help=r"Path to Poppler 'bin' directory, e.g., C:\path\to\poppler-xx\bin",
    )
    st.session_state["poppler_path"] = poppler_path
    st.divider()
    st.subheader("OCR Performance")
    ocr_dpi = st.slider("OCR DPI", min_value=100, max_value=250, value=150, step=10)
    max_pages = st.number_input(
        "Max pages per PDF (0 = all pages)",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
    )
    math_mode = st.checkbox("Enable math mode (slower)", value=False)
    use_ocr_cache = st.checkbox("Cache OCR result per PDF", value=True)

    if st.button("Select PDF Folder"):
        folder_path = select_folder()
        if folder_path:
            st.session_state["folder_path"] = folder_path

current_folder = st.session_state.get("folder_path", "No folder selected")
st.info(f"Current folder: `{current_folder}`")

if st.button("Run Extraction", type="primary"):
    if not os.path.exists(current_folder) or current_folder == "No folder selected":
        st.error("Please select a valid folder first.")
    else:
        pdf_files = glob(os.path.join(current_folder, "*.pdf"))
        if not pdf_files:
            st.warning("No PDF files found in selected folder.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_results = []

            models = load_surya_models()
            if not all(models):
                st.error("OCR model load failed. Please fix dependencies and retry.")
            else:
                for idx, pdf in enumerate(pdf_files):
                    filename = os.path.basename(pdf)
                    status_text.text(f"Processing ({idx + 1}/{len(pdf_files)}): {filename}")

                    try:
                        if use_ocr_cache:
                            lines = ocr_pdf_to_lines_cached(
                                pdf,
                                _pdf_signature(pdf),
                                poppler_path=st.session_state.get("poppler_path"),
                                ocr_dpi=ocr_dpi,
                                max_pages=max_pages,
                                math_mode=math_mode,
                            )
                        else:
                            lines = ocr_pdf_to_lines(
                                pdf,
                                models,
                                poppler_path=st.session_state.get("poppler_path"),
                                ocr_dpi=ocr_dpi,
                                max_pages=max_pages,
                                math_mode=math_mode,
                            )

                        filtered_context = filter_interest_context(lines)

                        if not filtered_context.strip():
                            progress_bar.progress((idx + 1) / len(pdf_files))
                            continue

                        prompt = f"""
You are a professional bank auditor. Extract all "Interest Credit" entries from this statement.

The text has already been pre-filtered to lines that likely contain interest-related information.

### TASK:
From the text below, extract every record where bank interest is credited to the account.

### FIELDS:
- date: Transaction date in YYYY-MM-DD format (normalize if needed)
- description: Short description of the transaction
- amount: Interest amount as a number (do NOT use balance amounts)

### RULES:
1. Only include rows that correspond to interest credit / interest income.
2. Ignore running balances, totals, and non-interest fees.
3. If a row has multiple numbers, choose the one that is clearly the interest amount.
4. If you are unsure, skip that row instead of guessing.

### OUTPUT FORMAT (STRICT):
Return ONLY a JSON array. No explanation, no text around it.

Example:
[
  {{"date": "2024-04-30", "description": "INTEREST PAID", "amount": 16.49}}
]

### FILTERED CONTEXT TO ANALYZE:
{filtered_context}
""".strip()

                        payload = {
                            "model": selected_model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                            "options": {
                                "temperature": 0,
                                "num_predict": 1000,
                                "top_k": 20,
                                "top_p": 0.9,
                            },
                        }

                        res = requests.post(f"{ollama_ip}/api/generate", json=payload, timeout=120)
                        res.raise_for_status()
                        response_data = res.json().get("response", "[]").strip()

                        json_match = re.search(r"\[.*\]", response_data, re.DOTALL)
                        clean_json = json_match.group(0) if json_match else response_data
                        items = json.loads(clean_json or "[]")

                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    item["source"] = filename
                                    all_results.append(item)
                        elif isinstance(items, dict):
                            items["source"] = filename
                            all_results.append(items)

                    except Exception as e:
                        st.error(f"Error in {filename}: {e}")

                    progress_bar.progress((idx + 1) / len(pdf_files))

            if all_results:
                df = pd.DataFrame(all_results)

                rename_map = {
                    "amount_value": "amount",
                    "Value": "amount",
                    "price": "amount",
                    "date_value": "date",
                    "desc": "description",
                }
                df.rename(columns=rename_map, inplace=True)

                if "amount" in df.columns:
                    df["amount"] = (
                        df["amount"].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
                    )
                    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
                else:
                    df["amount"] = 0.0

                st.success("Extraction completed.")
                st.subheader("Interest Records")
                st.dataframe(df, use_container_width=True)

                total = df["amount"].sum()
                st.metric("Total Interest", f"${total:,.2f}")

                csv_data = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="interest_summary.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No interest records found.")

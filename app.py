import json
import os
import re
import shutil
import traceback
import io
import base64
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
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
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import torch
except Exception:
    torch = None

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


def detect_compute():
    if torch is None:
        return "cpu", 0
    try:
        if torch.cuda.is_available():
            return "cuda", 1
    except Exception:
        pass
    return "cpu", 0


@st.cache_resource(show_spinner="Loading Surya OCR models...")
def load_surya_models(device="cpu"):
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
        foundation = FoundationPredictor(device=device)
        det_predictor = DetectionPredictor(device=device)
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

                foundation = FoundationPredictor(device=device)
                det_predictor = DetectionPredictor(device=device)
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


def _is_substantial_text(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip()
    return len(clean) > 100


def _is_gibberish_text(text: str) -> bool:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return True
    alnum_ratio = sum(c.isalnum() for c in clean) / max(1, len(clean))
    if alnum_ratio < 0.40:
        return True
    words = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", clean)
    return len(words) < 6


def _render_single_page_image(pdf_path, page_index, dpi=150, poppler_path=None):
    page_no = int(page_index) + 1
    if convert_from_path is not None:
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path or None,
                first_page=page_no,
                last_page=page_no,
            )
            if pages:
                return pages[0]
        except Exception:
            pass
    if pdfium is None:
        raise RuntimeError("No PDF renderer available for OCR fallback.")
    doc = pdfium.PdfDocument(pdf_path)
    try:
        page = doc[int(page_index)]
        img = page.render(scale=dpi / 72).to_pil()
        page.close()
        return img
    finally:
        doc.close()


def _ocr_image_to_lines(image, models, math_mode=False):
    foundation, det_predictor, rec_predictor = models
    if not all([foundation, det_predictor, rec_predictor]):
        return []
    predictions = rec_predictor(
        [image],
        det_predictor=det_predictor,
        sort_lines=False,
        math_mode=math_mode,
    )
    out = []
    if predictions:
        for line_obj in predictions[0].text_lines:
            text = getattr(line_obj, "text", "")
            if text:
                out.append(text)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def extract_pdf_lines_hybrid(pdf_path, models, poppler_path=None, ocr_dpi=150, max_pages=0, math_mode=False, max_workers=4):
    if fitz is not None:
        with fitz.open(pdf_path) as doc:
            page_count = doc.page_count
    elif pdfium is not None:
        doc = pdfium.PdfDocument(pdf_path)
        page_count = len(doc)
        doc.close()
    else:
        page_count = 0

    if max_pages and max_pages > 0:
        page_count = min(page_count, int(max_pages))
    if page_count <= 0:
        return []

    def _process_text_page(page_idx):
        page_text = ""
        if fitz is not None:
            with fitz.open(pdf_path) as fdoc:
                page_text = fdoc.load_page(page_idx).get_text("text") or ""
        if _is_substantial_text(page_text) and not _is_gibberish_text(page_text):
            extracted_lines = [ln.strip() for ln in page_text.splitlines() if ln and ln.strip()]
            return page_idx, extracted_lines, False
        return page_idx, [], True

    results = {}
    ocr_needed_pages = []
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_text_page, i) for i in range(page_count)]
        for fut in as_completed(futures):
            page_idx, page_lines, needs_ocr = fut.result()
            results[page_idx] = page_lines
            if needs_ocr:
                ocr_needed_pages.append(page_idx)

    # Surya predictors are not reliably thread-safe. Run OCR fallback sequentially.
    for page_idx in sorted(ocr_needed_pages):
        img = _render_single_page_image(
            pdf_path,
            page_idx,
            dpi=ocr_dpi,
            poppler_path=poppler_path,
        )
        results[page_idx] = _ocr_image_to_lines(img, models, math_mode=math_mode)

    lines = []
    for i in range(page_count):
        lines.extend(results.get(i, []))
    return lines


def unload_surya_models(models):
    try:
        load_surya_models.clear()
    except Exception:
        pass
    try:
        foundation, det_predictor, rec_predictor = models
        del foundation
        del det_predictor
        del rec_predictor
    except Exception:
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_pdf_to_images(pdf_path, poppler_path=None, dpi=150, max_pages=0):
    pages = []
    if convert_from_path is not None:
        try:
            last_page = int(max_pages) if max_pages and max_pages > 0 else None
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path or None,
                first_page=1,
                last_page=last_page,
            )
        except Exception as e:
            # Fall back to pypdfium2 if Poppler is not installed or not in PATH.
            if pdfium is None:
                raise RuntimeError(
                    "Failed to convert PDF pages. Poppler was not found and pypdfium2 is unavailable. "
                    f"Details: {e}"
                )
    if not pages:
        if pdfium is None:
            raise RuntimeError(
                "PDF conversion failed. Install Poppler (pdftoppm) or install pypdfium2."
            )
        try:
            doc = pdfium.PdfDocument(pdf_path)
            scale = dpi / 72
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
            raise RuntimeError(
                "Failed to convert PDF pages with both pdf2image and pypdfium2. "
                f"Details: {e}"
            )

    if not pages:
        return []
    if max_pages and max_pages > 0:
        pages = pages[: int(max_pages)]
    return pages


def ocr_pdf_to_lines(pdf_path, models, poppler_path=None, ocr_dpi=150, max_pages=0, math_mode=False):
    foundation, det_predictor, rec_predictor = models
    if not all([foundation, det_predictor, rec_predictor]):
        raise RuntimeError("Surya OCR models are not ready.")

    pages = render_pdf_to_images(pdf_path, poppler_path=poppler_path, dpi=ocr_dpi, max_pages=max_pages)

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
def ocr_pdf_to_lines_cached(
    pdf_path,
    pdf_signature,
    poppler_path=None,
    ocr_dpi=150,
    max_pages=0,
    math_mode=False,
    max_workers=4,
):
    _ = pdf_signature
    device, _ollama_num_gpu = detect_compute()
    models = load_surya_models(device=device)
    if not all(models):
        return []
    return extract_pdf_lines_hybrid(
        pdf_path,
        models,
        poppler_path=poppler_path,
        ocr_dpi=ocr_dpi,
        max_pages=max_pages,
        math_mode=math_mode,
        max_workers=max_workers,
    )


def pil_to_base64_jpeg(image, quality=75, max_side=1600):
    w, h = image.size
    longest = max(w, h)
    if longest > max_side and max_side > 0:
        scale = max_side / float(longest)
        image = image.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_json_array(text):
    text = (text or "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = match.group(0) if match else text
    data = json.loads(candidate or "[]")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def extract_interest_from_vision_page(
    ollama_ip,
    selected_model,
    page_img,
    request_timeout=300,
    max_retries=2,
    jpeg_quality=70,
    image_max_side=1600,
    ollama_num_gpu=0,
):
    image_b64 = pil_to_base64_jpeg(page_img, quality=jpeg_quality, max_side=image_max_side)
    prompt = """
You are a bank statement auditor.
Analyze this statement page image and extract ONLY interest-credit transactions.

Rules:
1. Include only entries clearly related to interest paid/credited/income.
2. Ignore balances, totals, and non-interest rows.
3. If uncertain, skip.
4. Date format must be YYYY-MM-DD when possible.
5. amount must be numeric, no currency symbol.

Return ONLY a JSON array:
[
  {"date":"2024-04-30","description":"INTEREST PAID","amount":16.49}
]

If no interest row exists, return [].
""".strip()
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 1000},
    }
    payload["options"]["num_gpu"] = int(ollama_num_gpu)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            res = requests.post(f"{ollama_ip}/api/generate", json=payload, timeout=request_timeout)
            res.raise_for_status()
            response_data = res.json().get("response", "[]")
            return parse_json_array(response_data)
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Ollama read timeout after {request_timeout}s (retries={max_retries}). "
                "Try lower DPI/max pages, smaller image size, or higher timeout."
            ) from e
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(f"Vision extraction failed: {last_err}")


def filter_interest_context(lines, token_limit=1000):
    if not lines:
        return "", False

    matched_indices = set()
    for i, line in enumerate(lines):
        lower_line = line.lower()
        if "interest" in lower_line:
            for j in range(max(0, i - 2), min(len(lines), i + 3)):
                matched_indices.add(j)

    if not matched_indices:
        return "", False

    ordered_indices = sorted(matched_indices)
    filtered_lines = [lines[i] for i in ordered_indices]
    out_lines = []
    used_tokens = 0
    truncated = False
    for line in filtered_lines:
        line_tokens = len(re.findall(r"\S+", line))
        if used_tokens + line_tokens > int(token_limit):
            truncated = True
            break
        out_lines.append(line)
        used_tokens += line_tokens
    return "\n".join(out_lines), truncated


def get_context_snippets(text):
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    keywords = ("利息", "interest", "int", "cr")
    picked = set()
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(k in lower for k in keywords):
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                picked.add(j)
    if not picked:
        return []
    return [lines[i] for i in sorted(picked)]


def truncate_text_to_tokens(text, token_limit=1000):
    lines = (text or "").splitlines()
    out = []
    used = 0
    truncated = False
    for line in lines:
        t = len(re.findall(r"\S+", line))
        if used + t > int(token_limit):
            truncated = True
            break
        out.append(line)
        used += t
    return "\n".join(out), truncated


st.title("Bank Statement Interest Checker")
st.markdown("Use pdf2image + Surya OCR to read PDFs and Ollama to extract interest credits.")

with st.sidebar:
    st.header("Settings")
    ollama_ip = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")

    model_list = get_ollama_models(ollama_ip)
    default_model_name = "llama3.2:3b"
    default_model_index = model_list.index(default_model_name) if default_model_name in model_list else 0
    selected_model = st.selectbox("LLM Model", model_list, index=default_model_index)
    compute_device, ollama_num_gpu = detect_compute()
    st.caption(f"Compute: `{compute_device}` | Ollama num_gpu: `{ollama_num_gpu}`")
    extraction_mode = st.radio(
        "Extraction Mode",
        ["Surya OCR + Text LLM", "Vision Model (PDF images)"],
    )

    st.divider()
    if st.button("Select PDF Folder"):
        folder_path = select_folder()
        if folder_path:
            st.session_state["folder_path"] = folder_path

    st.divider()
    poppler_path = st.text_input(
        "Poppler Path",
        help=r"Path to Poppler 'bin' directory, e.g., C:\path\to\poppler-xx\bin",
    )
    st.session_state["poppler_path"] = poppler_path
    st.divider()
    st.subheader("OCR Performance")
    ocr_dpi = 150
    st.caption("OCR DPI is fixed to 150 for lower VRAM usage.")
    max_pages = st.number_input(
        "Max pages per PDF (0 = all pages)",
        min_value=0,
        max_value=500,
        value=5,
        step=1,
    )
    math_mode = st.checkbox("Enable math mode (slower)", value=False)
    use_ocr_cache = st.checkbox("Cache OCR result per PDF", value=True)
    page_workers = st.slider("Page parallel workers", min_value=1, max_value=8, value=4, step=1)
    st.divider()
    st.subheader("Vision Settings")
    vision_timeout = st.number_input(
        "Vision timeout per page (seconds)",
        min_value=60,
        max_value=1200,
        value=300,
        step=30,
    )
    vision_retries = st.number_input(
        "Vision retries per page",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )
    vision_image_max_side = st.slider(
        "Vision image max side (px)",
        min_value=900,
        max_value=2500,
        value=1600,
        step=100,
    )
    vision_jpeg_quality = st.slider(
        "Vision JPEG quality",
        min_value=50,
        max_value=95,
        value=70,
        step=5,
    )

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
            models = (None, None, None)
            if extraction_mode == "Surya OCR + Text LLM":
                models = load_surya_models(device=compute_device)
                if not all(models):
                    st.error("OCR model load failed. Please fix dependencies and retry.")
                    st.stop()

            if extraction_mode == "Vision Model (PDF images)" and "vision" not in selected_model.lower():
                st.warning("Selected model may not support images. Recommended: llama3.2-vision:11b")

            all_snippets = []
            for idx, pdf in enumerate(pdf_files):
                filename = os.path.basename(pdf)
                status_text.text(f"Processing ({idx + 1}/{len(pdf_files)}): {filename}")

                try:
                    if extraction_mode == "Surya OCR + Text LLM":
                        if use_ocr_cache:
                            lines = ocr_pdf_to_lines_cached(
                                pdf,
                                _pdf_signature(pdf),
                                poppler_path=st.session_state.get("poppler_path"),
                                ocr_dpi=ocr_dpi,
                                max_pages=max_pages,
                                math_mode=math_mode,
                                max_workers=page_workers,
                            )
                        else:
                            lines = extract_pdf_lines_hybrid(
                                pdf,
                                models,
                                poppler_path=st.session_state.get("poppler_path"),
                                ocr_dpi=ocr_dpi,
                                max_pages=max_pages,
                                math_mode=math_mode,
                                max_workers=page_workers,
                            )

                        page_text = "\n".join(lines)
                        snippets = get_context_snippets(page_text)
                        if not snippets:
                            progress_bar.progress((idx + 1) / len(pdf_files))
                            continue
                        all_snippets.append(f"[FILE: {filename}]\n" + "\n".join(snippets))
                    else:
                        pages = render_pdf_to_images(
                            pdf,
                            poppler_path=st.session_state.get("poppler_path"),
                            dpi=ocr_dpi,
                            max_pages=max_pages,
                        )
                        for page_idx, page_img in enumerate(pages, start=1):
                            try:
                                page_items = extract_interest_from_vision_page(
                                    ollama_ip=ollama_ip,
                                    selected_model=selected_model,
                                    page_img=page_img,
                                    request_timeout=int(vision_timeout),
                                    max_retries=int(vision_retries),
                                    jpeg_quality=int(vision_jpeg_quality),
                                    image_max_side=int(vision_image_max_side),
                                    ollama_num_gpu=int(ollama_num_gpu),
                                )
                                for item in page_items:
                                    if isinstance(item, dict):
                                        item["source"] = filename
                                        item["page"] = page_idx
                                        all_results.append(item)
                            except Exception as page_err:
                                st.warning(f"{filename} page {page_idx}: {page_err}")

                except Exception as e:
                    st.error(f"Error in {filename}: {e}")

                progress_bar.progress((idx + 1) / len(pdf_files))

            if extraction_mode == "Surya OCR + Text LLM":
                unload_surya_models(models)
                if all_snippets:
                    combined_context = "\n\n".join(all_snippets)
                    combined_context, was_truncated = truncate_text_to_tokens(combined_context, token_limit=1000)
                    if was_truncated:
                        st.info("Combined context exceeded 1000 tokens and was truncated.")
                    prompt = f"""
You are a professional bank auditor. Extract all interest-credit transactions from the combined context.

Return ONLY a JSON array.
Each record must include:
- source: PDF filename from the [FILE: ...] header
- date: YYYY-MM-DD
- description
- amount: numeric

Ignore non-interest rows, balances, and totals.
If unsure, skip.

Context:
{combined_context}
""".strip()
                    payload = {
                        "model": selected_model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": 0,
                            "num_predict": 1200,
                            "num_gpu": int(ollama_num_gpu),
                        },
                    }
                    try:
                        res = requests.post(f"{ollama_ip}/api/generate", json=payload, timeout=180)
                        res.raise_for_status()
                        response_data = res.json().get("response", "[]").strip()
                        items = parse_json_array(response_data)
                        for item in items:
                            if isinstance(item, dict):
                                all_results.append(item)
                    except Exception as llm_err:
                        st.error(f"LLM extraction failed: {llm_err}")

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

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
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from paddleocr import PaddleOCR as _PaddleOCR
except Exception:
    _PaddleOCR = None
try:
    import torch
except Exception:
    torch = None

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


def get_ollama_models(base_url: str):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return models or ["llama3:8b", "llama3.2"]
    except Exception:
        return ["Could not fetch models - check Ollama URL"]


def select_pdf_files():
    if tk is None or filedialog is None:
        st.error("tkinter is not available in this environment.")
        return []

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    files_selected = filedialog.askopenfilenames(
        master=root,
        title="Select PDF files",
        filetypes=[("PDF files", "*.pdf")],
    )
    root.destroy()
    return list(files_selected)


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


def _paddle_ocr_image_to_lines(image, paddle_ocr):
    """Run PaddleOCR on a PIL image; returns list of text lines or None on failure."""
    if paddle_ocr is None:
        return None
    try:
        import numpy as np
        result = paddle_ocr.ocr(np.array(image), cls=True)
        if not result or not result[0]:
            return []
        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else ""
                if text:
                    lines.append(str(text))
        return lines
    except Exception as e:
        print(f"[OCR] PaddleOCR inference error: {e}")
        return None  # signal failure → caller falls back to Surya OCR


def extract_pdf_lines_hybrid(pdf_path, poppler_path=None, ocr_dpi=150, max_pages=0, max_workers=4, force_ocr=False, paddle_ocr=None):
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
        if force_ocr:
            print(f"[OCR] Page {page_idx + 1}: force_ocr=True → queued for OCR")
            return page_idx, [], True
        page_text = ""
        # pdfplumber preserves table/column row order better than PyMuPDF
        if pdfplumber is not None:
            try:
                with pdfplumber.open(pdf_path) as plumb_doc:
                    page_text = plumb_doc.pages[page_idx].extract_text() or ""
            except Exception:
                page_text = ""
        # fall back to PyMuPDF if pdfplumber unavailable or returned nothing
        if not page_text and fitz is not None:
            with fitz.open(pdf_path) as fdoc:
                page_text = fdoc.load_page(page_idx).get_text("text") or ""
        if _is_substantial_text(page_text) and not _is_gibberish_text(page_text):
            engine = "pdfplumber" if pdfplumber is not None else "PyMuPDF"
            print(f"[OCR] Page {page_idx + 1}: text extracted via {engine} ({len(page_text)} chars)")
            extracted_lines = [ln.strip() for ln in page_text.splitlines() if ln and ln.strip()]
            return page_idx, extracted_lines, False
        print(f"[OCR] Page {page_idx + 1}: text extraction insufficient → queued for OCR")
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

    # Run OCR fallback sequentially using PaddleOCR.
    for page_idx in sorted(ocr_needed_pages):
        img = _render_single_page_image(
            pdf_path,
            page_idx,
            dpi=ocr_dpi,
            poppler_path=poppler_path,
        )
        paddle_lines = _paddle_ocr_image_to_lines(img, paddle_ocr)
        if paddle_lines is not None:
            joined = "\n".join(paddle_lines)
            if _is_substantial_text(joined) and not _is_gibberish_text(joined):
                print(f"[OCR] Page {page_idx + 1}: used PaddleOCR ({len(paddle_lines)} lines)")
                results[page_idx] = paddle_lines
                continue
            else:
                print(f"[OCR] Page {page_idx + 1}: PaddleOCR output insufficient, page skipped")
        else:
            print(f"[OCR] Page {page_idx + 1}: PaddleOCR unavailable/failed, page skipped")
        results[page_idx] = []

    lines = []
    for i in range(page_count):
        lines.extend(results.get(i, []))
    return lines


@st.cache_resource(show_spinner=False)
def load_paddle_ocr(use_gpu=False):
    if _PaddleOCR is None:
        return None
    gpu_variants = (
        [{"device": "gpu"}, {"use_gpu": True}] if use_gpu else []
    ) + [{}]
    # Try progressively simpler configs across PaddleOCR 3.x and 2.x APIs
    base_options = [
        {"use_textline_orientation": True, "lang": "en"},  # PaddleOCR 3.x
        {"use_angle_cls": True, "lang": "en"},             # PaddleOCR 2.x
        {"lang": "en"},                                    # minimal fallback
    ]
    last_err = None
    for base in base_options:
        for extra in gpu_variants:
            try:
                ocr = _PaddleOCR(**base, **extra)
                print(f"[OCR] PaddleOCR loaded — base={base} gpu={extra or 'auto'}")
                return ocr
            except TypeError:
                continue
            except Exception as e:
                last_err = e
                break  # non-TypeError: try simpler base, not another GPU variant
    print(f"[OCR] PaddleOCR failed to load: {last_err}")
    return None


@st.cache_data(show_spinner=False)
def ocr_pdf_to_lines_cached(
    pdf_path,
    pdf_signature,
    poppler_path=None,
    ocr_dpi=150,
    max_pages=0,
    max_workers=4,
    force_ocr=False,
):
    _ = pdf_signature
    device, _ = detect_compute()
    paddle_ocr = load_paddle_ocr(use_gpu=(device == "cuda"))
    return extract_pdf_lines_hybrid(
        pdf_path,
        poppler_path=poppler_path,
        ocr_dpi=ocr_dpi,
        max_pages=max_pages,
        max_workers=max_workers,
        force_ocr=force_ocr,
        paddle_ocr=paddle_ocr,
    )


def parse_json_array(text):
    raw = (text or "").strip()
    if not raw:
        return []

    # Remove markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()

    candidates = [cleaned]
    arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if arr_match:
        candidates.append(arr_match.group(0))
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    def _extract_balanced(s, open_ch, close_ch):
        start = s.find(open_ch)
        if start < 0:
            return ""
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return ""

    bal_arr = _extract_balanced(cleaned, "[", "]")
    if bal_arr:
        candidates.append(bal_arr)
    bal_obj = _extract_balanced(cleaned, "{", "}")
    if bal_obj:
        candidates.append(bal_obj)

    for c in candidates:
        c = c.strip()
        if not c:
            continue
        try:
            data = json.loads(c)
        except Exception:
            # common cleanup: trailing commas before ] or }
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", c))
            except Exception:
                continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Common model wrappers.
            for key in ("records", "items", "data", "results", "transactions"):
                wrapped = data.get(key)
                if isinstance(wrapped, list):
                    return wrapped
            return [data]
    return []


def repair_json_with_ollama(ollama_ip, model, bad_output, timeout_sec=120, num_gpu=0):
    repair_prompt = f"""
Convert the following content into a STRICT JSON array only.

Rules:
1. Output must be valid JSON.
2. Output must be an array, e.g. [{{...}}, {{...}}]
3. No markdown, no explanation, no extra text.
4. If content has no valid transaction records, output [].

Content:
{bad_output}
""".strip()
    payload = {
        "model": model,
        "prompt": repair_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 1200,
            "num_gpu": int(num_gpu),
        },
    }
    res = requests.post(f"{ollama_ip}/api/generate", json=payload, timeout=int(timeout_sec))
    res.raise_for_status()
    try:
        repaired_text = res.json().get("response", "[]")
    except Exception:
        repaired_text = res.text
    return parse_json_array((repaired_text or "").strip())


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


def call_ollama_json_with_retry(ollama_ip, payload, timeout_sec=180, retries=2):
    last_err = None
    for attempt in range(int(retries) + 1):
        try:
            res = requests.post(f"{ollama_ip}/api/generate", json=payload, timeout=int(timeout_sec))
            res.raise_for_status()
            try:
                response_text = res.json().get("response", "[]")
            except Exception:
                # fallback when server returns a malformed wrapper JSON
                response_text = res.text
            parsed = parse_json_array((response_text or "").strip())
            if parsed:
                return parsed
            # Automatic repair pass: ask model to convert output to strict JSON.
            repaired = repair_json_with_ollama(
                ollama_ip=ollama_ip,
                model=payload.get("model", "llama3.2:3b"),
                bad_output=(response_text or "").strip(),
                timeout_sec=min(int(timeout_sec), 180),
                num_gpu=payload.get("options", {}).get("num_gpu", 0),
            )
            if repaired:
                return repaired
            raise ValueError("Unable to parse valid JSON array from model output.")
        except requests.exceptions.ReadTimeout as e:
            last_err = e
            if attempt < int(retries):
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(
                f"Ollama read timeout after {timeout_sec}s (retries={retries}). "
                "Increase timeout, reduce max pages, or reduce context token limit."
            ) from e
        except Exception as e:
            last_err = e
            if attempt < int(retries):
                time.sleep(1.0 * (attempt + 1))
                continue
            break
    raise RuntimeError(f"Ollama call failed: {last_err}")


st.title("Bank Statement Interest Checker")
st.markdown("Use pdfplumber + PaddleOCR to read PDFs and Ollama to extract interest credits.")

with st.sidebar:
    st.header("Settings")
    ollama_ip = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")

    model_list = get_ollama_models(ollama_ip)
    default_model_name = "llama3.2:3b"
    default_model_index = model_list.index(default_model_name) if default_model_name in model_list else 0
    selected_model = st.selectbox("LLM Model", model_list, index=default_model_index)
    compute_device, ollama_num_gpu = detect_compute()
    st.caption(f"Compute: `{compute_device}` | Ollama num_gpu: `{ollama_num_gpu}`")

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
    force_ocr = st.checkbox("Force OCR (skip text extraction)", value=False, help="Always use PaddleOCR on every page. More accurate for scanned or complex-layout PDFs.")
    use_ocr_cache = st.checkbox("Cache OCR result per PDF", value=True)
    page_workers = st.slider("Page parallel workers", min_value=1, max_value=8, value=4, step=1)
    st.divider()
    st.subheader("Text LLM Settings")
    text_llm_timeout = st.number_input(
        "Text LLM timeout (seconds)",
        min_value=60,
        max_value=1200,
        value=600,
        step=30,
    )
    text_llm_retries = st.number_input(
        "Text LLM retries",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
    )
    text_token_limit = st.number_input(
        "Combined context token limit",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
    )

if st.button("Select PDF Files"):
    files = select_pdf_files()
    if files:
        st.session_state["selected_pdfs"] = files

selected_pdfs = st.session_state.get("selected_pdfs", [])
if selected_pdfs:
    st.info(f"{len(selected_pdfs)} PDF(s) selected:\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in selected_pdfs))
else:
    st.info("No PDF files selected.")

if st.button("Run Extraction", type="primary"):
    if not selected_pdfs:
        st.error("Please select at least one PDF file first.")
    else:
        pdf_files = [p for p in selected_pdfs if os.path.exists(p)]
        if not pdf_files:
            st.warning("None of the selected PDF files could be found.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_results = []
            paddle_ocr = load_paddle_ocr(use_gpu=(compute_device == "cuda"))
            if paddle_ocr is None:
                st.error("PaddleOCR failed to load. Check installation: pip install paddleocr")
                st.stop()

            all_snippets = []
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
                            max_workers=page_workers,
                            force_ocr=force_ocr,
                        )
                    else:
                        lines = extract_pdf_lines_hybrid(
                            pdf,
                            poppler_path=st.session_state.get("poppler_path"),
                            ocr_dpi=ocr_dpi,
                            max_pages=max_pages,
                            max_workers=page_workers,
                            force_ocr=force_ocr,
                            paddle_ocr=paddle_ocr,
                        )

                    page_text = "\n".join(lines)
                    snippets = get_context_snippets(page_text)
                    if not snippets:
                        progress_bar.progress((idx + 1) / len(pdf_files))
                        continue
                    all_snippets.append(f"[FILE: {filename}]\n" + "\n".join(snippets))

                except Exception as e:
                    st.error(f"Error in {filename}: {e}")

                progress_bar.progress((idx + 1) / len(pdf_files))

            if all_snippets:
                combined_context = "\n\n".join(all_snippets)
                combined_context, was_truncated = truncate_text_to_tokens(
                    combined_context, token_limit=int(text_token_limit)
                )
                if was_truncated:
                    st.info(f"Combined context exceeded {int(text_token_limit)} tokens and was truncated.")
                approx_tokens = len(combined_context.split())
                print(f"\n{'='*60}")
                print(f"[DEBUG] Combined context sent to LLM (~{approx_tokens} tokens, truncated={was_truncated}):")
                print(f"{'='*60}")
                print(combined_context)
                print(f"{'='*60}\n")
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
                    items = call_ollama_json_with_retry(
                        ollama_ip=ollama_ip,
                        payload=payload,
                        timeout_sec=int(text_llm_timeout),
                        retries=int(text_llm_retries),
                    )
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

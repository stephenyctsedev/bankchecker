import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pypdfium2 as pdfium
except Exception:
    pdfium = None
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

# Month abbreviations used by Bangkok Bank date format (e.g. 01MAR24)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Match any interest-related keyword (English or Chinese)
_INTEREST_RE = re.compile(r"INTEREST|利息|INT\b", re.IGNORECASE)

# Match decimal numbers (currency values like 6.76 or 9,728.25)
# Using only decimal numbers avoids false positives from integer date parts (01, 31, 24)
_DECIMAL_RE = re.compile(r"[\d,]+\.\d+")


# ---------------------------------------------------------------------------
# Compute detection
# ---------------------------------------------------------------------------

def detect_compute():
    if torch is None:
        return "cpu", 0
    try:
        if torch.cuda.is_available():
            return "cuda", 1
    except Exception:
        pass
    return "cpu", 0


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

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
        return None


def extract_pdf_lines_hybrid(pdf_path, poppler_path=None, ocr_dpi=150, max_pages=0, max_workers=4, force_ocr=False, paddle_ocr=None):
    if pdfium is not None:
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
        if pdfplumber is not None:
            try:
                with pdfplumber.open(pdf_path) as plumb_doc:
                    page_text = plumb_doc.pages[page_idx].extract_text() or ""
            except Exception:
                page_text = ""
        if _is_substantial_text(page_text) and not _is_gibberish_text(page_text):
            print(f"[OCR] Page {page_idx + 1}: text extracted via pdfplumber ({len(page_text)} chars)")
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

    for page_idx in sorted(ocr_needed_pages):
        img = _render_single_page_image(pdf_path, page_idx, dpi=ocr_dpi, poppler_path=poppler_path)
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
                break
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


# ---------------------------------------------------------------------------
# Interest extraction (no LLM — pure regex)
# ---------------------------------------------------------------------------

def _parse_date(token: str):
    """
    Try to parse a single token as a date.
    Supports:
      - Bangkok Bank format: DDMMMYY  e.g. 01MAR24
      - Citibank format:     MM/DD/YY or MM/DD/YYYY  e.g. 01/31/24
    Returns 'YYYY-MM-DD' string or None.
    """
    # Bangkok Bank: 01MAR24
    m = re.fullmatch(r"(\d{2})([A-Z]{3})(\d{2})", token, re.IGNORECASE)
    if m:
        day, mon, yr = m.groups()
        month = _MONTH_MAP.get(mon.upper())
        if month:
            return f"20{yr}-{month:02d}-{int(day):02d}"
    # Citibank: 01/31/24 or 01/31/2024
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", token)
    if m:
        mm, dd, yr = m.groups()
        year = int(yr) if len(yr) == 4 else 2000 + int(yr)
        return f"{year}-{int(mm):02d}-{int(dd):02d}"
    return None


def _extract_description(line: str) -> str:
    """
    Strip date prefixes and trailing amount/balance numbers from a line
    to leave the core description text.
    """
    s = line.strip()
    # Remove up to two leading date tokens (Citibank repeats the date twice)
    for _ in range(2):
        s = re.sub(r"^\d{2}[A-Z]{3}\d{2}\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\d{1,2}/\d{1,2}/\d{2,4}\s*", "", s)
    # Remove leading 4-digit time/reference code (Bangkok Bank "0000")
    s = re.sub(r"^\d{4}\s+", "", s)
    # Remove trailing two decimal numbers (amount + running balance)
    s = re.sub(r"\s+[\d,]+\.\d+\s+[\d,]+\.\d+\s*$", "", s)
    # Remove one trailing decimal number if still present
    s = re.sub(r"\s+[\d,]+\.\d+\s*$", "", s)
    return s.strip()


def extract_interest_from_lines(lines: list, filename: str) -> list:
    """
    Scan text lines for interest transactions using keyword matching.
    No LLM required.

    Line formats handled:
      Bangkok Bank: 01MAR24 0000 INTEREST 利息收入 6.76 9,728.25
      Citibank:     01/31/24 01/31/24 存入利息 (JAN) 16.62 19,678.26

    Strategy:
      - Only decimal numbers (e.g. 6.76, 9,728.25) are collected — this avoids
        false positives from integer date parts like "01", "31", "24".
      - The second-to-last decimal number is the transaction amount.
      - The last decimal number is the running balance (ignored).
      - If only one decimal number is on the line it is taken as the amount.
    """
    results = []
    for line in lines:
        if not _INTEREST_RE.search(line):
            continue

        # Collect all decimal (currency-style) numbers on the line
        raw_nums = _DECIMAL_RE.findall(line)
        nums = []
        for n in raw_nums:
            try:
                nums.append(float(n.replace(",", "")))
            except ValueError:
                pass
        if not nums:
            continue

        # Second-to-last = amount, last = running balance
        amount = nums[-2] if len(nums) >= 2 else nums[-1]

        # Find the first recognisable date token; skip lines with no date
        date_str = ""
        for token in line.split():
            d = _parse_date(token)
            if d:
                date_str = d
                break
        if not date_str:
            print(f"[EXTRACT] {filename} | skipped (no date): {line!r}")
            continue

        description = _extract_description(line)
        print(f"[EXTRACT] {filename} | date={date_str} | amount={amount} | desc={description!r}")
        results.append({
            "source": filename,
            "date": date_str,
            "description": description,
            "amount": amount,
        })
    return results


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Bank Statement Interest Checker")
st.markdown("Extracts interest transactions from bank PDFs using keyword matching — no LLM required.")

with st.sidebar:
    st.header("Settings")
    compute_device, _ = detect_compute()
    st.caption(f"Compute: `{compute_device}`")

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
    force_ocr = st.checkbox(
        "Force OCR (skip text extraction)",
        value=False,
        help="Always use PaddleOCR on every page. Useful for scanned or complex-layout PDFs.",
    )
    use_ocr_cache = st.checkbox("Cache OCR result per PDF", value=True)
    page_workers = st.slider("Page parallel workers", min_value=1, max_value=8, value=4, step=1)

# Main page — file uploader (works in headless containers, no display required)
if "tmp_dir" not in st.session_state:
    st.session_state["tmp_dir"] = tempfile.mkdtemp()

uploaded = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

# Save each uploaded file to the session temp dir so the rest of the code
# can use normal file paths.
pdf_files = []
if uploaded:
    for uf in uploaded:
        dst = os.path.join(st.session_state["tmp_dir"], uf.name)
        with open(dst, "wb") as f:
            f.write(uf.getbuffer())
        pdf_files.append(dst)
    st.info(f"{len(pdf_files)} PDF(s) ready:\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in pdf_files))

if st.button("Run Extraction", type="primary"):
    if not pdf_files:
        st.error("Please upload at least one PDF file first.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        all_results = []

        paddle_ocr = load_paddle_ocr(use_gpu=(compute_device == "cuda"))
        if paddle_ocr is None:
            st.error("PaddleOCR failed to load. Check installation: pip install paddleocr")
            st.stop()

        for idx, pdf in enumerate(pdf_files):
            filename = os.path.basename(pdf)
            status_text.text(f"Processing ({idx + 1}/{len(pdf_files)}): {filename}")

            try:
                if use_ocr_cache:
                    lines = ocr_pdf_to_lines_cached(
                        pdf,
                        _pdf_signature(pdf),
                        poppler_path=st.session_state.get("poppler_path") or None,
                        ocr_dpi=ocr_dpi,
                        max_pages=max_pages,
                        max_workers=page_workers,
                        force_ocr=force_ocr,
                    )
                else:
                    lines = extract_pdf_lines_hybrid(
                        pdf,
                        poppler_path=st.session_state.get("poppler_path") or None,
                        ocr_dpi=ocr_dpi,
                        max_pages=max_pages,
                        max_workers=page_workers,
                        force_ocr=force_ocr,
                        paddle_ocr=paddle_ocr,
                    )

                records = extract_interest_from_lines(lines, filename)
                all_results.extend(records)

            except Exception as e:
                st.error(f"Error in {filename}: {e}")

            progress_bar.progress((idx + 1) / len(pdf_files))

        status_text.empty()

        if all_results:
            df = pd.DataFrame(all_results)
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)

            st.success(f"Extraction completed — {len(df)} interest record(s) found.")
            st.subheader("Interest Records")
            st.dataframe(df, use_container_width=True)

            total = df["amount"].sum()
            st.metric("Total Interest", f"{total:,.2f}")

            csv_data = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="interest_summary.csv",
                mime="text/csv",
            )
        else:
            st.warning("No interest records found.")

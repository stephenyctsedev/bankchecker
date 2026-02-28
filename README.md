# Bank Statement Interest Checker

A Streamlit web app that extracts interest credit transactions from bank PDF statements using text extraction and OCR — no LLM required.

**Live demo:** https://bankchecker.sharecloud-me.synology.me

## How it works

1. Upload one or more PDF bank statements
2. Text is extracted via **pdfplumber** (for digital PDFs) or **PaddleOCR** (for scanned/image-based PDFs)
3. Lines containing interest keywords (`INTEREST`, `利息`, `INT`) are matched by regex
4. Date and amount are parsed directly from the matched line
5. Results are displayed in a table and available as CSV download

## Privacy

Uploaded PDFs are **deleted from the server immediately after extraction completes** — no bank statements are stored at rest. Files exist on the server only for the duration of processing (typically a few seconds).

## Supported formats

| Bank | Date format | Example line |
|------|-------------|--------------|
| HKbea Bank | `DDMMMYY` | `01MAR24 0000 INTEREST 利息收入 6.76 9,728.25` |
| Citibank HK | `MM/DD/YY` | `01/31/24 01/31/24 存入利息 (JAN) 16.62 19,678.26` |

Amount detection: the **second-to-last** decimal number on the line is taken as the transaction amount; the last is the running balance.

## Local setup

### Prerequisites

- Python 3.10+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) (Windows: download and set path in sidebar)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows: install binary and add to PATH)

### Install

```bash
pip install -r requirements.txt
```

> **Note:** PaddlePaddle from PyPI requires a CPU with AVX2 support. Most modern desktop/laptop CPUs support AVX2. For NAS or older CPUs without AVX2, use the Docker deployment below — the Dockerfile installs the correct no-AVX2 build automatically.

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Docker / Synology NAS deployment

The Docker image handles all dependencies automatically:
- **Poppler** and **Tesseract OCR** (English + Traditional/Simplified Chinese) via apt
- **PaddlePaddle (no-AVX2)** installed from the official PaddlePaddle no-AVX2 wheel index — compatible with Synology NAS CPUs (Intel Celeron J-series) that do not support AVX2
- **PaddleOCR** installed after PaddlePaddle

### Build and run with Docker Compose

Clone the repository, then run:

```bash
git clone https://github.com/stephenyctsedev/bankchecker.git
cd bankchecker
docker compose up -d --build
```

Open `http://<host-ip>:8501`.

> **Note:** If you want to build directly from the GitHub URL without cloning, use the Docker CLI:
> ```bash
> docker build https://github.com/stephenyctsedev/bankchecker.git
> ```
> This is not supported by GUI container managers (e.g. Portainer, Synology Container Manager).

### Synology Container Manager

1. Clone or download the full repository to your NAS (e.g. `/docker/bankchecker/`) via File Station
2. Open **Container Manager → Project → Create**
3. Set the path to `/docker/bankchecker/`
4. Click **Build** — Container Manager detects `docker-compose.yml` automatically

> First build takes several minutes (PaddleOCR models are downloaded on first use).
> Subsequent restarts are fast.

### Synology reverse proxy (HTTPS)

If serving via Synology's reverse proxy over HTTPS, enable **WebSocket** in the reverse proxy rule:

> **Application Portal → Reverse Proxy → your rule → Edit → enable WebSocket**

The destination should be `http://localhost:8501` (plain HTTP — TLS is terminated by the proxy).

### Verify PaddleOCR installation

After the container starts, run the built-in verification script:

```bash
docker exec bank-statement-checker python verify_ocr.py
```

This checks PaddlePaddle import, no-AVX2 wheel provenance, PaddleOCR import, and runs a test inference.

## Sidebar settings

| Setting | Description |
|---------|-------------|
| Poppler Path | Windows only — path to Poppler `bin/` directory. Leave empty in Docker. |
| Max pages per PDF | Limit pages processed (0 = all). Useful for large statements. |
| Force OCR | Skip text extraction and always use PaddleOCR (for fully scanned PDFs). |
| Cache OCR result | Cache extracted text per PDF to avoid re-processing on rerun. |
| Page parallel workers | Number of threads for simultaneous page text extraction. |

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI |
| pdfplumber | Digital PDF text extraction (preserves column order) |
| pypdfium2 | PDF page rendering for OCR |
| pdf2image | PDF to image conversion (requires Poppler) |
| paddleocr | OCR for scanned pages (Traditional/Simplified Chinese + English) |
| paddlepaddle | PaddleOCR backend — no-AVX2 build used in Docker for NAS compatibility |
| pytesseract | Fallback OCR engine (requires Tesseract binary) |
| pandas | Result table and CSV export |

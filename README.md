# Bank Statement Interest Checker

A Streamlit web app that extracts interest credit transactions from bank PDF statements using text extraction and OCR — no LLM required.

**Live demo:** https://bankchecker.sharecloud-me.synology.me

## How it works

1. Upload one or more PDF bank statements
2. Text is extracted via **pdfplumber** (for digital PDFs) or **Tesseract OCR** (for scanned/image-based PDFs)
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
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (Windows: install the UB-Mannheim binary and ensure it is on your PATH or use the default install location `C:\Program Files\Tesseract-OCR\`)
  - During installation, select **Additional language data → Chinese (Simplified)** and **Chinese (Traditional)**

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Docker / Synology NAS deployment

The Docker image handles all dependencies automatically:
- **Poppler** and **Tesseract OCR** (English + Traditional/Simplified Chinese) installed via apt
- No large ML model downloads — Tesseract works immediately after the image is built

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

> First build takes a minute or two (system packages are installed via apt).
> Subsequent restarts are fast — no model downloads.

### Synology reverse proxy (HTTPS)

If serving via Synology's reverse proxy over HTTPS, enable **WebSocket** in the reverse proxy rule:

> **Application Portal → Reverse Proxy → your rule → Edit → enable WebSocket**

The destination should be `http://localhost:8501` (plain HTTP — TLS is terminated by the proxy).

## Sidebar settings

| Setting | Description |
|---------|-------------|
| Poppler Path | Windows only — path to Poppler `bin/` directory. Leave empty in Docker. |
| Max pages per PDF | Limit pages processed (0 = all). Useful for large statements. |
| Force OCR | Skip text extraction and always use Tesseract OCR (for fully scanned PDFs). |
| Cache OCR result | Cache extracted text per PDF to avoid re-processing on rerun. |
| Page parallel workers | Number of threads for simultaneous page text extraction. |

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web UI |
| pdfplumber | Digital PDF text extraction (preserves column order) |
| pypdfium2 | PDF page rendering for OCR fallback |
| pdf2image | PDF to image conversion (requires Poppler) |
| pytesseract | OCR for scanned pages (requires Tesseract binary) |
| pandas | Result table and CSV export |

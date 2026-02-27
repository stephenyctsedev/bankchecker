# Bank Statement Interest Checker

A Streamlit web app that extracts interest credit transactions from bank PDF statements using text extraction and OCR — no LLM required.

## How it works

1. Upload one or more PDF bank statements
2. Text is extracted via **pdfplumber** (for digital PDFs) or **PaddleOCR** (for scanned/image-based PDFs)
3. Lines containing interest keywords (`INTEREST`, `利息`, `INT`) are matched by regex
4. Date and amount are parsed directly from the matched line
5. Results are displayed in a table and available as CSV download

## Supported formats

| Bank | Date format | Example line |
|------|-------------|--------------|
| Bangkok Bank | `DDMMMYY` | `01MAR24 0000 INTEREST 利息收入 6.76 9,728.25` |
| Citibank HK | `MM/DD/YY` | `01/31/24 01/31/24 存入利息 (JAN) 16.62 19,678.26` |

Amount detection: the **second-to-last** decimal number on the line is taken as the transaction amount; the last is the running balance.

## Local setup

### Prerequisites

- Python 3.10+
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) (Windows: download and set path in sidebar)

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

Poppler is included in the Docker image — no extra configuration needed.

### Build and run with Docker Compose

```bash
docker compose up -d --build
```

Open `http://<host-ip>:8501`.

### Synology Container Manager

1. Copy the project folder to your NAS (e.g. `/docker/bankchecker/`) via File Station
2. Open **Container Manager → Project → Create**
3. Set the path to `/docker/bankchecker/`
4. Click **Build** — Container Manager detects `docker-compose.yml` automatically

> First build takes several minutes (PaddlePaddle + PaddleOCR are ~1 GB).
> Subsequent restarts are fast.

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
| PyMuPDF | Fallback text extraction |
| pypdfium2 | PDF page rendering for OCR |
| pdf2image | PDF to image conversion (requires Poppler) |
| paddleocr 2.7.3 | OCR for scanned pages |
| paddlepaddle 2.6.2 | PaddleOCR backend |
| pandas | Result table and CSV export |

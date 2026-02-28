FROM python:3.10-slim-bookworm

# System packages:
#   poppler-utils  — pdf2image PDF rendering
#   tesseract-ocr  — fallback OCR engine
#   libgl1         — OpenCV / PaddleOCR (replaces libgl1-mesa-glx in bookworm+)
#   libglib2.0-0   — OpenCV runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PaddlePaddle (CPU, no-AVX2) from the official PaddlePaddle no-AVX2
# wheel index.  This MUST come before requirements.txt so that paddleocr picks
# up this build rather than the default AVX2 wheel from PyPI.
RUN pip install --no-cache-dir paddlepaddle \
        -f https://www.paddlepaddle.org.cn/whl/linux/cpu-noavx/stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py verify_ocr.py ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]

FROM python:3.10-slim-bookworm

# System packages:
#   poppler-utils  — pdf2image PDF rendering
#   libgomp1       — OpenMP runtime required by onnxruntime
#   libgl1         — OpenGL runtime required by opencv (used by rapidocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgomp1 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]

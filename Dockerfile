FROM python:3.12-slim

# Install system dependencies (Tesseract + build tools)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8000

# Streamlit startup command
CMD ["streamlit", "run", "biopsy_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
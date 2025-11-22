FROM python:3.12-slim

# Install system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
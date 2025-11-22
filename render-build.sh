#!/bin/bash
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev
pip install -r requirements.txt
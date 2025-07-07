# Smart_Scan

Smart_Scan is a Flask-based web application for object detection and product quality analysis using AI. It supports real-time detection of fruits and packaged goods, along with text extraction for quality labels using OCR.

## Features

- Real-time object detection using YOLO
- Product and fruit quality assessment
- OCR-powered text extraction via EasyOCR
- Confidence graph visualization
- End-to-end deployment-ready with Flask

## Tech Stack

- Python, Flask
- YOLO (You Only Look Once)
- EasyOCR
- Selenium, Scrapy (for dataset collection)
- HTML, CSS

## Architecture

1. **Frontend**: HTML + CSS
2. **Backend**: Flask server handling image uploads and inference
3. **Model Inference**: YOLOv5 models for detection, EasyOCR for text
4. **Visualization**: Graphs and annotated images for output display

## Setup Instructions

```bash
git clone https://github.com/Sundramrai3691/Smart_Scan.git
cd Smart_Scan
pip install -r requirements.txt
python app.py

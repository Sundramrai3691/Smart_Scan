# Smart_Scan

Smart_Scan is a Flask web app for AI-assisted shelf/product scanning.  
It detects objects from uploaded images, estimates quality/freshness, extracts OCR text for product scans, and exposes summary/history/trend APIs.

## Features

- YOLO-based object detection for `product` and `fruit` modes
- Freshness scoring and shelf-gap summary metrics
- OCR extraction for product scans (EasyOCR)
- Annotated output image and confidence graph rendering
- Scan history + trends via SQLite-backed APIs
- Live alerts endpoint for latest scan

## Tech Stack

- Python + Flask
- PyTorch + Ultralytics YOLO
- EasyOCR + OpenCV + Pillow
- Matplotlib + NumPy
- SQLite

## Project Structure

- `app.py` - Flask app, inference pipeline, routes
- `scan.py` - scan-related helper logic
- `database.py` - schema + history/trend access
- `alerts.py` - alert evaluation rules
- `config.py` - thresholds/model paths/shelf settings
- `templates/` - HTML templates
- `static/` - CSS/assets
- `test_smoke.py` - runtime smoke tests

## Setup (Windows / PowerShell)

> Recommended Python: **3.12.x** (`runtime.txt` pins `python-3.12.5`).

```powershell
cd C:\github-all\Smart_Scan
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Then open: `http://127.0.0.1:5000/`

## Quick Health Check

With the server running in one terminal, run in another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python test_smoke.py
```

Expected output includes:

- `PASS /api/history`
- `PASS /api/trends`
- `PASS /detect`
- `All smoke tests passed.`

## API Endpoints

- `GET /` - main UI
- `POST /detect` - run detection (`image` file + `mode=product|fruit`)
- `GET /api/history` - scan history
- `GET /api/trends` - trend/aggregate data
- `GET /api/alerts/live` - alerts for latest scan
- `GET /image` - last annotated image buffer
- `GET /graph` - last confidence graph buffer

Operational utility routes:

- `GET /favicon.ico` returns `204`
- `GET /.well-known/appspecific/com.chrome.devtools.json` returns `204`

These are intentionally handled to keep development logs clean.

## Notes

- First install can be slow due to large ML dependencies (`torch`, `ultralytics`, `easyocr`).
- CPU mode is supported; CUDA is used automatically if available.
- Model files (`*.pt`, `*.pkl`) are loaded at startup from paths in `config.py`.

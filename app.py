from flask import Flask, request, render_template, jsonify, send_file
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import easyocr
import cv2
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import json
import sqlite3

from database import DB_PATH, init_db, log_scan, get_history, get_trend_data
from alerts import check_alerts
from config import (
    EXPECTED_SHELF_SLOTS,
    YOLO_CONFIDENCE_THRESHOLD,
    PRODUCT_MODEL_PATH,
    QUALITY_MODEL_PATH,
    QUALITY_CLASSIFIER_PATH,
)

app = Flask(__name__)


# Load YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model_product = YOLO(PRODUCT_MODEL_PATH).to(device)
    model_fruit = YOLO(QUALITY_MODEL_PATH).to(device)
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# EasyOCR Reader
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    exit()

# Quality classifier
try:
    with open(QUALITY_CLASSIFIER_PATH, 'rb') as f:
        quality_classifier = pickle.load(f)
except Exception as e:
    print(f"Warning: failed to load Quality.pkl: {e}")
    quality_classifier = None

# Initialize database
init_db()

# Temporary storage for buffers
buffers = {}


def _predict_freshness_score(crop_np, fallback_confidence):
    fallback_score = int(max(0, min(100, fallback_confidence * 100)))
    if quality_classifier is None or crop_np.size == 0:
        return fallback_score

    try:
        resized = cv2.resize(crop_np, (64, 64))
        features = resized.astype(np.float32).flatten().reshape(1, -1)

        if hasattr(quality_classifier, "predict_proba"):
            proba = quality_classifier.predict_proba(features)
            if len(proba) > 0 and len(proba[0]) > 1:
                return int(max(0, min(100, round(float(np.max(proba[0])) * 100))))

        pred = quality_classifier.predict(features)
        value = float(pred[0])
        if 0 <= value <= 1:
            value *= 100
        return int(max(0, min(100, round(value))))
    except Exception:
        return fallback_score


# Helper function for object detection
def detect_objects(image, model, mode):
    try:
        results = model(image)
        detected_objects = []
        img_h, img_w = image.shape[:2]

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, label, confidence in zip(boxes, labels, confidences):
                if confidence <= YOLO_CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = box
                x1 = max(0, min(img_w - 1, int(x1)))
                y1 = max(0, min(img_h - 1, int(y1)))
                x2 = max(0, min(img_w, int(x2)))
                y2 = max(0, min(img_h, int(y2)))
                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = model.names[int(label)]
                crop = image[y1:y2, x1:x2]
                freshness_score = int(round(float(confidence) * 100))
                if mode == "fruit":
                    freshness_score = _predict_freshness_score(crop, float(confidence))

                detected_objects.append({
                    "label": class_name,
                    "confidence": round(float(confidence), 4),
                    "freshness_score": freshness_score,
                    "bbox": [x1, y1, x2, y2]
                })
        return detected_objects
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return []


# Draw bounding boxes on the image
def draw_bounding_boxes(image, detections):
    try:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=20)
        except IOError:
            print("Font not found. Using default font.")
            font = ImageFont.load_default()

        for obj in detections:
            x1, y1, x2, y2 = obj["bbox"]
            class_name = obj["label"]
            confidence = obj["confidence"]
            freshness = obj.get("freshness_score")

            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            label = f"{class_name}: {confidence:.2f}"
            if freshness is not None:
                label = f"{label} | F:{freshness}"
            try:
                text_size = draw.textsize(label, font=font)
                draw.text((x1, y1 - text_size[1] - 5), label, fill="red", font=font)
            except Exception as e:
                print(f"Error drawing text: {e}")

        return image
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        return image


# Generate a confidence graph
def generate_confidence_graph(detections):
    try:
        classes = [obj["label"] for obj in detections]
        confidences = [obj["confidence"] for obj in detections]

        plt.figure(figsize=(8, 4))
        plt.bar(classes, confidences, color='blue')
        plt.xlabel('Class')
        plt.ylabel('Confidence')
        plt.title('Confidence Values')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    except Exception as e:
        print(f"Error generating graph: {e}")
        return io.BytesIO()


def _get_last_scan_alerts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT mode, detections_json
        FROM scans
        ORDER BY id DESC
        LIMIT 1
    """)
    row = c.fetchone()
    conn.close()

    if not row:
        return []

    mode, detections_json = row
    detections = json.loads(detections_json) if detections_json else []
    return check_alerts(detections, mode)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        uploaded_file = request.files.get('image')
        mode = request.form.get('mode')
        if not uploaded_file or mode not in ("product", "fruit"):
            return jsonify({"success": False, "error": "Invalid input!"}), 400

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        model = model_product if mode == "product" else model_fruit
        detections = detect_objects(image_np, model, mode)

        image_with_boxes = draw_bounding_boxes(image.copy(), detections)
        image_buf = io.BytesIO()
        image_with_boxes.save(image_buf, format='PNG')
        image_buf.seek(0)

        graph_buf = generate_confidence_graph(detections)
        buffers["image"] = image_buf
        buffers["graph"] = graph_buf

        ocr_text = None
        if mode == "product":
            try:
                _, img_encoded = cv2.imencode('.png', image_np)
                image_bytes = img_encoded.tobytes()
                ocr_results = reader.readtext(image_bytes)
                ocr_text = "\n".join([result[1] for result in ocr_results]) if ocr_results else None
            except Exception as ocr_error:
                print(f"Error during OCR: {ocr_error}")
                ocr_text = None

        product_count = len(detections)
        avg_freshness = round(
            float(np.mean([d["freshness_score"] for d in detections])) if detections else 0.0, 2
        )
        shelf_gaps = max(0, EXPECTED_SHELF_SLOTS - len(detections))

        alerts = check_alerts(detections, mode)
        flagged_count = len(alerts)

        summary = {
            "product_count": product_count,
            "avg_freshness": avg_freshness,
            "flagged_count": flagged_count,
            "shelf_gaps": shelf_gaps
        }

        log_scan(mode, product_count, avg_freshness, flagged_count, shelf_gaps, detections)

        return jsonify({
            "success": True,
            "detections": detections,
            "summary": summary,
            "alerts": alerts,
            "ocr_text": ocr_text
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/history')
def api_history():
    try:
        return jsonify(get_history())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/trends')
def api_trends():
    try:
        return jsonify(get_trend_data())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/alerts/live')
def api_alerts_live():
    try:
        return jsonify(_get_last_scan_alerts())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/image')
def serve_image():
    image_buf = buffers.get('image')
    if image_buf:
        return send_file(image_buf, mimetype='image/png')
    return jsonify({'error': 'Image not found!'}), 404


@app.route('/graph')
def serve_graph():
    graph_buf = buffers.get('graph')
    if graph_buf:
        return send_file(graph_buf, mimetype='image/png')
    return jsonify({'error': 'Graph not found!'}), 404


if __name__ == "__main__":
    app.run(debug=True)
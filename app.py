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

app = Flask(__name__)

# Load YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    model_product = YOLO('best.pt').to(device)
    model_fruit = YOLO('Quality.pt').to(device)
except Exception as e:
    print(f"Error loading YOLO models: {e}")
    exit()

# EasyOCR Reader
try:
    reader = easyocr.Reader(['en'])
except Exception as e:
    print(f"Error loading EasyOCR: {e}")
    exit()

# Temporary storage for buffers
buffers = {}

# Helper function for object detection
def detect_objects(image, model):
    try:
        resized_image = cv2.resize(image, (640, 640))
        results = model(resized_image)

        detected_objects = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, label, confidence in zip(boxes, labels, confidences):
                if confidence > 0.5:
                    x1, y1, x2, y2 = box
                    class_name = model.names[int(label)]
                    detected_objects.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'box': [int(x1), int(y1), int(x2), int(y2)]
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
            x1, y1, x2, y2 = obj['box']
            class_name = obj['class']
            confidence = obj['confidence']

            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

            label = f"{class_name}: {confidence:.2f}"
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
        classes = [obj['class'] for obj in detections]
        confidences = [obj['confidence'] for obj in detections]

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        uploaded_file = request.files.get('image')
        mode = request.form.get('mode')

        if uploaded_file and mode:
            image = Image.open(uploaded_file).convert('RGB')
            original_width, original_height = image.size

            image_np = np.array(image)

            model = model_product if mode == 'product' else model_fruit
            detections = detect_objects(image_np, model)

            image_with_boxes = draw_bounding_boxes(image.copy(), detections)

            image_with_boxes = image_with_boxes.resize((original_width, original_height))

            image_buf = io.BytesIO()
            image_with_boxes.save(image_buf, format='PNG')
            image_buf.seek(0)

            graph_buf = generate_confidence_graph(detections)

            ocr_text = ""
            if mode == 'product':
                try:
                    image_np = np.array(image)
                    _, img_encoded = cv2.imencode('.png', image_np)
                    image_bytes = img_encoded.tobytes()
                    ocr_results = reader.readtext(image_bytes)
                    ocr_text = "\n".join([result[1] for result in ocr_results])
                except Exception as e:
                    print(f"Error during OCR: {e}")

            buffers['image'] = image_buf
            buffers['graph'] = graph_buf

            return jsonify({
                'detections': detections,
                'image_url': '/image',
                'graph_url': '/graph',
                'ocr_text': ocr_text
            })

        return jsonify({'error': 'Invalid input!'}), 400
    except Exception as e:
        print(f"Error in /detect route: {e}")
        return jsonify({'error': 'An error occurred during processing.'}), 500

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
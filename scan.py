import argparse
import io
import json
import os
import sys

import requests
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="ShelfIQ scan runner")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--mode", default="product", choices=["product", "fruit"], help="Scan mode")
    parser.add_argument("--base", default="http://localhost:5000", help="Server base URL")
    parser.add_argument("--no-server", action="store_true", help="Run inference inline without Flask API")
    return parser.parse_args()


def print_result(body):
    print(json.dumps(body, indent=2))
    summary = body.get("summary", {})
    print(
        "Summary:",
        f"items={summary.get('product_count', 0)}",
        f"avg_freshness={summary.get('avg_freshness', 0)}",
        f"flagged={summary.get('flagged_count', 0)}",
        f"gaps={summary.get('shelf_gaps', 0)}",
    )


def run_server_mode(args):
    with open(args.image, "rb") as f:
        resp = requests.post(
            f"{args.base}/detect",
            files={"image": (os.path.basename(args.image), f, "image/jpeg")},
            data={"mode": args.mode},
            timeout=60,
        )
    resp.raise_for_status()
    body = resp.json()
    print_result(body)


def run_offline_mode(args):
    import numpy as np
    import cv2
    import app as app_module

    from alerts import check_alerts
    from config import EXPECTED_SHELF_SLOTS

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)

    model = app_module.model_product if args.mode == "product" else app_module.model_fruit
    detections = app_module.detect_objects(image_np, model, args.mode)

    image_with_boxes = app_module.draw_bounding_boxes(image.copy(), detections)
    image_buf = io.BytesIO()
    image_with_boxes.save(image_buf, format="PNG")
    image_buf.seek(0)
    app_module.buffers["image"] = image_buf
    app_module.buffers["graph"] = app_module.generate_confidence_graph(detections)

    ocr_text = None
    if args.mode == "product":
        try:
            _, img_encoded = cv2.imencode(".png", image_np)
            image_bytes = img_encoded.tobytes()
            ocr_results = app_module.reader.readtext(image_bytes)
            ocr_text = "\n".join([result[1] for result in ocr_results]) if ocr_results else None
        except Exception as ocr_error:
            print(f"OCR warning: {ocr_error}")
            ocr_text = None

    product_count = len(detections)
    avg_freshness = round(
        float(np.mean([d["freshness_score"] for d in detections])) if detections else 0.0, 2
    )
    shelf_gaps = max(0, EXPECTED_SHELF_SLOTS - len(detections))
    alerts = check_alerts(detections, args.mode)
    flagged_count = len(alerts)

    summary = {
        "product_count": product_count,
        "avg_freshness": avg_freshness,
        "flagged_count": flagged_count,
        "shelf_gaps": shelf_gaps,
    }
    body = {
        "success": True,
        "detections": detections,
        "summary": summary,
        "alerts": alerts,
        "ocr_text": ocr_text,
    }
    print_result(body)


def main():
    args = parse_args()
    try:
        if args.no_server:
            run_offline_mode(args)
        else:
            run_server_mode(args)
    except Exception as e:
        print(f"Scan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

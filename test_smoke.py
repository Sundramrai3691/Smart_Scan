import requests
import os
import sys

BASE = "http://localhost:5000"


def test_history_endpoint():
    r = requests.get(f"{BASE}/api/history")
    assert r.status_code == 200, f"History failed: {r.status_code}"
    data = r.json()
    assert isinstance(data, list), "History must return a list"
    print("PASS /api/history")


def test_trends_endpoint():
    r = requests.get(f"{BASE}/api/trends")
    assert r.status_code == 200, f"Trends failed: {r.status_code}"
    print("PASS /api/trends")


def test_detect_endpoint():
    # Use any small test image in the project, or create a 10x10 white image
    import io
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(200, 200, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        r = requests.post(
            f"{BASE}/detect",
            files={"image": ("test.jpg", buf, "image/jpeg")},
            data={"mode": "product"}
        )
        assert r.status_code == 200, f"Detect failed HTTP: {r.status_code}"
        body = r.json()
        assert "success" in body, "Missing 'success' key"
        assert "summary" in body, "Missing 'summary' key"
        assert "detections" in body, "Missing 'detections' key"
        summary = body["summary"]
        for key in ["product_count", "avg_freshness", "flagged_count", "shelf_gaps"]:
            assert key in summary, f"Missing summary key: {key}"
        print(f"PASS /detect — returned {body['summary']['product_count']} detections")
    except Exception as e:
        print(f"FAIL /detect — {e}")
        sys.exit(1)


if __name__ == "__main__":
    print("Running ShelfIQ smoke tests against", BASE)
    try:
        test_history_endpoint()
        test_trends_endpoint()
        test_detect_endpoint()
        print("\nAll smoke tests passed.")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

# ShelfIQ configuration — edit these values without touching app.py or alerts.py

# Detection settings
EXPECTED_SHELF_SLOTS = 8       # How many products a full shelf holds
YOLO_CONFIDENCE_THRESHOLD = 0.4  # Min YOLO confidence to count a detection

# Alert thresholds
FRESHNESS_ALERT_THRESHOLD = 60   # Freshness score below this triggers an alert
STOCK_ALERT_THRESHOLD = 5        # Detections below this count trigger low-stock alert
EXPIRY_WARNING_DAYS = 3          # Days until expiry to trigger warning

# Model paths
PRODUCT_MODEL_PATH = "best.pt"
QUALITY_MODEL_PATH = "Quality.pt"
QUALITY_CLASSIFIER_PATH = "Quality.pkl"

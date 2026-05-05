FRESHNESS_THRESHOLD = 60
STOCK_THRESHOLD = 5
EXPIRY_DAYS_WARNING = 3


def check_alerts(detections, mode):
    alerts = []
    if mode == "fruit":
        for d in detections:
            score = d.get("freshness_score", 100)
            if score < FRESHNESS_THRESHOLD:
                alerts.append({
                    "type": "freshness",
                    "severity": "high" if score < 40 else "medium",
                    "message": f"{d.get('label','Item')} freshness critical: {score}/100",
                    "item": d.get("label")
                })
    if mode == "product":
        category_counts = {}
        for d in detections:
            label = d.get("label", "unknown")
            category_counts[label] = category_counts.get(label, 0) + 1
        for category, count in category_counts.items():
            if count < STOCK_THRESHOLD:
                alerts.append({
                    "type": "stock",
                    "severity": "medium",
                    "message": f"Low stock: {category} ({count} units detected)",
                    "item": category
                })
    return alerts

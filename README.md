---
title: Anomaly Detection API
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

# 🔍 Anomaly Detection API

Real-time anomaly detection for electrical components using PatchCore + OpenCV classification.

## 🚀 Quick Start

### API Endpoint

**POST** `/infer`

**Request:**
```json
{
  "image_url": "https://example.com/your-image.jpg"
}
```

**Response:**
```json
{
  "label": "Normal",
  "boxed_url": "https://cloudinary.com/boxed_image.jpg",
  "mask_url": "https://cloudinary.com/anomaly_mask.png",
  "filtered_url": "https://cloudinary.com/filtered_anomalies.png",
  "boxes": []
}
```

### Example Usage

```bash
curl -X POST "https://YOUR_USERNAME-anomaly-detection-api.hf.space/infer" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg"}'
```

```python
import requests

response = requests.post(
    "https://YOUR_USERNAME-anomaly-detection-api.hf.space/infer",
    json={"image_url": "https://example.com/test.jpg"}
)

result = response.json()
print(f"Classification: {result['label']}")
print(f"Boxed Image: {result['boxed_url']}")
```

## 📋 Classification Labels

- **Normal** - No anomalies detected
- **Full Wire Overload** - Entire wire showing overload
- **Point Overload (Faulty)** - Localized overload points

## 🔧 Technical Details

- **Model:** PatchCore (anomaly detection)
- **Classification:** OpenCV-based heuristics
- **Response Time:** ~5 seconds
- **Max Image Size:** Unlimited (auto-resized)

## 🌐 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/health` | GET | Health check |
| `/infer` | POST | Run inference |

## 📦 Output Files

All processed images are uploaded to Cloudinary:
- **boxed_url:** Original image with bounding boxes
- **mask_url:** Grayscale anomaly heatmap
- **filtered_url:** Filtered image showing only anomalous regions

## 🛠️ Built With

- PyTorch 2.4.1
- Anomalib (PatchCore)
- OpenCV
- Flask
- Cloudinary

## 📄 License

MIT License

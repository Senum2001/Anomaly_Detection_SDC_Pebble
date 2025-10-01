---
title: Anomaly Detection API
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Anomaly Detection API

Real-time anomaly detection API using PatchCore + OpenCV.

## Usage

Send POST request to `/infer` with JSON body:

```json
{
  "image_url": "https://example.com/image.jpg"
}
```

Response:

```json
{
  "label": "Normal",
  "boxed_url": "https://cloudinary.com/...",
  "mask_url": "https://cloudinary.com/...",
  "filtered_url": "https://cloudinary.com/...",
  "boxes": []
}
```

## Endpoints

- `GET /` - API documentation
- `GET /health` - Health check
- `POST /infer` - Run inference

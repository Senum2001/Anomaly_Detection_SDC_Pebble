# Hugging Face Spaces Deployment Guide

## Why Hugging Face Spaces?
- ✅ FREE forever for public spaces
- ✅ Direct JSON responses (no job queue)
- ✅ Free GPU (T4)
- ✅ Custom domain support
- ✅ Same response format as your local Flask app

## Setup Steps

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Create free account

### 2. Create New Space
- Go to https://huggingface.co/new-space
- Name: `anomaly-detection-api` (or your choice)
- License: MIT
- SDK: **Docker**
- Hardware: **CPU basic** (upgrade to GPU later if needed)

### 3. Clone Your Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/anomaly-detection-api
cd anomaly-detection-api
```

### 4. Copy Files to Space
Copy these files from your project:
- `app.py` → Main Flask app
- `inference_core.py` → Core inference logic
- `requirements.txt` → Dependencies
- `Dockerfile.hf` → Rename to `Dockerfile`
- `scripts/` folder → PatchCore scripts
- `configs/` folder → Config files
- `README_HF.md` → Rename to `README.md`

### 5. Push to Hugging Face
```bash
git add .
git commit -m "Initial deployment"
git push
```

### 6. Wait for Build (5-10 minutes)
Hugging Face will automatically build your Docker image and deploy.

### 7. Test Your API
```bash
curl -X POST "https://YOUR_USERNAME-anomaly-detection-api.hf.space/infer" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg"}'
```

### Response (Direct JSON, no queue!)
```json
{
  "label": "Normal",
  "boxed_url": "https://cloudinary.com/...",
  "mask_url": "https://cloudinary.com/...",
  "filtered_url": "https://cloudinary.com/...",
  "boxes": []
}
```

## Upgrade to GPU (Optional)
- Go to your Space settings
- Change hardware to **T4 small** (FREE)
- Restart space

## Cost
- **$0/month** for CPU
- **$0/month** for T4 GPU (limited hours, but generous for development)

---

## Alternative: Modal.com

If you prefer Python decorators over Docker:

### 1. Install Modal
```bash
pip install modal
```

### 2. Create modal_app.py
```python
import modal

stub = modal.Stub("anomaly-detection")
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@stub.function(image=image, gpu="T4")
@modal.web_endpoint(method="POST")
def infer(data: dict):
    from inference_core import run_pipeline_for_image, download_image_from_url, upload_to_cloudinary
    import os
    
    image_url = data.get("image_url")
    local_path = download_image_from_url(image_url)
    results = run_pipeline_for_image(local_path)
    
    boxed_url = upload_to_cloudinary(results["boxed_path"], folder="pipeline_outputs")
    mask_url = upload_to_cloudinary(results["mask_path"], folder="pipeline_outputs")
    filtered_url = upload_to_cloudinary(results["filtered_path"], folder="pipeline_outputs")
    
    os.remove(local_path)
    
    return {
        "label": results["label"],
        "boxed_url": boxed_url,
        "mask_url": mask_url,
        "filtered_url": filtered_url,
        "boxes": results.get("boxes", [])
    }
```

### 3. Deploy
```bash
modal deploy modal_app.py
```

You'll get a URL like: `https://your-username--anomaly-detection-infer.modal.run`

---

## Comparison

| Service | Free GPU | Direct JSON | Setup | Cold Start |
|---------|----------|-------------|-------|------------|
| **Hugging Face** | ✅ T4 | ✅ Yes | Docker | ~10s |
| **Modal** | ✅ T4/A10G | ✅ Yes | Python | ~5s |
| **RunPod** | ✅ Yes | ❌ Job Queue | Docker | ~5s |
| **Replicate** | ✅ Yes | ⚠️ Webhook | Docker | ~10s |

**Recommendation:** Start with **Hugging Face Spaces** - it's the easiest and completely free!

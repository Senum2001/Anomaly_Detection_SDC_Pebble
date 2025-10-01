# 🚀 Quick Start: Deploy to Hugging Face Spaces

## Step 1: Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Sign up (it's free!)

## Step 2: Create New Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `anomaly-detection-api` (or your choice)
   - **License:** MIT
   - **Select SDK:** **Docker** ⚠️ (Important!)
   - **Hardware:** CPU basic (free)
   - Make it **Public** (for free hosting)
3. Click **Create Space**

## Step 3: Get Your Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `spaces-deploy`
4. Role: **Write**
5. Copy the token (you'll need it)

## Step 4: Run Deployment Script

Open PowerShell in this directory and run:

```powershell
.\deploy_hf.ps1
```

Or manually follow these steps:

### Manual Deployment:

```powershell
# 1. Clone your Space (replace YOUR_USERNAME and YOUR_SPACE_NAME)
cd ..
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# 2. Copy files from your project
Copy-Item ..\AI\app.py .
Copy-Item ..\AI\inference_core.py .
Copy-Item ..\AI\requirements.txt .
Copy-Item ..\AI\README.md .
Copy-Item ..\AI\Dockerfile.hf .\Dockerfile
Copy-Item -Recurse ..\AI\scripts .
Copy-Item -Recurse ..\AI\configs .

# 3. Commit and push
git add .
git commit -m "Deploy Anomaly Detection API"
git push
```

When prompted for credentials:
- **Username:** Your Hugging Face username
- **Password:** Your access token (from Step 3)

## Step 5: Wait for Build

1. Go to your Space page: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click on **Logs** tab to watch the build
3. Build takes ~5-10 minutes (downloading model + dependencies)

## Step 6: Test Your API!

Once the build completes and shows "Running", test it:

```bash
curl -X POST "https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space/infer" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/test.jpg"}'
```

Replace:
- `YOUR_USERNAME` with your HF username
- `YOUR_SPACE_NAME` with your space name

## Expected Response:

```json
{
  "label": "Normal",
  "boxed_url": "https://res.cloudinary.com/...",
  "mask_url": "https://res.cloudinary.com/...",
  "filtered_url": "https://res.cloudinary.com/...",
  "boxes": []
}
```

## 🎉 Done!

Your API is now live and FREE!

### API Endpoints:
- `GET /` - API docs
- `GET /health` - Health check  
- `POST /infer` - Run inference

### Upgrade to GPU (Optional):
1. Go to Space Settings
2. Change Hardware to **T4 small** (still free!)
3. Restart Space

---

## Troubleshooting

### Build fails?
- Check **Logs** tab in your Space
- Common issues:
  - Model download timeout → Retry build
  - Missing files → Check all files copied
  - Dependencies error → Check requirements.txt

### Need help?
- Hugging Face docs: https://huggingface.co/docs/hub/spaces-sdks-docker
- Community: https://discuss.huggingface.co/

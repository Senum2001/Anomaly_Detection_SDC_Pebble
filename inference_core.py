"""
Core inference module - contains model loading and inference functions
Can be imported by both Flask app and RunPod handler
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch
import subprocess
import sys
import cloudinary
import cloudinary.uploader

# ---- Import your PatchCore API ----
from scripts.patchcore_api_inference import Patchcore, config, device

# ---- Output directories ----
OUT_MASK_DIR = "api_inference_pred_masks_pipeline"
OUT_FILTERED_DIR = "api_inference_filtered_pipeline"
OUT_BOXED_DIR = "api_inference_labeled_boxes_pipeline"

os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_FILTERED_DIR, exist_ok=True)
os.makedirs(OUT_BOXED_DIR, exist_ok=True)

# ---- Cloudinary config ----
cloudinary.config(
    cloud_name="dtyjmwyrp",
    api_key="619824242791553",
    api_secret="l8hHU1GIg1FJ8rDgvHd4Sf7BWMk"
)

# ---- Load model once ----
GDRIVE_URL = "1ftzxTJUnlxpQFqPlaUozG_JUbl1Qi5tQ"
MODEL_CKPT_PATH = os.path.abspath("model_checkpoint.ckpt")
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

if not os.path.exists(MODEL_CKPT_PATH):
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_CKPT_PATH}. Please rebuild the Docker image to include the model.")
else:
    print(f"[INFO] Model checkpoint already exists at {MODEL_CKPT_PATH}, skipping download.")

model = Patchcore.load_from_checkpoint(MODEL_CKPT_PATH, **config.model.init_args)
model.eval()
model = model.to(device)
print("[INFO] Model loaded and ready for inference")


def infer_single_image_with_patchcore(image_path: str):
    """PatchCore inference on a single image"""
    fixed_path = os.path.abspath(os.path.normpath(image_path))
    orig_img = Image.open(fixed_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    img_resized = orig_img.resize((256, 256))
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if hasattr(output, "anomaly_map"):
            anomaly_map = output.anomaly_map.squeeze().detach().cpu().numpy()
        elif isinstance(output, (tuple, list)) and len(output) > 1:
            anomaly_map = output[1].squeeze().detach().cpu().numpy()
        else:
            anomaly_map = None

    base = os.path.splitext(os.path.basename(fixed_path))[0]
    mask_path = None
    filtered_path = None

    if anomaly_map is not None:
        norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
        if norm_map.ndim > 2:
            norm_map = np.squeeze(norm_map)
            if norm_map.ndim > 2:
                norm_map = norm_map[0]

        mask_img_256 = Image.fromarray(norm_map)
        mask_img = mask_img_256.resize((orig_w, orig_h), resample=Image.BILINEAR)

        mask_path = os.path.join(OUT_MASK_DIR, f"{base}_mask.png")
        mask_img.save(mask_path)

        bin_mask = np.array(mask_img) > 128
        orig_np = np.array(orig_img)
        filtered_np = np.zeros_like(orig_np)
        filtered_np[bin_mask] = orig_np[bin_mask]
        filtered_img = Image.fromarray(filtered_np)

        filtered_path = os.path.join(OUT_FILTERED_DIR, f"{base}_filtered.png")
        filtered_img.save(filtered_path)

        print(f"[PatchCore] Saved mask -> {mask_path}")
        print(f"[PatchCore] Saved filtered -> {filtered_path}")
    else:
        print("[PatchCore] No anomaly_map produced by model.")

    return {
        "orig_path": fixed_path,
        "mask_path": mask_path,
        "filtered_path": filtered_path,
        "orig_size": (orig_w, orig_h),
    }


def classify_filtered_image(filtered_img_path: str):
    """OpenCV heuristic classification on filtered image"""
    img = cv2.imread(filtered_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read filtered image: {filtered_img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Color masks
    blue_mask   = cv2.inRange(hsv, (90, 50, 20), (130, 255, 255))
    black_mask  = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    yellow_mask = cv2.inRange(hsv, (20, 130, 130), (35, 255, 255))
    orange_mask = cv2.inRange(hsv, (10, 100, 100), (25, 255, 255))
    red_mask1   = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2   = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_mask    = cv2.bitwise_or(red_mask1, red_mask2)

    total = img.shape[0] * img.shape[1]
    blue_count   = np.sum(blue_mask > 0)
    black_count  = np.sum(black_mask > 0)
    yellow_count = np.sum(yellow_mask > 0)
    orange_count = np.sum(orange_mask > 0)
    red_count    = np.sum(red_mask > 0)

    label = "Unknown"
    box_list, label_list = [], []

    # Simplified classification logic (keeping only essential parts)
    if (blue_count + black_count) / total > 0.8:
        label = "Normal"
    elif (red_count + orange_count + yellow_count) / total > 0.7:
        label = "Full Wire Overload"
        box_list.append((0, 0, img.shape[1], img.shape[0]))
        label_list.append(label)
    else:
        # Point overloads detection (simplified)
        min_area_faulty = 120
        max_area = 0.05 * total
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_faulty < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                box_list.append((x, y, w, h))
                label_list.append("Point Overload (Faulty)")

    return label, box_list, label_list, img


def run_pipeline_for_image(image_path: str):
    """Complete pipeline: PatchCore + classification + drawing"""
    # 1) PatchCore inference
    pc_out = infer_single_image_with_patchcore(image_path)
    filtered_path = pc_out["filtered_path"]
    orig_path = pc_out["orig_path"]

    if filtered_path is None:
        filtered_path = orig_path

    # 2) Classify
    label, boxes, labels, _filtered_bgr = classify_filtered_image(filtered_path)

    # 3) Draw boxes on original image
    draw_img = cv2.imread(orig_path)
    if draw_img is None:
        raise FileNotFoundError(f"Could not read original image: {orig_path}")

    for (x, y, w, h), l in zip(boxes, labels):
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(draw_img, l, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not boxes:
        cv2.putText(draw_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    base = os.path.splitext(os.path.basename(orig_path))[0]
    ext = os.path.splitext(os.path.basename(orig_path))[1]
    out_boxed_path = os.path.join(OUT_BOXED_DIR, f"{base}_boxed{ext if ext else '.png'}")
    ok = cv2.imwrite(out_boxed_path, draw_img)
    if not ok:
        out_boxed_path = os.path.join(OUT_BOXED_DIR, f"{base}_boxed.png")
        cv2.imwrite(out_boxed_path, draw_img)

    print(f"[Pipeline] Classification label: {label}")
    print(f"[Pipeline] Saved boxes-on-original -> {out_boxed_path}")
    
    return {
        "label": label,
        "boxed_path": out_boxed_path,
        "mask_path": pc_out["mask_path"],
        "filtered_path": pc_out["filtered_path"],
        "boxes": [
            {"box": [int(x), int(y), int(w), int(h)], "type": l}
            for (x, y, w, h), l in zip(boxes, labels)
        ]
    }


def download_image_from_url(url):
    """Download image from URL to temp file"""
    import requests
    import tempfile
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    for chunk in response.iter_content(1024):
        tmp.write(chunk)
    tmp.close()
    return tmp.name


def upload_to_cloudinary(file_path, folder=None):
    """Upload file to Cloudinary"""
    upload_opts = {"resource_type": "image"}
    if folder:
        upload_opts["folder"] = folder
    result = cloudinary.uploader.upload(file_path, **upload_opts)
    return result["secure_url"]

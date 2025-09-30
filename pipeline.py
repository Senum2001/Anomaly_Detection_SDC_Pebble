# pipeline.py
# Combines PatchCore single-image inference + OpenCV heuristic classification
# Usage: python pipeline.py <image_path>

import sys
import os
import cv2
import numpy as np
from PIL import Image
import torch
import tempfile
import requests
from flask import Flask, request, jsonify
import cloudinary
import cloudinary.uploader

# ---- Import your PatchCore API (must be available in PYTHONPATH) ----
from scripts.patchcore_api_inference import Patchcore, config, device

# ---- Output directories (pipeline-scoped) ----
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
MODEL_CKPT = "results/Patchcore/transformers/v7/weights/lightning/model.ckpt"
model = Patchcore.load_from_checkpoint(MODEL_CKPT, **config.model.init_args)
model.eval()
model = model.to(device)

# ---- Flask app ----
app = Flask(__name__)


# =========================
# Step 1: PatchCore Inference
# =========================
def infer_single_image_with_patchcore(image_path: str):
    """
    Runs PatchCore on a single image, saves:
      - grayscale mask (uint8, 0..255) to OUT_MASK_DIR
      - filtered image (pixels kept where mask>128) to OUT_FILTERED_DIR (same size as original)
    Returns:
      dict with keys:
        'orig_path', 'mask_path', 'filtered_path',
        'orig_size' = (W, H)
    """
    fixed_path = os.path.abspath(os.path.normpath(image_path))
    orig_img = Image.open(fixed_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Minimal resize + tensorize (adjust if your training used a different preprocessing)
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
        # Normalize to 0..255
        norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
        if norm_map.ndim > 2:
            norm_map = np.squeeze(norm_map)
            if norm_map.ndim > 2:
                norm_map = norm_map[0]

        mask_img_256 = Image.fromarray(norm_map)
        # Resize mask to original size
        mask_img = mask_img_256.resize((orig_w, orig_h), resample=Image.BILINEAR)

        mask_path = os.path.join(OUT_MASK_DIR, f"{base}_mask.png")
        mask_img.save(mask_path)

        # Create filtered image (keep original pixels where mask>128)
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


# =========================
# Step 2: OpenCV Heuristic Classification (single image)
# =========================

# IOU for NMS
def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def _merge_close_boxes(boxes, labels, dist_thresh=20):
    merged, merged_labels = [], []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        label1 = labels[i]
        x2, y2, w2, h2 = x1, y1, w1, h1
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bx, by, bw, bh = boxes[j]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = bx + bw // 2, by + bh // 2
            if abs(cx1 - cx2) < dist_thresh and abs(cy1 - cy2) < dist_thresh and label1 == labels[j]:
                x2 = min(x2, bx)
                y2 = min(y2, by)
                w2 = max(x1 + w1, bx + bw) - x2
                h2 = max(y1 + h1, by + bh) - y2
                used[j] = True
        merged.append((x2, y2, w2, h2))
        merged_labels.append(label1)
        used[i] = True
    return merged, merged_labels

def _nms_iou(boxes, labels, iou_thresh=0.4):
    if len(boxes) == 0:
        return [], []
    idxs = np.argsort([w * h for (x, y, w, h) in boxes])[::-1]
    keep, keep_labels = [], []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(boxes[i])
        keep_labels.append(labels[i])
        remove = [0]
        for j in range(1, len(idxs)):
            if _iou(boxes[i], boxes[idxs[j]]) > iou_thresh:
                remove.append(j)
        idxs = np.delete(idxs, remove)
    return keep, keep_labels

def _filter_faulty_inside_potential(boxes, labels):
    filtered_boxes, filtered_labels = [], []
    for (box, label) in zip(boxes, labels):
        if label == "Point Overload (Potential)":
            keep = True
            x, y, w, h = box
            for (fbox, flabel) in zip(boxes, labels):
                if flabel == "Point Overload (Faulty)":
                    fx, fy, fw, fh = fbox
                    if fx >= x and fy >= y and fx + fw <= x + w and fy + fh <= y + h:
                        keep = False
                        break
            if keep:
                filtered_boxes.append(box)
                filtered_labels.append(label)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
    return filtered_boxes, filtered_labels

def _filter_faulty_overlapping_potential(boxes, labels):
    def is_overlapping(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        return (xB > xA) and (yB > yA)

    filtered_boxes, filtered_labels = [], []
    for (box, label) in zip(boxes, labels):
        if label == "Point Overload (Potential)":
            keep = True
            for (fbox, flabel) in zip(boxes, labels):
                if flabel == "Point Overload (Faulty)" and is_overlapping(box, fbox):
                    keep = False
                    break
            if keep:
                filtered_boxes.append(box)
                filtered_labels.append(label)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
    return filtered_boxes, filtered_labels

def classify_filtered_image(filtered_img_path: str):
    """
    Runs the heuristic color-based classification on the FILTERED image.
    Returns:
      label: str
      box_list: [(x, y, w, h), ...]
      label_list: [str, ...]
      img_bgr: the filtered image as BGR (for dimensions if needed)
    """
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

    # Full image checks
    if (blue_count + black_count) / total > 0.8:
        label = "Normal"
    elif (red_count + orange_count) / total > 0.5:
        label = "Full Wire Overload"
    elif (yellow_count) / total > 0.5:
        label = "Full Wire Overload"

    # Check for full wire overload (dominant warm colors)
    full_wire_thresh = 0.7
    if (red_count + orange_count + yellow_count) / total > full_wire_thresh:
        label = "Full Wire Overload"
        box_list.append((0, 0, img.shape[1], img.shape[0]))
        label_list.append(label)
    else:
        # Point overloads (areas + thresholds)
        min_area_faulty = 120
        min_area_potential = 1000
        max_area = 0.05 * total

        for mask, spot_label, min_a in [
            (red_mask, "Point Overload (Faulty)", min_area_faulty),
            (yellow_mask, "Point Overload (Potential)", min_area_potential),
        ]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_a < area < max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    box_list.append((x, y, w, h))
                    label_list.append(spot_label)

        # Middle area checks
        h, w = img.shape[:2]
        center = img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        center_yellow = cv2.inRange(center_hsv, (20, 130, 130), (35, 255, 255))
        center_orange = cv2.inRange(center_hsv, (10, 100, 100), (25, 255, 255))
        center_red1 = cv2.inRange(center_hsv, (0, 100, 100), (10, 255, 255))
        center_red2 = cv2.inRange(center_hsv, (160, 100, 100), (180, 255, 255))
        center_red = cv2.bitwise_or(center_red1, center_red2)

        if np.sum(center_red > 0) + np.sum(center_orange > 0) > 0.1 * center.size:
            label = "Loose Joint (Faulty)"
            box_list.append((w // 4, h // 4, w // 2, h // 2))
            label_list.append(label)
        elif np.sum(center_yellow > 0) > 0.1 * center.size:
            label = "Loose Joint (Potential)"
            box_list.append((w // 4, h // 4, w // 2, h // 2))
            label_list.append(label)

    # Tiny spots (always check)
    min_area_tiny, max_area_tiny = 10, 30
    for mask, spot_label in [
        (red_mask, "Tiny Faulty Spot"),
        (yellow_mask, "Tiny Potential Spot"),
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_tiny < area < max_area_tiny:
                x, y, w, h = cv2.boundingRect(cnt)
                box_list.append((x, y, w, h))
                label_list.append(spot_label)

    # Detect wire-shaped (long/thin) warm regions
    aspect_ratio_thresh = 5
    min_strip_area = 0.01 * total
    wire_boxes, wire_labels = [], []
    for mask, strip_label in [
        (red_mask, "Wire Overload (Red Strip)"),
        (yellow_mask, "Wire Overload (Yellow Strip)"),
        (orange_mask, "Wire Overload (Orange Strip)"),
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_strip_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                if aspect_ratio > aspect_ratio_thresh:
                    wire_boxes.append((x, y, w, h))
                    wire_labels.append(strip_label)

    # Overwrite with wire boxes first
    box_list = wire_boxes[:] + box_list
    label_list = wire_labels[:] + label_list

    # Final pruning/merging
    box_list, label_list = _nms_iou(box_list, label_list, iou_thresh=0.4)
    box_list, label_list = _filter_faulty_inside_potential(box_list, label_list)
    box_list, label_list = _filter_faulty_overlapping_potential(box_list, label_list)
    box_list, label_list = _merge_close_boxes(box_list, label_list, dist_thresh=100)

    return label, box_list, label_list, img


# =========================
# Step 3: Orchestration (single image)
# =========================
def run_pipeline_for_image(image_path: str):
    # 1) PatchCore -> mask + filtered
    pc_out = infer_single_image_with_patchcore(image_path)
    filtered_path = pc_out["filtered_path"]
    orig_path = pc_out["orig_path"]

    if filtered_path is None:
        # No mask produced; fall back to using original image for classification
        print("[Pipeline] No filtered image available. Running classifier on the original image.")
        filtered_path = orig_path

    # 2) Classify the FILTERED image
    label, boxes, labels, _filtered_bgr = classify_filtered_image(filtered_path)

    # 3) Draw boxes & labels ON THE ORIGINAL INPUT IMAGE
    draw_img = cv2.imread(orig_path)
    if draw_img is None:
        raise FileNotFoundError(f"Could not read original image for drawing: {orig_path}")

    for (x, y, w, h), l in zip(boxes, labels):
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(draw_img, l, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not boxes:
        # Still annotate the overall label
        cv2.putText(draw_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    base = os.path.splitext(os.path.basename(orig_path))[0]
    ext = os.path.splitext(os.path.basename(orig_path))[1]
    out_boxed_path = os.path.join(OUT_BOXED_DIR, f"{base}_boxed{ext if ext else '.png'}")
    ok = cv2.imwrite(out_boxed_path, draw_img)
    if not ok:
        # fallback to .png
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


# =========================
# Flask API endpoint
# =========================
def download_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    for chunk in response.iter_content(1024):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

def upload_to_cloudinary(file_path, folder=None):
    upload_opts = {"resource_type": "image"}
    if folder:
        upload_opts["folder"] = folder
    result = cloudinary.uploader.upload(file_path, **upload_opts)
    return result["secure_url"]

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    image_url = data.get("image_url")
    if not image_url:
        return jsonify({"error": "Missing image_url"}), 400
    try:
        # 1. Download image
        local_path = download_image_from_url(image_url)
        # 2. Run pipeline
        results = run_pipeline_for_image(local_path)
        # 3. Upload outputs to Cloudinary
        boxed_url = upload_to_cloudinary(results["boxed_path"], folder="pipeline_outputs") if results["boxed_path"] else None
        mask_url = upload_to_cloudinary(results["mask_path"], folder="pipeline_outputs") if results["mask_path"] else None
        filtered_url = upload_to_cloudinary(results["filtered_path"], folder="pipeline_outputs") if results["filtered_path"] else None
        # 4. Build response
        response = {
            "label": results["label"],
            "boxed_url": boxed_url,
            "mask_url": mask_url,
            "filtered_url": filtered_url,
            "boxes": results.get("boxes", [])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PatchCore pipeline service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args, unknown = parser.parse_known_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

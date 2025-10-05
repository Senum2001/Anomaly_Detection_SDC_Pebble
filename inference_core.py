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
import requests
import tempfile
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


# Helper functions for classification
def _iou(boxA, boxB):
    """Calculate Intersection over Union"""
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


def _merge_close_boxes(boxes, labels, dist_thresh=20, confidences=None):
    """Merge boxes that are close to each other, maintaining confidence alignment"""
    if confidences is None:
        confidences = [0.5] * len(boxes)
    
    merged, merged_labels, merged_confidences = [], [], []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        label1 = labels[i]
        conf1 = confidences[i]
        x2, y2, w2, h2 = x1, y1, w1, h1
        max_conf = conf1
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
                max_conf = max(max_conf, confidences[j])
                used[j] = True
        merged.append((x2, y2, w2, h2))
        merged_labels.append(label1)
        merged_confidences.append(max_conf)
        used[i] = True
    return merged, merged_labels, merged_confidences


def _nms_iou(boxes, labels, iou_thresh=0.4):
    """Non-Maximum Suppression based on IOU"""
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


def _nms_iou_with_confidence(boxes, labels, confidences, iou_thresh=0.4):
    """Non-maximum suppression using IOU, keeping confidence aligned"""
    if len(boxes) == 0:
        return [], [], []
    idxs = np.argsort([w * h for (x, y, w, h) in boxes])[::-1]
    keep, keep_labels, keep_confidences = [], [], []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(boxes[i])
        keep_labels.append(labels[i])
        keep_confidences.append(confidences[i])
        remove = [0]
        for j in range(1, len(idxs)):
            if _iou(boxes[i], boxes[idxs[j]]) > iou_thresh:
                remove.append(j)
        idxs = np.delete(idxs, remove)
    return keep, keep_labels, keep_confidences


def _filter_faulty_inside_potential(boxes, labels, confidences=None):
    """Remove potential boxes that contain faulty boxes, maintaining confidence alignment"""
    if confidences is None:
        confidences = [0.5] * len(boxes)
    
    filtered_boxes, filtered_labels, filtered_confidences = [], [], []
    for (box, label, conf) in zip(boxes, labels, confidences):
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
                filtered_confidences.append(conf)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_confidences.append(conf)
    return filtered_boxes, filtered_labels, filtered_confidences


def _filter_faulty_overlapping_potential(boxes, labels, confidences=None):
    """Remove potential boxes that overlap with faulty boxes, maintaining confidence alignment"""
    if confidences is None:
        confidences = [0.5] * len(boxes)
    
    def is_overlapping(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        return (xB > xA) and (yB > yA)

    filtered_boxes, filtered_labels, filtered_confidences = [], [], []
    for (box, label, conf) in zip(boxes, labels, confidences):
        if label == "Point Overload (Potential)":
            keep = True
            for (fbox, flabel) in zip(boxes, labels):
                if flabel == "Point Overload (Faulty)" and is_overlapping(box, fbox):
                    keep = False
                    break
            if keep:
                filtered_boxes.append(box)
                filtered_labels.append(label)
                filtered_confidences.append(conf)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_confidences.append(conf)
    return filtered_boxes, filtered_labels, filtered_confidences


def _calculate_confidence(img, box, mask, label):
    """
    Calculate confidence score for a detection based on:
    - Color intensity within the bounding box
    - Coverage ratio (how much of the box contains the target color)
    - Size relative to image
    """
    x, y, w, h = box
    
    # Extract region of interest
    roi = img[y:y+h, x:x+w]
    mask_roi = mask[y:y+h, x:x+w]
    
    if roi.size == 0 or mask_roi.size == 0:
        return 0.5
    
    # Calculate coverage (what % of the box has the target color)
    coverage = np.sum(mask_roi > 0) / mask_roi.size
    
    # Calculate intensity (average value in the detected region)
    if np.sum(mask_roi > 0) > 0:
        intensity = np.mean(roi[mask_roi > 0]) / 255.0
    else:
        intensity = 0.0
    
    # Calculate relative size (boxes that are too small or too large are less confident)
    total_pixels = img.shape[0] * img.shape[1]
    box_size = w * h
    size_ratio = box_size / total_pixels
    
    # Size confidence: optimal between 0.001 and 0.05 of image
    if size_ratio < 0.0001:
        size_conf = size_ratio / 0.0001  # Very small
    elif size_ratio > 0.1:
        size_conf = max(0.3, 1.0 - (size_ratio - 0.1) / 0.9)  # Very large
    else:
        size_conf = 1.0  # Good size
    
    # Label-specific confidence adjustments
    if "Faulty" in label:
        base_conf = 0.7  # Higher base for faulty (red is more definitive)
    elif "Potential" in label:
        base_conf = 0.6  # Lower base for potential (yellow is warning)
    elif "Tiny" in label:
        base_conf = 0.5  # Lower for tiny spots
    elif "Wire" in label or "Full" in label:
        base_conf = 0.8  # High for large patterns
    elif "Loose Joint" in label:
        base_conf = 0.7  # Moderate for center detections
    else:
        base_conf = 0.6
    
    # Weighted combination
    confidence = (
        base_conf * 0.4 +
        coverage * 0.35 +
        intensity * 0.15 +
        size_conf * 0.10
    )
    
    # Clamp to [0.3, 0.99] range
    confidence = max(0.3, min(0.99, confidence))
    
    return round(confidence, 3)


def _get_severity_color(label, confidence):
    """
    Get color-coded bounding box color based on severity and confidence.
    Returns BGR color tuple for OpenCV.
    
    Severity levels:
    - CRITICAL (Red spectrum): Faulty detections
    - WARNING (Yellow/Orange spectrum): Potential detections
    - INFO (Green/Blue spectrum): Normal/Informational
    
    Color intensity increases with confidence (0-1):
    - Low confidence (0.3-0.5): Lighter, more transparent colors
    - Medium confidence (0.5-0.7): Medium intensity
    - High confidence (0.7-1.0): Bright, vivid colors
    """
    # Normalize confidence to 0-1 range for color interpolation
    conf_normalized = max(0.0, min(1.0, confidence))
    
    # Determine severity level based on label
    if "Faulty" in label or "Full Wire Overload" in label:
        # CRITICAL - Red spectrum
        # Low conf: Light red/pink -> High conf: Bright red
        severity = "CRITICAL"
        if conf_normalized < 0.5:
            # Light red/pink (BGR)
            intensity = int(100 + (conf_normalized / 0.5) * 155)  # 100-255
            color = (100, 100, intensity)  # Light red
        elif conf_normalized < 0.7:
            # Medium red (BGR)
            intensity = int(150 + ((conf_normalized - 0.5) / 0.2) * 105)  # 150-255
            color = (50, 50, intensity)  # Medium red
        else:
            # Bright red/dark red (BGR)
            intensity = int(200 + ((conf_normalized - 0.7) / 0.3) * 55)  # 200-255
            color = (0, 0, intensity)  # Bright red
            
    elif "Potential" in label:
        # WARNING - Yellow/Orange spectrum
        # Low conf: Light yellow -> High conf: Bright orange/yellow
        severity = "WARNING"
        if conf_normalized < 0.5:
            # Light yellow (BGR)
            intensity = int(150 + (conf_normalized / 0.5) * 105)  # 150-255
            color = (intensity, intensity, 50)  # Light yellow
        elif conf_normalized < 0.7:
            # Medium yellow-orange (BGR)
            b_val = int(50 + ((conf_normalized - 0.5) / 0.2) * 50)  # 50-100
            g_val = int(200 + ((conf_normalized - 0.5) / 0.2) * 55)  # 200-255
            color = (b_val, g_val, 255)  # Yellow-orange
        else:
            # Bright orange (BGR)
            b_val = int(0 + ((conf_normalized - 0.7) / 0.3) * 50)  # 0-50
            color = (b_val, 165, 255)  # Bright orange
            
    elif "Wire" in label:
        # HIGH SEVERITY - Deep red/magenta for wire overload
        severity = "HIGH"
        if conf_normalized < 0.5:
            intensity = int(150 + (conf_normalized / 0.5) * 105)
            color = (intensity // 2, 0, intensity)  # Pink-magenta
        else:
            intensity = int(200 + ((conf_normalized - 0.5) / 0.5) * 55)
            color = (intensity // 3, 0, intensity)  # Deep red-magenta
            
    elif "Tiny" in label:
        # MINOR - Purple spectrum (still concerning but smaller)
        severity = "MINOR"
        if conf_normalized < 0.5:
            intensity = int(150 + (conf_normalized / 0.5) * 105)
            color = (intensity, 50, intensity - 50)  # Light purple
        else:
            intensity = int(200 + ((conf_normalized - 0.5) / 0.5) * 55)
            color = (intensity, 0, intensity - 50)  # Bright purple
            
    elif "Normal" in label:
        # INFO - Green spectrum
        severity = "INFO"
        intensity = int(100 + conf_normalized * 155)  # 100-255
        color = (50, intensity, 50)  # Green
    else:
        # UNKNOWN - Blue spectrum
        severity = "UNKNOWN"
        intensity = int(100 + conf_normalized * 155)
        color = (intensity, 100, 50)  # Blue
    
    return color, severity


def classify_filtered_image(filtered_img_path: str):
    """
    Runs the heuristic color-based classification on the FILTERED image.
    Returns:
      label: str
      box_list: [(x, y, w, h), ...]
      label_list: [str, ...]
      confidence_list: [float, ...] - confidence scores (0-1) for each box
      img_bgr: the filtered image as BGR
    """
    img = cv2.imread(filtered_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read filtered image: {filtered_img_path}")

    # Ensure consistent color space
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
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

    # Debug logging
    print(f"[Classification] Image shape: {img.shape}")
    print(f"[Classification] Color counts - Blue: {blue_count}, Black: {black_count}, "
          f"Yellow: {yellow_count}, Orange: {orange_count}, Red: {red_count}")

    label = "Unknown"
    box_list, label_list, confidence_list = [], [], []

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
        box = (0, 0, img.shape[1], img.shape[0])
        box_list.append(box)
        label_list.append(label)
        # Full image detection - high confidence based on color coverage
        conf = min(0.95, 0.7 + ((red_count + orange_count + yellow_count) / total - full_wire_thresh) * 0.8)
        confidence_list.append(round(conf, 3))
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
                    box = (x, y, w, h)
                    box_list.append(box)
                    label_list.append(spot_label)
                    confidence_list.append(_calculate_confidence(img, box, mask, spot_label))

        # Middle area checks (Loose Joint detection)
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
            box = (w // 4, h // 4, w // 2, h // 2)
            box_list.append(box)
            label_list.append(label)
            center_coverage = (np.sum(center_red > 0) + np.sum(center_orange > 0)) / center.size
            confidence_list.append(round(min(0.85, 0.6 + center_coverage), 3))
        elif np.sum(center_yellow > 0) > 0.1 * center.size:
            label = "Loose Joint (Potential)"
            box = (w // 4, h // 4, w // 2, h // 2)
            box_list.append(box)
            label_list.append(label)
            center_coverage = np.sum(center_yellow > 0) / center.size
            confidence_list.append(round(min(0.75, 0.5 + center_coverage), 3))

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
                box = (x, y, w, h)
                box_list.append(box)
                label_list.append(spot_label)
                confidence_list.append(_calculate_confidence(img, box, mask, spot_label))

    # Detect wire-shaped (long/thin) warm regions
    aspect_ratio_thresh = 5
    min_strip_area = 0.01 * total
    wire_boxes, wire_labels, wire_confidences = [], [], []
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
                    box = (x, y, w, h)
                    wire_boxes.append(box)
                    wire_labels.append(strip_label)
                    wire_confidences.append(_calculate_confidence(img, box, mask, strip_label))

    # Prioritize wire boxes first
    box_list = wire_boxes[:] + box_list
    label_list = wire_labels[:] + label_list
    confidence_list = wire_confidences[:] + confidence_list

    # Final pruning/merging - need to keep confidence aligned
    box_list, label_list, confidence_list = _nms_iou_with_confidence(box_list, label_list, confidence_list, iou_thresh=0.4)
    box_list, label_list, confidence_list = _filter_faulty_inside_potential(box_list, label_list, confidence_list)
    box_list, label_list, confidence_list = _filter_faulty_overlapping_potential(box_list, label_list, confidence_list)
    box_list, label_list, confidence_list = _merge_close_boxes(box_list, label_list, dist_thresh=100, confidences=confidence_list)

    print(f"[Classification] Final label: {label}, Boxes found: {len(box_list)}")
    return label, box_list, label_list, confidence_list, img


def run_pipeline_for_image(image_path: str):
    """Complete pipeline: PatchCore + classification + drawing"""
    # 1) PatchCore inference
    pc_out = infer_single_image_with_patchcore(image_path)
    filtered_path = pc_out["filtered_path"]
    orig_path = pc_out["orig_path"]

    if filtered_path is None:
        filtered_path = orig_path

    # 2) Classify (now returns confidence_list as well)
    label, boxes, labels, confidences, _filtered_bgr = classify_filtered_image(filtered_path)

    # 3) Draw boxes on original image with severity-based colors
    draw_img = cv2.imread(orig_path)
    if draw_img is None:
        raise FileNotFoundError(f"Could not read original image: {orig_path}")

    for (x, y, w, h), l, conf in zip(boxes, labels, confidences):
        # Get color based on severity and confidence
        box_color, severity = _get_severity_color(l, conf)
        
        # Draw bounding box with severity color (thicker for higher confidence)
        thickness = 2 if conf < 0.7 else 3
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), box_color, thickness)
        
        # Prepare text with severity indicator
        text = f"{l} ({conf:.2f})"
        severity_badge = f"[{severity}]"
        
        # Calculate text sizes
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        (badge_w, badge_h), _ = cv2.getTextSize(severity_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        # Draw semi-transparent background for text
        bg_y_start = max(0, y - text_h - badge_h - 8)
        bg_y_end = max(text_h + badge_h + 8, y - 2)
        cv2.rectangle(draw_img, (x, bg_y_start), (x + max(text_w, badge_w) + 10, bg_y_end), (0, 0, 0), -1)
        
        # Draw severity badge in severity color
        cv2.putText(draw_img, severity_badge, (x + 2, max(badge_h + 2, y - text_h - 4)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1, cv2.LINE_AA)
        
        # Draw label and confidence in white
        cv2.putText(draw_img, text, (x + 2, max(text_h + badge_h + 4, y - 2)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add corner markers for high confidence detections (conf > 0.8)
        if conf > 0.8:
            marker_size = 8
            # Top-left corner
            cv2.line(draw_img, (x, y), (x + marker_size, y), box_color, 3)
            cv2.line(draw_img, (x, y), (x, y + marker_size), box_color, 3)
            # Top-right corner
            cv2.line(draw_img, (x + w, y), (x + w - marker_size, y), box_color, 3)
            cv2.line(draw_img, (x + w, y), (x + w, y + marker_size), box_color, 3)
            # Bottom-left corner
            cv2.line(draw_img, (x, y + h), (x + marker_size, y + h), box_color, 3)
            cv2.line(draw_img, (x, y + h), (x, y + h - marker_size), box_color, 3)
            # Bottom-right corner
            cv2.line(draw_img, (x + w, y + h), (x + w - marker_size, y + h), box_color, 3)
            cv2.line(draw_img, (x + w, y + h), (x + w, y + h - marker_size), box_color, 3)

    if not boxes:
        # If no boxes, show overall label in green (normal)
        cv2.putText(draw_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    base = os.path.splitext(os.path.basename(orig_path))[0]
    ext = os.path.splitext(os.path.basename(orig_path))[1]
    out_boxed_path = os.path.join(OUT_BOXED_DIR, f"{base}_boxed{ext if ext else '.png'}")
    ok = cv2.imwrite(out_boxed_path, draw_img)
    if not ok:
        out_boxed_path = os.path.join(OUT_BOXED_DIR, f"{base}_boxed.png")
        cv2.imwrite(out_boxed_path, draw_img)

    print(f"[Pipeline] Classification label: {label}")
    print(f"[Pipeline] Saved boxes-on-original -> {out_boxed_path}")
    
    # Build boxes array with severity and color information
    boxes_output = []
    for (x, y, w, h), l, conf in zip(boxes, labels, confidences):
        box_color, severity = _get_severity_color(l, conf)
        boxes_output.append({
            "box": [int(x), int(y), int(w), int(h)],
            "type": l,
            "confidence": float(conf),
            "severity": severity,
            "color": {
                "bgr": [int(box_color[0]), int(box_color[1]), int(box_color[2])],
                "rgb": [int(box_color[2]), int(box_color[1]), int(box_color[0])],
                "hex": "#{:02x}{:02x}{:02x}".format(int(box_color[2]), int(box_color[1]), int(box_color[0]))
            }
        })
    
    return {
        "label": label,
        "boxed_path": out_boxed_path,
        "mask_path": pc_out["mask_path"],
        "filtered_path": pc_out["filtered_path"],
        "boxes": boxes_output
    }


def download_image_from_url(url):
    """Download image from URL to temp file"""
    import requests
    import tempfile
    from urllib.parse import urlparse
    import mimetypes
    
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {url}")
    
    # Determine file extension from URL or Content-Type
    content_type = response.headers.get('content-type', '')
    if 'image/png' in content_type:
        suffix = '.png'
    elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
        suffix = '.jpg'
    else:
        # Try to get extension from URL
        parsed_url = urlparse(url)
        path = parsed_url.path
        ext = os.path.splitext(path)[1]
        suffix = ext if ext in ['.jpg', '.jpeg', '.png', '.bmp'] else '.jpg'
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
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

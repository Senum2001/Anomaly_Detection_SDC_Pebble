import cv2
import numpy as np
import os

# Directory containing filtered images
dir_path = 'api_inference_filtered_pipeline'
output_dir = 'api_inference_labeled_boxes_pipeline'
os.makedirs(output_dir, exist_ok=True)

# IOU function for non-max suppression
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Merge close bounding boxes (same label, centers within dist_thresh)
def merge_close_boxes(boxes, labels, dist_thresh=20):
    merged = []
    merged_labels = []
    used = [False]*len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        label1 = labels[i]
        x2, y2, w2, h2 = x1, y1, w1, h1
        for j in range(i+1, len(boxes)):
            if used[j]:
                continue
            bx, by, bw, bh = boxes[j]
            # If boxes are close (distance between centers < dist_thresh)
            cx1, cy1 = x1 + w1//2, y1 + h1//2
            cx2, cy2 = bx + bw//2, by + bh//2
            if abs(cx1-cx2) < dist_thresh and abs(cy1-cy2) < dist_thresh and label1 == labels[j]:
                # Merge boxes
                x2 = min(x2, bx)
                y2 = min(y2, by)
                w2 = max(x1+w1, bx+bw) - x2
                h2 = max(y1+h1, by+bh) - y2
                used[j] = True
        merged.append((x2, y2, w2, h2))
        merged_labels.append(label1)
        used[i] = True
    return merged, merged_labels

# Non-max suppression using IOU
def non_max_suppression_iou(boxes, labels, iou_thresh=0.4):
    if len(boxes) == 0:
        return [], []
    idxs = np.argsort([w*h for (x, y, w, h) in boxes])[::-1]
    keep = []
    keep_labels = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(boxes[i])
        keep_labels.append(labels[i])
        remove = [0]
        for j in range(1, len(idxs)):
            if iou(boxes[i], boxes[idxs[j]]) > iou_thresh:
                remove.append(j)
        idxs = np.delete(idxs, remove)
    return keep, keep_labels

# Filter out potential boxes that contain a faulty box inside
def filter_faulty_inside_potential(boxes, labels):
    filtered_boxes = []
    filtered_labels = []
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if label == 'Point Overload (Potential)':
            # Check if any faulty box is inside this potential box
            keep = True
            for j, (fbox, flabel) in enumerate(zip(boxes, labels)):
                if flabel == 'Point Overload (Faulty)':
                    # Check if faulty box is inside potential box
                    x, y, w, h = box
                    fx, fy, fw, fh = fbox
                    if fx >= x and fy >= y and fx+fw <= x+w and fy+fh <= y+h:
                        keep = False
                        break
            if keep:
                filtered_boxes.append(box)
                filtered_labels.append(label)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
    return filtered_boxes, filtered_labels

# Remove potential boxes that overlap with a faulty box (not just inside)
def filter_faulty_overlapping_potential(boxes, labels):
    # Remove potential boxes that overlap at all with a faulty box (any intersection)
    filtered_boxes = []
    filtered_labels = []
    def is_overlapping(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        return (xB > xA) and (yB > yA)
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if label == 'Point Overload (Potential)':
            keep = True
            for j, (fbox, flabel) in enumerate(zip(boxes, labels)):
                if flabel == 'Point Overload (Faulty)':
                    if is_overlapping(box, fbox):
                        keep = False
                        break
            if keep:
                filtered_boxes.append(box)
                filtered_labels.append(label)
        else:
            filtered_boxes.append(box)
            filtered_labels.append(label)
    return filtered_boxes, filtered_labels

# Heuristic classification function
def classify_image(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color masks
    blue_mask = cv2.inRange(hsv, (90, 50, 20), (130, 255, 255))
    black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
    yellow_mask = cv2.inRange(hsv, (20, 130, 130), (35, 255, 255))  # increased threshold
    orange_mask = cv2.inRange(hsv, (10, 100, 100), (25, 255, 255))
    red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    total = img.shape[0] * img.shape[1]
    blue_count = np.sum(blue_mask > 0)
    black_count = np.sum(black_mask > 0)
    yellow_count = np.sum(yellow_mask > 0)
    orange_count = np.sum(orange_mask > 0)
    red_count = np.sum(red_mask > 0)

    label = 'Unknown'
    box_list = []
    label_list = []

    # Full image checks
    if (blue_count + black_count) / total > 0.8:
        label = 'Normal'
    elif (red_count + orange_count) / total > 0.5:
        label = 'Full Wire Overload'
    elif (yellow_count) / total > 0.5:
        label = 'Full Wire Overload'
    # Check for full wire overload (entire image reddish or yellowish)
    full_wire_thresh = 0.7  # 70% of image is reddish or yellowish
    if (red_count + orange_count + yellow_count) / total > full_wire_thresh:
        label = 'Full Wire Overload'
        # Add a box covering the whole image
        box_list.append((0, 0, img.shape[1], img.shape[0]))
        label_list.append(label)
    else:
        # Small spot checks (improved: filter tiny spots, merge overlapping boxes)
        min_area_faulty = 120  # increased min area for red/orange (faulty)
        min_area_potential = 1000  # much higher min area for yellow (potential)
        max_area = 0.05 * total
        # Faulty (red/orange) spots
        for mask, spot_label, min_a in [
            (red_mask, 'Point Overload (Faulty)', min_area_faulty),
            (yellow_mask, 'Point Overload (Potential)', min_area_potential)
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
        center = img[h//4:3*h//4, w//4:3*w//4]
        center_hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        center_yellow = cv2.inRange(center_hsv, (20, 130, 130), (35, 255, 255))
        center_orange = cv2.inRange(center_hsv, (10, 100, 100), (25, 255, 255))
        center_red1 = cv2.inRange(center_hsv, (0, 100, 100), (10, 255, 255))
        center_red2 = cv2.inRange(center_hsv, (160, 100, 100), (180, 255, 255))
        center_red = cv2.bitwise_or(center_red1, center_red2)
        if np.sum(center_red > 0) + np.sum(center_orange > 0) > 0.1 * center.size:
            label = 'Loose Joint (Faulty)'
            box_list.append((w//4, h//4, w//2, h//2))
            label_list.append(label)
        elif np.sum(center_yellow > 0) > 0.1 * center.size:
            label = 'Loose Joint (Potential)'
            box_list.append((w//4, h//4, w//2, h//2))
            label_list.append(label)
    # Always check for tiny spots, even if image is labeled as Normal
    min_area_tiny = 10
    max_area_tiny = 30
    for mask, spot_label in [
        (red_mask, 'Tiny Faulty Spot'),
        (yellow_mask, 'Tiny Potential Spot')
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area_tiny < area < max_area_tiny:
                x, y, w, h = cv2.boundingRect(cnt)
                box_list.append((x, y, w, h))
                label_list.append(spot_label)
    # Detect wire-shaped (long, thin) regions for wire overloads only
    aspect_ratio_thresh = 5
    min_strip_area = 0.01 * total
    wire_boxes = []
    wire_labels = []
    for mask, strip_label in [
        (red_mask, 'Wire Overload (Red Strip)'),
        (yellow_mask, 'Wire Overload (Yellow Strip)'),
        (orange_mask, 'Wire Overload (Orange Strip)')
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
    # Add wire overloads to box_list/label_list
    box_list = wire_boxes[:]
    label_list = wire_labels[:]
    # For point overloads, do not require wire shape
    min_area_faulty = 120
    min_area_potential = 1000
    max_area = 0.05 * total
    for mask, spot_label, min_a in [
        (red_mask, 'Point Overload (Faulty)', min_area_faulty),
        (yellow_mask, 'Point Overload (Potential)', min_area_potential)
    ]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_a < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                box_list.append((x, y, w, h))
                label_list.append(spot_label)
    # Remove overlapping boxes using IOU
    box_list, label_list = non_max_suppression_iou(box_list, label_list, iou_thresh=0.4)
    box_list, label_list = filter_faulty_inside_potential(box_list, label_list)
    box_list, label_list = filter_faulty_overlapping_potential(box_list, label_list)
    box_list, label_list = merge_close_boxes(box_list, label_list, dist_thresh=100)
    return label, box_list, label_list, img

# Batch process all images in the directory
for fname in os.listdir(dir_path):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    label, box_list, label_list, img = classify_image(os.path.join(dir_path, fname))
    # Load the original (unfiltered) image for drawing boxes
    orig_dir = 'api_inference_pred_masks'  # or the directory with original images
    orig_img_path = os.path.join(orig_dir, fname)
    if os.path.exists(orig_img_path):
        draw_img = cv2.imread(orig_img_path)
        if draw_img is None:
            draw_img = img.copy()
    else:
        draw_img = img.copy()
    # Draw bounding boxes and labels on the original image
    for (x, y, w, h), l in zip(box_list, label_list):
        cv2.rectangle(draw_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(draw_img, l, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if not box_list:
        cv2.putText(draw_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, draw_img)
    print(f"{fname}: {label} (saved with boxes on original image)")

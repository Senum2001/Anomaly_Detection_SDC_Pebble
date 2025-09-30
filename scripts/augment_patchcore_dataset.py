import os
from PIL import Image, ImageOps, ImageEnhance
import random

# Directory containing normal images for training
NORMAL_DIR = 'dataset/train/normal'
TARGET_COUNT = 100

# List all images in the directory
images = [f for f in os.listdir(NORMAL_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
current_count = len(images)
print(f"[INFO] Found {current_count} images in {NORMAL_DIR}")

if current_count == 0:
    print("[ERROR] No images found to augment.")
    exit(1)

# Augmentation functions
AUGS = [
    lambda img: img.rotate(random.randint(-30, 30)),
    lambda img: ImageOps.mirror(img),
    lambda img: ImageOps.flip(img),
    lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
    lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
    lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
]

aug_idx = 0
while current_count < TARGET_COUNT:
    for fname in images:
        if current_count >= TARGET_COUNT:
            break
        img_path = os.path.join(NORMAL_DIR, fname)
        img = Image.open(img_path).convert('RGB')
        aug = random.choice(AUGS)
        aug_img = aug(img)
        aug_fname = f"aug_{aug_idx}_{fname}"
        aug_img.save(os.path.join(NORMAL_DIR, aug_fname))
        aug_idx += 1
        current_count += 1
        print(f"[AUG] Saved {aug_fname}")

print(f"[DONE] {NORMAL_DIR} now contains {current_count} images.")

# Repeat for faulty images in dataset/test/faulty
FAULTY_DIR = 'dataset/test/faulty'
TARGET_COUNT = 100
images = [f for f in os.listdir(FAULTY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
current_count = len(images)
print(f"[INFO] Found {current_count} images in {FAULTY_DIR}")

if current_count == 0:
    print("[ERROR] No faulty images found to augment.")
    exit(1)

aug_idx = 0
while current_count < TARGET_COUNT:
    for fname in images:
        if current_count >= TARGET_COUNT:
            break
        img_path = os.path.join(FAULTY_DIR, fname)
        img = Image.open(img_path).convert('RGB')
        aug = random.choice(AUGS)
        aug_img = aug(img)
        aug_fname = f"aug_{aug_idx}_{fname}"
        aug_img.save(os.path.join(FAULTY_DIR, aug_fname))
        aug_idx += 1
        current_count += 1
        print(f"[AUG] Saved {aug_fname}")

print(f"[DONE] {FAULTY_DIR} now contains {current_count} images.")

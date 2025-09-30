import os
from PIL import Image

TRAIN_NORMAL = 'dataset/train/normal'
TRAIN_FAULTY = 'dataset/train/faulty'
CROP_PERCENT = 0.125  # 12.5%

for folder in [TRAIN_NORMAL, TRAIN_FAULTY]:
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        fpath = os.path.join(folder, fname)
        img = Image.open(fpath)
        w, h = img.size
        crop_w = int(w * CROP_PERCENT)
        # Crop 12.5% from left and right
        cropped = img.crop((crop_w, 0, w - crop_w, h))
        cropped.save(fpath)
        print(f"[CROP] {fpath} -> size {cropped.size}")

print("[DONE] All training images cropped 12.5% from both sides.")

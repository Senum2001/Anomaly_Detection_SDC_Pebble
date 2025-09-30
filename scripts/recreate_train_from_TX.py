import os
import shutil
import random

TX_DIR = 'TX'
TRAIN_NORMAL = 'dataset/train/normal'
TRAIN_FAULTY = 'dataset/train/faulty'
TARGET_COUNT = 100

# Clean train folders
shutil.rmtree(TRAIN_NORMAL, ignore_errors=True)
os.makedirs(TRAIN_NORMAL, exist_ok=True)
shutil.rmtree(TRAIN_FAULTY, ignore_errors=True)
os.makedirs(TRAIN_FAULTY, exist_ok=True)

# Helper to collect images from TX folders
def collect_images(src_pattern, dst_folder, needed):
    collected = 0
    for root, dirs, files in os.walk(TX_DIR):
        if src_pattern in root:
            imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(imgs)
            for f in imgs:
                if collected >= needed:
                    return
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_folder, f)
                if not os.path.exists(dst_path):
                    shutil.copyfile(src_path, dst_path)
                    collected += 1
                    print(f"[COPY] {src_path} -> {dst_path}")

# Fill train/normal and train/faulty to TARGET_COUNT from TX
collect_images('normal', TRAIN_NORMAL, TARGET_COUNT)
collect_images('faulty', TRAIN_FAULTY, TARGET_COUNT)

print(f"[DONE] {TRAIN_NORMAL} images: {len(os.listdir(TRAIN_NORMAL))}")
print(f"[DONE] {TRAIN_FAULTY} images: {len(os.listdir(TRAIN_FAULTY))}")

import os
import shutil
import random

TEST_NORMAL = 'dataset/test/normal'
TEST_FAULTY = 'dataset/test/faulty'
TX_DIR = 'TX'
TARGET_COUNT = 100

os.makedirs(TEST_NORMAL, exist_ok=True)
os.makedirs(TEST_FAULTY, exist_ok=True)

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

# Fill test/normal and test/faulty to TARGET_COUNT from TX (do not touch train)
collect_images('normal', TEST_NORMAL, TARGET_COUNT)
collect_images('faulty', TEST_FAULTY, TARGET_COUNT)

print(f"[DONE] {TEST_NORMAL} images: {len(os.listdir(TEST_NORMAL))}")
print(f"[DONE] {TEST_FAULTY} images: {len(os.listdir(TEST_FAULTY))}")

import os
from PIL import Image, ImageOps, ImageEnhance
import random
import shutil

TRAIN_NORMAL = 'dataset/train/normal'
TRAIN_FAULTY = 'dataset/train/faulty'
TEST_NORMAL = 'dataset/test/normal'
TEST_FAULTY = 'dataset/test/faulty'
TARGET_COUNT = 100

os.makedirs(TRAIN_NORMAL, exist_ok=True)
os.makedirs(TRAIN_FAULTY, exist_ok=True)

# Helper: move images from test to train if needed
def move_images(src, dst, needed):
    imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    moved = 0
    for f in imgs:
        if moved >= needed:
            break
        shutil.move(os.path.join(src, f), os.path.join(dst, f))
        moved += 1
    return moved

# 1. Move images from test to train if train folders have < TARGET_COUNT
normal_needed = TARGET_COUNT - len([f for f in os.listdir(TRAIN_NORMAL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
faulty_needed = TARGET_COUNT - len([f for f in os.listdir(TRAIN_FAULTY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
if normal_needed > 0:
    move_images(TEST_NORMAL, TRAIN_NORMAL, normal_needed)
if faulty_needed > 0:
    move_images(TEST_FAULTY, TRAIN_FAULTY, faulty_needed)

# 2. Augment if still not enough
AUGS = [
    lambda img: img.rotate(random.randint(-30, 30)),
    lambda img: ImageOps.mirror(img),
    lambda img: ImageOps.flip(img),
    lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
    lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
    lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
]

def augment_to_count(folder, target):
    images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(images)
    aug_idx = 0
    while current_count < target:
        for fname in images:
            if current_count >= target:
                break
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('RGB')
            aug = random.choice(AUGS)
            aug_img = aug(img)
            aug_fname = f"aug_{aug_idx}_{fname}"
            aug_img.save(os.path.join(folder, aug_fname))
            aug_idx += 1
            current_count += 1
            print(f"[AUG] Saved {aug_fname} in {folder}")

augment_to_count(TRAIN_NORMAL, TARGET_COUNT)
augment_to_count(TRAIN_FAULTY, TARGET_COUNT)

print(f"[DONE] {TRAIN_NORMAL} images: {len(os.listdir(TRAIN_NORMAL))}")
print(f"[DONE] {TRAIN_FAULTY} images: {len(os.listdir(TRAIN_FAULTY))}")

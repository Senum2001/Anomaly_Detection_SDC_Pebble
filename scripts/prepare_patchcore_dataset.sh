#!/usr/bin/env bash
set -euo pipefail

# 1. Gather all normal images from TX folders into train/normal and test/normal
mkdir -p dataset/train/normal
mkdir -p dataset/test/normal

# Move/copy all normal images from TX folders to train/normal (for training)
find TX/ -type f -name '*.jpg' -path '*/normal/*' -exec cp {} dataset/train/normal/ \;

# Optionally, split some normal images into test/normal (for evaluation)
# Here, move 20% of images to test/normal (adjust as needed)
cd dataset/train/normal
mkdir -p ../test/normal
count=$(ls -1 | wc -l)
test_count=$((count / 5))
ls | shuf | head -n $test_count | xargs -I{} mv {} ../test/normal/
cd ../../..

# 2. Gather all faulty images from TX folders into test/faulty (for evaluation)
mkdir -p dataset/test/faulty
find TX/ -type f -name '*.jpg' -path '*/faulty/*' -exec cp {} dataset/test/faulty/ \;

# 3. (Optional) Remove duplicates between train/normal and test/normal
# This step assumes filenames are unique. If not, use a more robust deduplication method.
cd dataset/test/normal
for f in *; do
  [ -e "../../train/normal/$f" ] && rm -f "../../train/normal/$f"
done
cd ../../..

# 4. Print summary
train_n=$(ls dataset/train/normal | wc -l)
test_n=$(ls dataset/test/normal | wc -l)
test_f=$(ls dataset/test/faulty | wc -l)
echo "[✓] Normal images in train: $train_n"
echo "[✓] Normal images in test:  $test_n"
echo "[✓] Faulty images in test:  $test_f"

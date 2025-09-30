#!/usr/bin/env bash
set -euo pipefail

# 0) Environment
python3 -V || true
echo "[*] Installing python3-full and python3-venv if needed…"
sudo apt update && sudo apt install -y python3-full python3.12-venv
echo "[*] Creating virtual environment if not exists…"
if [ ! -d ~/anomalib_env ]; then
  python3 -m venv ~/anomalib_env
  source ~/anomalib_env/bin/activate
  pip install -U pip
  pip install "anomalib[full]" flask requests cloudinary pillow numpy opencv-python omegaconf torch
else
  source ~/anomalib_env/bin/activate
  pip install -U pip
  pip install "anomalib[full]" flask requests cloudinary pillow numpy opencv-python omegaconf torch
fi


# # 1) Train (PatchCore builds the memory bank from normals)
# anomalib train \
#   --config configs/patchcore_transformers.yaml
#
# CKPT=$(ls -1t results/transformers/patchcore/*/weights/*.ckpt | head -n 1)
# echo "[*] Using checkpoint: $CKPT"
#
# # 2) Test/Eval on test/{normal,faulty}
# anomalib test \
#   --config configs/patchcore_transformers.yaml \
#   --ckpt_path "$CKPT"

echo
echo "[✓] Done. Check:"
echo "  • results/transformers/patchcore/**/images/    (heatmaps & overlays)"
echo "  • results/transformers/patchcore/**/metrics.csv (AUROC/F1 etc.)"

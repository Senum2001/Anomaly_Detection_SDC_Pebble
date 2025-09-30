import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# PatchCore import for your config (update if needed for your Anomalib version)
from anomalib.models.image.patchcore import Patchcore

# Config from your YAML
CKPT_PATH = "results/Patchcore/transformers/v2/weights/lightning/model.ckpt"
IMAGE_SIZE = (256, 256)  # Update if you used a different size in training

# Inference directory (change as needed)
INFER_DIR = "./dataset/test/faulty"

# Output directory for masks
OUT_MASK_DIR = "inference_masks"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Load model
model = Patchcore.load_from_checkpoint(CKPT_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Transforms (should match your training config)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inference loop
for fname in sorted(os.listdir(INFER_DIR)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue
    fpath = os.path.join(INFER_DIR, fname)
    img = Image.open(fpath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        # PatchCore returns (anomaly_score, anomaly_map, ...)
        if hasattr(output, 'anomaly_map'):
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
        elif isinstance(output, (tuple, list)) and len(output) > 1:
            anomaly_map = output[1].squeeze().cpu().numpy()
        else:
            anomaly_map = None
    # Save mask as PNG
    if anomaly_map is not None:
        # Normalize to 0-255 for visualization
        norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
        mask_img = Image.fromarray(norm_map)
        mask_img.save(os.path.join(OUT_MASK_DIR, f"{os.path.splitext(fname)[0]}_mask.png"))
        print(f"Saved mask for {fname}")
    else:
        print(f"No mask generated for {fname}")

print(f"All masks saved to {OUT_MASK_DIR}")

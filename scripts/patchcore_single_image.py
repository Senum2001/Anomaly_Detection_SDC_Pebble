import sys
from patchcore_api_inference import Patchcore, config, device
from PIL import Image
import torch
import numpy as np
import os

# Output directories (should match those in patchcore_api_inference.py)
OUT_MASK_DIR = "api_inference_pred_masks_pipeline"
OUT_FILTERED_DIR = "api_inference_filtered_pipeline"
os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_FILTERED_DIR, exist_ok=True)

# Load model
model = Patchcore.load_from_checkpoint(
    "results/Patchcore/transformers/v7/weights/lightning/model.ckpt",
    **config.model.init_args
)
model.eval()
model = model.to(device)

def infer_single_image(image_path):
    # Load and preprocess image
    # Normalize path for cross-platform compatibility
    import os
    fixed_path = os.path.normpath(image_path)
    fixed_path = os.path.abspath(fixed_path)
    orig_img = Image.open(fixed_path).convert("RGB")
    # Use the same transforms as in training (resize, normalize)
    # If you have a transform pipeline, import and use it here
    img_resized = orig_img.resize((256, 256))  # Change if your model uses a different size
    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        if hasattr(output, 'anomaly_map'):
            anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
        elif isinstance(output, (tuple, list)) and len(output) > 1:
            anomaly_map = output[1].squeeze().cpu().numpy()
        else:
            anomaly_map = None
    if anomaly_map is not None:
        norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
        if norm_map.ndim > 2:
            norm_map = np.squeeze(norm_map)
            if norm_map.ndim > 2:
                norm_map = norm_map[0]
        mask_img = Image.fromarray(norm_map)
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        mask_img.save(os.path.join(OUT_MASK_DIR, out_name))
        print(f"Saved mask for {image_path}")

        # Resize mask to match original image size if needed
        if mask_img.size != orig_img.size:
            mask_img_resized = mask_img.resize(orig_img.size, resample=Image.BILINEAR)
        else:
            mask_img_resized = mask_img
        bin_mask = np.array(mask_img_resized) > 128
        orig_np = np.array(orig_img)
        filtered_np = np.zeros_like(orig_np)
        filtered_np[bin_mask] = orig_np[bin_mask]
        filtered_img = Image.fromarray(filtered_np)
        filtered_name = os.path.splitext(os.path.basename(image_path))[0] + "_filtered.png"
        filtered_img.save(os.path.join(OUT_FILTERED_DIR, filtered_name))
        print(f"Saved filtered image for {image_path}")
    else:
        print(f"No mask generated for {image_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python patchcore_single_image.py <image_path>")
        sys.exit(1)
    infer_single_image(sys.argv[1])

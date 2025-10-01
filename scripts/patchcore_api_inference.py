import os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from anomalib.models import Patchcore
from anomalib.data import Folder
from pytorch_lightning import Trainer
# --- Load config ---
CONFIG_PATH = "configs/patchcore_transformers.yaml"
CKPT_PATH = "results/Patchcore/transformers/v7/weights/lightning/model.ckpt"

OUT_MASK_DIR = "api_inference_pred_pipeline"
OUT_FILTERED_DIR = "api_inference_filtered_pipeline"

os.makedirs(OUT_MASK_DIR, exist_ok=True)
os.makedirs(OUT_FILTERED_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load config
config = OmegaConf.load(CONFIG_PATH)

if __name__ == "__main__":
    os.makedirs(OUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUT_FILTERED_DIR, exist_ok=True)
    # Setup datamodule for prediction (use test set)
    # Use arguments matching the YAML config and Folder datamodule signature
    data_module = Folder(
        name=config.data.init_args.name,
        root=config.data.init_args.root,
        normal_dir=config.data.init_args.normal_dir,
        abnormal_dir=config.data.init_args.abnormal_dir,
        normal_test_dir=config.data.init_args.normal_test_dir,
        train_batch_size=config.data.init_args.train_batch_size,
        eval_batch_size=config.data.init_args.eval_batch_size,
        num_workers=config.data.init_args.num_workers,
    )
    data_module.setup()

    # Load model
    model = Patchcore.load_from_checkpoint(CKPT_PATH, **config.model.init_args)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Inference loop
    for batch in data_module.test_dataloader():
        img = batch.image.to(device)
        fname = batch.image_path[0]
        with torch.no_grad():
            output = model(img)
            # PatchCore returns (anomaly_score, anomaly_map, ...)
            if hasattr(output, 'anomaly_map'):
                anomaly_map = output.anomaly_map.squeeze().cpu().numpy()
            elif isinstance(output, (tuple, list)) and len(output) > 1:
                anomaly_map = output[1].squeeze().cpu().numpy()
            else:
                anomaly_map = None
        if anomaly_map is not None:
            # Normalize to 0-255 for visualization
            norm_map = (255 * (anomaly_map - anomaly_map.min()) / (np.ptp(anomaly_map) + 1e-8)).astype(np.uint8)
            # Ensure norm_map is 2D for PIL
            if norm_map.ndim > 2:
                norm_map = np.squeeze(norm_map)
                if norm_map.ndim > 2:
                    norm_map = norm_map[0]
            mask_img = Image.fromarray(norm_map)
            out_name = os.path.splitext(os.path.basename(fname))[0] + "_mask.png"
            mask_img.save(os.path.join(OUT_MASK_DIR, out_name))
            print(f"Saved mask for {fname}")

            # Save filtered (masked) part of the original transformer image
            orig_img = Image.open(fname).convert("RGB")
            # Resize mask to match original image size if needed
            if mask_img.size != orig_img.size:
                mask_img_resized = mask_img.resize(orig_img.size, resample=Image.BILINEAR)
            else:
                mask_img_resized = mask_img
            # Binarize mask (threshold at 128)
            bin_mask = np.array(mask_img_resized) > 128
            # Apply mask to original image
            orig_np = np.array(orig_img)
            filtered_np = np.zeros_like(orig_np)
            filtered_np[bin_mask] = orig_np[bin_mask]
            filtered_img = Image.fromarray(filtered_np)
            filtered_name = os.path.splitext(os.path.basename(fname))[0] + "_filtered.png"
            filtered_img.save(os.path.join(OUT_FILTERED_DIR, filtered_name))
            print(f"Saved filtered image for {fname}")
        else:
            print(f"No mask generated for {fname}")

    print(f"All masks saved to {OUT_MASK_DIR}")

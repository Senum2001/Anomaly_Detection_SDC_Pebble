import os
import torch
from torchvision import transforms
from PIL import Image
from anomalib.models.image.patchcore import Patchcore
import pandas as pd

# Path to your trained checkpoint
CKPT_PATH = "results/Patchcore/transformers/v2/weights/lightning/model.ckpt"
# Folder with images to score
INPUT_DIR = "./dataset/test/faulty"  # Change as needed
# Output CSV for scores
OUTPUT_CSV = "patchcore_batch_scores.csv"

# Load PatchCore model
model = Patchcore.load_from_checkpoint(CKPT_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Define image transforms (should match your training transforms)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Change to your image_size if different
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img)

# Score all images in the input directory
results = []
for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        continue
    fpath = os.path.join(INPUT_DIR, fname)
    img_tensor = load_image(fpath).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
           output = model(img_tensor.to(device))  # Move to device
        # PatchCore returns a namedtuple with anomaly_score
           if hasattr(output, 'anomaly_score'):
               score = output.anomaly_score.item()
           elif isinstance(output, (tuple, list)):
               score = float(output[0])
           else:
               raise RuntimeError("Unknown PatchCore output type: {}".format(type(output)))
    results.append({"image": fname, "anomaly_score": score})
    print(f"{fname}: {score:.4f}")

# Save results to CSV
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"Saved batch scores to {OUTPUT_CSV}")

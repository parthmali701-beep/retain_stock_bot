import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import YolosImageProcessor, YolosForObjectDetection
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"
ASSETS_DIR = "assets"

# --- NEW: CUSTOM LABEL MAPPING ---
# You can explicitly change the name assigned to any class ID here.
# If an ID is not listed here, it will automatically use the cleaned-up filename.
CUSTOM_LABELS = {
    0: "Red Box",
    1: "Green Bottle",
    2: "Blue Box",
    3: "Yellow Bottle",
    4: "Purple Can",
    5: "Orange Box",
    6: "Pink Soap",
    7: "Black Energy Drink",
    8: "Sky Milk Carton",
    9: "Cyan Shampoo",
    10: "Brown Coffee Jar",
    11: "Forest Tea Box",
    12: "Magenta Lotion",
    13: "Silver Soup Can",
    14: "Gold Perfume",
    15: "Navy Body Wash",
    16: "White Deodorant",
    17: "Peach Juice",
    18: "Olive Oil",
    19: "Bronze Tuna Tin",
    20: "Coral Shampoo",
    21: "Neon Energy Drink",
    22: "Lime Juice Box",
    23: "Clear Water Bottle"
}

# Dynamically map the 24 products directly from your assets folder
asset_files = sorted(glob.glob(f"{ASSETS_DIR}/*.urdf"))
if not asset_files:
    raise FileNotFoundError("[ERROR] No assets found. Run build_assets.py first.")

id2label = {}
for i, f in enumerate(asset_files):
    # Get raw name (e.g., "00_Red_Box")
    raw_name = os.path.basename(f).replace(".urdf", "")
    
    # Strip the number prefix and replace underscores (e.g., "Red Box")
    clean_name = raw_name.split("_", 1)[-1].replace("_", " ") if "_" in raw_name else raw_name
    
    # Use custom label if provided, otherwise use the cleaned filename
    id2label[i] = CUSTOM_LABELS.get(i, clean_name)

label2id = {v: k for k, v in id2label.items()}

print(f"[SYSTEM] Loaded {len(id2label)} classes for training.")

MODEL_NAME = "hustvl/yolos-tiny"

# ==========================================
# 2. CUSTOM DATASET LOADER (COCO FORMAT)
# ==========================================
class RetailInventoryDataset(Dataset):
    def __init__(self, images_dir, labels_dir, processor):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        annotations = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        
                        # Convert YOLO normalized to COCO Absolute: [x_min, y_min, width, height]
                        abs_w = w * img_width
                        abs_h = h * img_height
                        x_min = (cx * img_width) - (abs_w / 2)
                        y_min = (cy * img_height) - (abs_h / 2)
                        
                        annotations.append({
                            "image_id": idx,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, abs_w, abs_h],
                            "area": abs_w * abs_h,
                            "iscrowd": 0
                        })
        
        target = {
            "image_id": idx,
            "annotations": annotations
        }
        
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0]
        }

# ==========================================
# 3. INITIALIZE AI MODEL (6GB VRAM OPTIMIZED)
# ==========================================
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    return {"pixel_values": pixel_values, "labels": labels}

print("[SYSTEM] Loading Neural Network...")
processor = YolosImageProcessor.from_pretrained(MODEL_NAME)

# Wipes the original COCO knowledge and forces it to learn our 24 custom classes
model = YolosForObjectDetection.from_pretrained(
    MODEL_NAME,
    ignore_mismatched_sizes=True,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

# MICRO-BATCHING: Batch size of 2 keeps VRAM usage far below the 6GB limit
BATCH_SIZE = 2
dataset = RetailInventoryDataset(IMAGES_DIR, LABELS_DIR, processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Clear unused memory before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ==========================================
# 4. THE TRAINING LOOP
# ==========================================
epochs = 20
accumulation_steps = 4  # Simulates a batch size of 8 (2 * 4)

# Mixed Precision Scaler for 16-bit math (massively reduces VRAM usage)
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[SYSTEM] Hardware Detected: {device}. Starting Training for {epochs} epochs...")

model.train()
for epoch in range(epochs):
    loop = tqdm(dataloader, leave=True)
    epoch_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loop):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        
        # Enable Mixed Precision
        with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / accumulation_steps
        
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        
        # Gradient Accumulation Step
        if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        loop.set_description(f"Epoch {epoch+1}/{epochs}")
        loop.set_postfix(loss=loss.item() * accumulation_steps)
        
    print(f"--- Epoch {epoch+1} Average Loss: {epoch_loss/len(dataloader):.4f} ---")

# ==========================================
# 5. SAVE THE TRAINED BRAIN
# ==========================================
model.save_pretrained("custom_retail_yolos")
processor.save_pretrained("custom_retail_yolos")
print("\n[SYSTEM] Training Complete! Custom weights safely saved to 'custom_retail_yolos/' directory.")
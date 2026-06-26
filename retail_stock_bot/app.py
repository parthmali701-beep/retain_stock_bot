import os
import cv2
import torch
import sqlite3
import re
from PIL import Image
from transformers import YolosImageProcessor, YolosForObjectDetection

print("[SYSTEM] Initializing Batch Inference & Database Pipeline...")

# ==========================================
# 1. DATABASE CONNECTION
# ==========================================
db_name = "retail_store.db"
if not os.path.exists(db_name):
    print(f"[ERROR] Database '{db_name}' not found! Run database_setup.py first.")
    exit()

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

# Fetch capacities so the AI knows how to calculate missing stock
cursor.execute("SELECT product_id, max_capacity FROM products")
MAX_CAPACITIES = {row[0]: row[1] for row in cursor.fetchall()}

# ==========================================
# 2. SETUP DIRECTORIES
# ==========================================
INPUT_DIR = "scans"
OUTPUT_DIR = "predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 3. SMART HARDWARE DETECTOR
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[SYSTEM] Hardware Engine: {device.type.upper()}")

# ==========================================
# 4. LOAD THE AI MODEL
# ==========================================
model_name = "custom_retail_yolos"
processor = YolosImageProcessor.from_pretrained(model_name)
model = YolosForObjectDetection.from_pretrained(model_name)
model.to(device)
print("[SYSTEM] YOLOS Model loaded successfully.\n")

valid_extensions = (".png", ".jpg", ".jpeg")
image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_extensions)]

if not image_files:
    print(f"[WARNING] No images found in '{INPUT_DIR}/'. Run your robot simulation first!")
    exit()

print(f"[SYSTEM] Found {len(image_files)} images to process. Starting pipeline...\n")

# --- THE FIX: Group images by their scan number so top & bottom shelf cameras combine forces ---
scan_aggregates = {}

# ==========================================
# 5. PROCESS EVERY IMAGE
# ==========================================
for filename in image_files:
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, f"pred_{filename}")
    
    # Extract the scan number from the filename (e.g., 'scan1' from 'run_live_scan1_Left_Low.png')
    match = re.search(r'(scan\d+)', filename)
    scan_id = match.group(1) if match else "unknown_scan"
    
    if scan_id not in scan_aggregates:
        scan_aggregates[scan_id] = {}
    
    frame = cv2.imread(input_path)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.7)[0]

    # Tally for THIS specific image
    scan_tally = {}
    boxes_drawn = 0
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        x1, y1, x2, y2 = box
        
        class_id = label.item()
        label_name = model.config.id2label[class_id]
        confidence = round(score.item() * 100, 1)
        
        scan_tally[class_id] = scan_tally.get(class_id, 0) + 1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{label_name} {confidence}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        boxes_drawn += 1

    # Add this image's count to the total count for this specific physical scan
    for class_id, count in scan_tally.items():
        scan_aggregates[scan_id][class_id] = scan_aggregates[scan_id].get(class_id, 0) + count

    cv2.imwrite(output_path, frame)
    print(f" -> Processed {filename} | Found {boxes_drawn} objects | Added to {scan_id}")

# ==========================================
# 6. AGGREGATE THE BEST OVERALL VIEW
# ==========================================
final_store_tally = {}

# Now we look at the combined totals of every scan location, and lock in the highest total!
for scan_id, combined_tally in scan_aggregates.items():
    for class_id, total_count in combined_tally.items():
        if class_id not in final_store_tally or total_count > final_store_tally[class_id]:
            final_store_tally[class_id] = total_count

# ==========================================
# 7. INJECT PERFECTED DATA INTO DATABASE
# ==========================================
print("\n[SYSTEM] All images processed. Writing final verified inventory to database...")

for class_id, best_count in final_store_tally.items():
    max_cap = MAX_CAPACITIES.get(class_id, 6)
    missing = max(0, max_cap - best_count)
    
    cursor.execute("""
        INSERT INTO inventory_scans (product_id, detected_count, missing_stock)
        VALUES (?, ?, ?)
    """, (class_id, best_count, missing))
    
conn.commit()
conn.close()

print(f"\n[SUCCESS] Pipeline complete. Refresh your Dashboard to see the accurate counts!")
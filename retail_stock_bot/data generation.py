import pybullet as p
import pybullet_data
import time
import math
import os
import random
import glob
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. SETUP DIRECTORIES & LOGS
# ==========================================
SESSION_ID = int(time.time())
IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

global_inventory_log = {}

# Read all 24 files, sort them alphabetically to guarantee YOLO IDs 0-23 stay locked
asset_files = sorted(glob.glob("assets/*.urdf"))
if not asset_files:
    print("[ERROR] No URDFs found in 'assets/'. Please run build_assets.py first!")
    exit()

# ==========================================
# 2. MATERIAL SPECULAR MAPPING (Updated for 24 Items)
# ==========================================
# [0,0,0] = Matte | [0.5,0.5,0.5] = Glossy Plastic | [1,1,1] = Shiny Metal/Glass
SPECULAR_MAP = {
    # Boxes (Glossy Plastic)
    0: [0.5, 0.5, 0.5], 2: [0.5, 0.5, 0.5], 5: [0.5, 0.5, 0.5], 6: [0.5, 0.5, 0.5], 
    8: [0.5, 0.5, 0.5], 11: [0.5, 0.5, 0.5], 14: [0.5, 0.5, 0.5], 17: [0.5, 0.5, 0.5],
    22: [0.5, 0.5, 0.5], # Lime Juice Box
    
    # Cans (Shiny Metal)
    4: [1.0, 1.0, 1.0], 7: [1.0, 1.0, 1.0], 10: [1.0, 1.0, 1.0], 
    13: [1.0, 1.0, 1.0], 16: [1.0, 1.0, 1.0], 19: [1.0, 1.0, 1.0],
    21: [1.0, 1.0, 1.0], # Neon Energy Drink
    
    # Bottles (Mixed Materials)
    1: [0.5, 0.5, 0.5],   # Sprite (Glossy)
    3: [1.0, 1.0, 1.0],   # Yellow Bottle (Glass)
    9: [0.0, 0.0, 0.0],   # Cyan Shampoo (Matte)
    12: [0.5, 0.5, 0.5],  # Magenta Lotion (Glossy)
    15: [0.5, 0.5, 0.5],  # Navy Body Wash (Glossy)
    18: [1.0, 1.0, 1.0],  # Olive Oil (Glass)
    20: [0.0, 0.0, 0.0],  # Coral Shampoo (Matte)
    23: [1.0, 1.0, 1.0]   # Clear Water (Glass)
}

# ==========================================
# 3. BUILD THE STORE
# ==========================================
def create_rack(name, x_center, facing_dir=1, y_start=-3.0, y_end=3.0):
    depth, length, thickness = 0.40, y_end - y_start, 0.02
    white_color, darkgrey_color = [0.95, 0.95, 0.95, 1], [0.1, 0.1, 0.1, 1]
    
    backboard_thickness, backboard_height = 0.02, 1.4
    bb_x = x_center - (facing_dir * (depth / 2)) + (facing_dir * (backboard_thickness / 2))
    bb_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[backboard_thickness/2, length/2, backboard_height/2], rgbaColor=white_color)
    bb_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[backboard_thickness/2, length/2, backboard_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bb_col, baseVisualShapeIndex=bb_vis, basePosition=[bb_x, 0, backboard_height/2])

    shelf_heights = [0.2, 0.6, 1.0]
    for height in shelf_heights:
        shelf_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[depth/2, length/2, thickness/2], rgbaColor=white_color)
        shelf_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[depth/2, length/2, thickness/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_col, baseVisualShapeIndex=shelf_vis, basePosition=[x_center, 0, height])
        
        strip_depth, strip_height = 0.01, 0.03
        strip_x = x_center + (facing_dir * (depth / 2)) - (facing_dir * (strip_depth / 2))
        strip_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[strip_depth/2, length/2, strip_height/2], rgbaColor=darkgrey_color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=strip_vis, basePosition=[strip_x, 0, height])

    for y_pos in range(int(y_start), int(y_end) + 1):
        div_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[depth/2, thickness/2, backboard_height/2], rgbaColor=white_color)
        div_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[depth/2, thickness/2, backboard_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=div_col, baseVisualShapeIndex=div_vis, basePosition=[x_center, y_pos, backboard_height/2])

def populate_rack(x_center, facing_dir, assigned_products):
    """Fills 1-meter bays tightly (15cm gaps) and logs them for YOLO dataset."""
    shelf_heights = [0.2, 0.6, 1.0]
    
    for height in shelf_heights:
        current_y = -2.9 # Start at the far edge of the 6-meter rack
        
        while current_y < 2.9:
            # Skip vertical dividers
            if abs(round(current_y) - current_y) < 0.06:
                current_y += 0.05
                continue

            # Calculate which 1-meter bay we are currently in (Zone 0 through 5)
            zone_idx = int((current_y + 3.0) // 1.0)
            zone_idx = max(0, min(zone_idx, 5)) 

            # Lock in the exact product assigned to this 1-meter bay
            selected_class_id = assigned_products[zone_idx]
            selected_urdf = asset_files[selected_class_id]
            
            # Depth stacking (front row and back row)
            row_offsets = [0.10, -0.10]  
            
            for offset in row_offsets:
                current_x = x_center + (facing_dir * offset)
                
                # 85% chance to spawn an item to leave realistic gaps
                if random.random() < 0.85:  
                    body_id = p.loadURDF(selected_urdf, [current_x, current_y, height + 0.01])
                    
                    # CRITICAL: Log the body_id for the segmentation mask to read later
                    global_inventory_log[body_id] = selected_class_id
                    
                    # Apply the material finish dynamically
                    p.changeVisualShape(body_id, linkIndex=-1, specularColor=SPECULAR_MAP.get(selected_class_id, [0.5, 0.5, 0.5]))
                    
                    # If it's a compound bottle, ensure the neck/cap gets the finish too
                    if "bottle" in selected_urdf:
                        p.changeVisualShape(body_id, linkIndex=0, specularColor=SPECULAR_MAP.get(selected_class_id, [0.5, 0.5, 0.5]))
            
            # Move 15cm down the shelf to pack the next item tightly
            current_y += 0.15 

def build_store_layout():
    """Builds the 4-rack layout mapped from the top-down sketch and populates it."""
    # Create a master plan of all 24 items and shuffle them
    store_inventory_plan = list(range(len(asset_files)))
    random.shuffle(store_inventory_plan)
    
    # Safely handle if you haven't generated all 24 assets yet
    if len(store_inventory_plan) < 24:
        store_inventory_plan = (store_inventory_plan * 4)[:24]

    create_rack("Rack 1", x_center=-0.8, facing_dir=1)
    populate_rack(-0.8, 1, store_inventory_plan[0:6])
    
    create_rack("Rack 2", x_center=0.705, facing_dir=-1) 
    populate_rack(0.705, -1, store_inventory_plan[6:12])
    
    create_rack("Rack 3", x_center=1.105, facing_dir=1) 
    populate_rack(1.105, 1, store_inventory_plan[12:18])
    
    create_rack("Rack 4", x_center=2.6, facing_dir=-1)
    populate_rack(2.6, -1, store_inventory_plan[18:24])

# ==========================================
# 4. RANDOMIZED DATA FACTORY LOOP
# ==========================================
p.connect(p.GUI) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

build_store_layout()

# Let physics settle
for _ in range(100):
    p.stepSimulation()

num_images_to_generate = 2000
width, height_res = 480, 360  # Matches your robot script resolution

print(f"\n[SYSTEM] Starting Synthetic Data Generation: {num_images_to_generate} images...")

for i in range(num_images_to_generate):
    # Randomize which aisle the camera is in (Aisle 1 or Aisle 2)
    in_aisle_1 = random.choice([True, False])
    cam_y = random.uniform(-2.8, 2.8)
    cam_z = random.uniform(0.3, 1.3)   
    
    if in_aisle_1:
        cam_x = random.uniform(-0.4, 0.3)
        target_x = random.choice([-0.8, 0.705]) # Look at Rack 1 or 2
    else:
        cam_x = random.uniform(1.5, 2.2)
        target_x = random.choice([1.105, 2.6])  # Look at Rack 3 or 4

    target_y = cam_y + random.uniform(-0.2, 0.2) 
    target_z = cam_z + random.uniform(-0.3, 0.3) 

    view_matrix = p.computeViewMatrix(cameraEyePosition=[cam_x, cam_y, cam_z], cameraTargetPosition=[target_x, target_y, target_z], cameraUpVector=[0, 0, 1])
    
    fov = random.uniform(60.0, 80.0)
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect=1.33, nearVal=0.01, farVal=3.0)

    _, _, rgb_img, _, seg_mask = p.getCameraImage(width, height_res, viewMatrix=view_matrix, projectionMatrix=projection_matrix, renderer=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    
    np_img = np.array(rgb_img, dtype=np.uint8).reshape((height_res, width, 4))
    seg_mask_arr = np.array(seg_mask, dtype=np.int32).reshape((height_res, width))
    
    yolo_labels = []
    
    # Compound URDF bitmask correction: 
    # Strips link index so caps and bodies group as the same object
    unique_pixel_values = np.unique(seg_mask_arr)
    object_pixels = {}
    
    for pixel_val in unique_pixel_values:
        actual_body_id = pixel_val & ((1 << 24) - 1) # Pure Body ID
        
        if actual_body_id in global_inventory_log:
            if actual_body_id not in object_pixels:
                object_pixels[actual_body_id] = {'x': [], 'y': []}
            
            y_indices, x_indices = np.where(seg_mask_arr == pixel_val)
            object_pixels[actual_body_id]['x'].extend(x_indices)
            object_pixels[actual_body_id]['y'].extend(y_indices)

    # Calculate bounding boxes
    for body_id, coords in object_pixels.items():
        if len(coords['x']) > 20: # Ignore objects that are barely visible 
            class_id = global_inventory_log[body_id]
            
            min_x, max_x = np.min(coords['x']), np.max(coords['x'])
            min_y, max_y = np.min(coords['y']), np.max(coords['y'])
            
            box_width = (max_x - min_x) / width
            box_height = (max_y - min_y) / height_res
            center_x = ((min_x + max_x) / 2.0) / width
            center_y = ((min_y + max_y) / 2.0) / height_res
            
            if box_width > 0.01 and box_height > 0.01:
                yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

    # Only save if products are visible
    if yolo_labels:
        file_prefix = f"run_{SESSION_ID}_img_{i}"
        plt.imsave(f"{IMAGES_DIR}/{file_prefix}.png", np_img)
        with open(f"{LABELS_DIR}/{file_prefix}.txt", "w") as f:
            f.write("\n".join(yolo_labels))

    if i % 50 == 0:
        print(f"Generated {i}/{num_images_to_generate} images...")

print("\n[SYSTEM] Dataset Generation Complete!")
p.disconnect()
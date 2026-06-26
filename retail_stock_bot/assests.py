import os
import shutil

# --- NEW: Aggressively delete the old folder if it exists ---
if os.path.exists("assets"):
    shutil.rmtree("assets")

# Create a fresh, completely empty assets folder
os.makedirs("assets", exist_ok=True)

def generate_box_urdf(name, r, g, b, a, x, y, z):
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <material name="mat_{name}"><color rgba="{r} {g} {b} {a}"/></material>
  <link name="baseLink">
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0 0 {z/2}"/><geometry><box size="{x} {y} {z}"/></geometry><material name="mat_{name}"/></visual>
    <collision><origin xyz="0 0 {z/2}"/><geometry><box size="{x} {y} {z}"/></geometry></collision>
  </link>
</robot>"""

def generate_can_urdf(name, r, g, b, a, radius, length):
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <material name="mat_{name}"><color rgba="{r} {g} {b} {a}"/></material>
  <link name="baseLink">
    <inertial><mass value="0.2"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0 0 {length/2}"/><geometry><cylinder radius="{radius}" length="{length}"/></geometry><material name="mat_{name}"/></visual>
    <collision><origin xyz="0 0 {length/2}"/><geometry><cylinder radius="{radius}" length="{length}"/></geometry></collision>
  </link>
</robot>"""

def generate_bottle_urdf(name, body_r, body_g, body_b, body_a, cap_r, cap_g, cap_b, radius, length):
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <material name="mat_body_{name}"><color rgba="{body_r} {body_g} {body_b} {body_a}"/></material>
  <material name="mat_cap_{name}"><color rgba="{cap_r} {cap_g} {cap_b} 1"/></material>
  <link name="baseLink">
    <inertial><mass value="0.15"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><origin xyz="0 0 {length/2}"/><geometry><cylinder radius="{radius}" length="{length}"/></geometry><material name="mat_body_{name}"/></visual>
    <collision><origin xyz="0 0 {length/2}"/><geometry><cylinder radius="{radius}" length="{length}"/></geometry></collision>
  </link>
  <link name="neck">
    <inertial><mass value="0.05"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
    <visual><origin xyz="0 0 {length + 0.025}"/><geometry><cylinder radius="{radius*0.4}" length="0.05"/></geometry><material name="mat_cap_{name}"/></visual>
    <collision><origin xyz="0 0 {length + 0.025}"/><geometry><cylinder radius="{radius*0.4}" length="0.05"/></geometry></collision>
  </link>
  <joint name="neck_joint" type="fixed"><parent link="baseLink"/><child link="neck"/><origin xyz="0 0 0"/></joint>
</robot>"""

# The Master Inventory List (24 Items)
inventory = {
    # The Boxes (Plastic - Solid)
    "00_Red_Box": ("box", [0.9, 0.0, 0.0, 1.0], [0.12, 0.08, 0.28]), # Pure Bright Red
    "02_Blue_Box": ("box", [0.0, 0.0, 0.8, 1.0], [0.1, 0.1, 0.1]), # Pure Dark Blue
    "05_Orange_Box": ("box", [1.0, 0.4, 0.0, 1.0], [0.08, 0.08, 0.08]), # Bright Orange
    "06_Pink_Soap": ("box", [1.0, 0.6, 0.8, 1.0], [0.1, 0.04, 0.06]), # Light Baby Pink
    "08_Sky_Milk_Carton": ("box", [0.4, 0.8, 1.0, 1.0], [0.08, 0.08, 0.22]), # Changed to Light Sky Blue
    "11_Forest_Tea_Box": ("box", [0.0, 0.3, 0.0, 1.0], [0.14, 0.06, 0.18]), # Very Dark Green
    "14_Gold_Perfume": ("box", [0.8, 0.7, 0.0, 1.0], [0.06, 0.06, 0.12]), # Yellow/Gold
    "17_Peach_Juice": ("box", [1.0, 0.8, 0.6, 1.0], [0.09, 0.09, 0.25]), # Changed from Crimson to Peach

    # The Cans (Metal - Solid)
    "04_Purple_Can": ("can", [0.5, 0.0, 0.9, 1.0], [0.03, 0.12]), # Vibrant Violet
    "07_Black_Energy_Drink": ("can", [0.1, 0.1, 0.1, 1.0], [0.025, 0.16]), # Pure Black
    "10_Brown_Coffee_Jar": ("can", [0.4, 0.2, 0.1, 1.0], [0.05, 0.12]), # Dark Brown
    "13_Silver_Soup_Can": ("can", [0.8, 0.8, 0.8, 1.0], [0.04, 0.10]), # Metallic Silver
    "16_White_Deodorant": ("can", [0.95, 0.95, 0.95, 1.0], [0.02, 0.14]), # Changed from Teal to pure White
    "19_Bronze_Tuna_Tin": ("can", [0.6, 0.4, 0.2, 1.0], [0.045, 0.04]), # Changed from Grey to Bronze

    # The Bottles (Mixed Materials)
    "01_Sprite_Bottle": ("bottle", [0.1, 0.8, 0.1, 1.0], [0.0, 0.0, 0.5], 0.04, 0.14), # Bright Green
    "03_Yellow_Bottle": ("bottle", [0.9, 0.9, 0.1, 0.45], [0.1, 0.1, 0.8], 0.035, 0.10), # Translucent Yellow
    "09_Cyan_Shampoo": ("bottle", [0.0, 0.8, 0.8, 1.0], [0.9, 0.9, 0.9], 0.04, 0.18), # Bright Cyan/Turquoise
    "12_Magenta_Lotion": ("bottle", [0.8, 0.0, 0.4, 1.0], [0.1, 0.1, 0.1], 0.03, 0.15), # Deep Magenta
    "15_Navy_Body_Wash": ("bottle", [0.0, 0.0, 0.3, 1.0], [0.8, 0.8, 0.8], 0.045, 0.20), # Very Dark Blue
    "18_Olive_Oil": ("bottle", [0.4, 0.5, 0.1, 0.45], [0.1, 0.1, 0.1], 0.035, 0.22), # Translucent Olive
    
    # The 4 New Products
    "20_Coral_Shampoo": ("bottle", [1.0, 0.4, 0.3, 1.0], [0.9, 0.9, 0.9], 0.04, 0.18), # Changed to vibrant Coral
    "21_Neon_Energy_Drink": ("can", [0.2, 1.0, 0.2, 1.0], [0.025, 0.16]), # Changed from Red to Neon Green
    "22_Lime_Juice": ("box", [0.6, 0.9, 0.1, 1.0], [0.09, 0.09, 0.25]), # Changed from plain green to bright Lime
    "23_Clear_Water_Bottle": ("bottle", [0.8, 0.9, 1.0, 0.45], [0.1, 0.1, 0.8], 0.035, 0.20) # Translucent Icy Blue
}

print("[SYSTEM] Building 24 Asset URDFs with Transparency...")
for name, data in inventory.items():
    filepath = os.path.join("assets", f"{name}.urdf")
    
    if data[0] == "box":
        xml = generate_box_urdf(name, data[1][0], data[1][1], data[1][2], data[1][3], data[2][0], data[2][1], data[2][2])
    elif data[0] == "can":
        xml = generate_can_urdf(name, data[1][0], data[1][1], data[1][2], data[1][3], data[2][0], data[2][1])
    elif data[0] == "bottle":
        xml = generate_bottle_urdf(name, data[1][0], data[1][1], data[1][2], data[1][3], data[2][0], data[2][1], data[2][2], data[3], data[4])
        
    with open(filepath, "w") as f:
        f.write(xml)

print(f"[SYSTEM] Successfully generated {len(inventory)} items in the /assets folder!")
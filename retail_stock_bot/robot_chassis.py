import pybullet as p
import pybullet_data
import time
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import glob

global_inventory_log = {}

SESSION_ID = int(time.time())

# Create the folder to save our photos!
os.makedirs("scans", exist_ok=True)

COLOR_RED = [0.8, 0.1, 0.1, 1]
COLOR_GREEN = [0.1, 0.8, 0.1, 1]
COLOR_BLUE = [0.1, 0.1, 0.8, 1]
COLOR_YELLOW = [0.8, 0.8, 0.1, 1]
COLOR_PURPLE = [0.6, 0.1, 0.8, 1]
COLOR_ORANGE = [0.9, 0.5, 0.1, 1]

asset_files = sorted(glob.glob("assets/*.urdf"))
id2label = {i: os.path.basename(f).replace(".urdf", "") for i, f in enumerate(asset_files)}

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

def create_rack(name, x_center, facing_dir=1, y_start=-3.0, y_end=3.0):
    """Builds a retail Gondola Rack with 40cm depth and vertical dividers every 1 meter."""
    depth = 0.40  
    length = y_end - y_start
    thickness = 0.02
    
    white_color = [0.95, 0.95, 0.95, 1]
    darkgrey_color = [0.1, 0.1, 0.1, 1]
    
    # Solid Backboard
    backboard_thickness = 0.02
    backboard_height = 1.4
    bb_x = x_center - (facing_dir * (depth / 2)) + (facing_dir * (backboard_thickness / 2))
    
    bb_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[backboard_thickness/2, length/2, backboard_height/2], rgbaColor=white_color)
    bb_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[backboard_thickness/2, length/2, backboard_height/2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=bb_col, baseVisualShapeIndex=bb_vis, basePosition=[bb_x, 0, backboard_height/2])

    # Shelves and Red Edge Strips
    shelf_heights = [0.2, 0.6, 1.0]
    
    for height in shelf_heights:
        shelf_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[depth/2, length/2, thickness/2], rgbaColor=white_color)
        shelf_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[depth/2, length/2, thickness/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shelf_col, baseVisualShapeIndex=shelf_vis, basePosition=[x_center, 0, height])
        
        strip_depth = 0.01
        strip_height = 0.03
        strip_x = x_center + (facing_dir * (depth / 2)) - (facing_dir * (strip_depth / 2))
        
        strip_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[strip_depth/2, length/2, strip_height/2], rgbaColor=darkgrey_color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=strip_vis, basePosition=[strip_x, 0, height])

    # --- NEW: Vertical Dividers Every 1 Meter ---
    for y_pos in range(int(y_start), int(y_end) + 1):
        div_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[depth/2, thickness/2, backboard_height/2], rgbaColor=white_color)
        div_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[depth/2, thickness/2, backboard_height/2])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=div_col, baseVisualShapeIndex=div_vis, basePosition=[x_center, y_pos, backboard_height/2])

def populate_rack(x_center, facing_dir, assigned_products):
    """Fills 1-meter bays tightly (15cm gaps) with exactly one assigned product per bay."""
    shelf_heights = [0.2, 0.6, 1.0]
    
    for height in shelf_heights:
        current_y = -2.9 # Start at the far edge of the 6-meter rack
        
        while current_y < 2.9:
            # Skip vertical dividers
            if abs(round(current_y) - current_y) < 0.06:
                current_y += 0.05
                continue

            # Calculate which 1-meter bay we are currently in (Zone 0 through 5)
            # Because current_y goes from -3.0 to +3.0, adding 3.0 scales it from 0.0 to 6.0
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
                    
                    # IF IN DATA_GENERATOR.PY, UNCOMMENT THIS LINE:
                    # global_inventory_log[body_id] = selected_class_id
                    
                    # Apply the material finish dynamically
                    p.changeVisualShape(body_id, linkIndex=-1, specularColor=SPECULAR_MAP.get(selected_class_id, [0.5, 0.5, 0.5]))
                    
                    # If it's a compound bottle, ensure the neck/cap gets the finish too
                    if "bottle" in selected_urdf:
                        p.changeVisualShape(body_id, linkIndex=0, specularColor=SPECULAR_MAP.get(selected_class_id, [0.5, 0.5, 0.5]))
            
            # Move 15cm down the shelf to pack the next item tightly!
            current_y += 0.15

def build_store_layout():
    """Builds your complete 4-rack layout."""
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

def draw_route(waypoints):
    """Draws a visual red line on the floor to show the robot's planned route."""
    for i in range(len(waypoints) - 1):
        start_pt = [waypoints[i][0], waypoints[i][1], 0.05]
        end_pt = [waypoints[i+1][0], waypoints[i+1][1], 0.05]
        p.addUserDebugLine(start_pt, end_pt, lineColorRGB=[1, 0, 0], lineWidth=3)
    
    p.addUserDebugText("START", [waypoints[0][0], waypoints[0][1], 0.2], textColorRGB=[0,1,0], textSize=1.5)
    p.addUserDebugText("END", [waypoints[-1][0], waypoints[-1][1], 0.2], textColorRGB=[1,0,0], textSize=1.5)

def generate_differential_urdf():
    """Generates the robot chassis with highly elevated camera lenses (0.5m, 0.9m, 1.3m)."""
    urdf_content = """<?xml version="1.0"?>
    <robot name="sketch_bot_v5_premium">
        
        <link name="base_link">
            <visual><origin xyz="0 0 0.14"/><geometry><cylinder radius="0.22" length="0.13"/></geometry><material name="matte_black"><color rgba="0.15 0.15 0.15 1"/></material></visual>
            <collision><origin xyz="0 0 0.14"/><geometry><cylinder radius="0.22" length="0.13"/></geometry></collision>
            <inertial><origin xyz="0 0 0.14"/><mass value="25"/><inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.3"/></inertial>
        </link>

        <link name="base_accent">
            <visual><origin xyz="0 0 0.205"/><geometry><cylinder radius="0.222" length="0.01"/></geometry><material name="cyan_led"><color rgba="0.0 0.8 1.0 1"/></material></visual>
        </link>
        <joint name="accent_joint" type="fixed"><parent link="base_link"/><child link="base_accent"/><origin xyz="0 0 0"/></joint>

        <link name="main_tower">
            <visual><origin xyz="0 0 0.85"/><geometry><box size="0.2 0.1 1.25"/></geometry><material name="gloss_white"><color rgba="0.95 0.95 0.95 1"/></material></visual>
            <collision><origin xyz="0 0 0.85"/><geometry><box size="0.2 0.1 1.25"/></geometry></collision>
            <inertial><mass value="4"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/></inertial>
        </link>
        <joint name="tower_joint" type="fixed"><parent link="base_link"/><child link="main_tower"/><origin xyz="0 0 0"/></joint>

        <link name="sensor_fascia_right">
            <visual><origin xyz="0 -0.051 0.85"/><geometry><box size="0.18 0.005 1.15"/></geometry><material name="dark_glass"><color rgba="0.05 0.05 0.08 1"/></material></visual>
        </link>
        <joint name="fascia_right_joint" type="fixed"><parent link="main_tower"/><child link="sensor_fascia_right"/><origin xyz="0 0 0"/></joint>

        <link name="cam_R_low"><visual><geometry><box size="0.01 0.01 0.01"/></geometry><material name="cyan_led"/></visual></link>
        <joint name="cam_R_low_joint" type="fixed"><parent link="main_tower"/><child link="cam_R_low"/><origin xyz="0 -0.052 0.58"/></joint>

        <link name="cam_R_high"><visual><geometry><box size="0.01 0.01 0.01"/></geometry><material name="cyan_led"/></visual></link>
        <joint name="cam_R_high_joint" type="fixed"><parent link="main_tower"/><child link="cam_R_high"/><origin xyz="0 -0.052 1.38"/></joint>

        <link name="sensor_fascia_left">
            <visual><origin xyz="0 0.051 0.85"/><geometry><box size="0.18 0.005 1.15"/></geometry><material name="dark_glass"/></visual>
        </link>
        <joint name="fascia_left_joint" type="fixed"><parent link="main_tower"/><child link="sensor_fascia_left"/><origin xyz="0 0 0"/></joint>

        <link name="cam_L_low"><visual><geometry><box size="0.01 0.01 0.01"/></geometry><material name="cyan_led"/></visual></link>
        <joint name="cam_L_low_joint" type="fixed"><parent link="main_tower"/><child link="cam_L_low"/><origin xyz="0 0.052 0.58"/></joint>

        <link name="cam_L_high"><visual><geometry><box size="0.01 0.01 0.01"/></geometry><material name="cyan_led"/></visual></link>
        <joint name="cam_L_high_joint" type="fixed"><parent link="main_tower"/><child link="cam_L_high"/><origin xyz="0 0.052 1.38"/></joint>

        <link name="left_wheel">
            <visual><geometry><cylinder radius="0.08" length="0.05"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/><material name="wheel_dark"><color rgba="0.08 0.08 0.08 1"/></material></visual>
            <collision><geometry><cylinder radius="0.08" length="0.05"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/></collision>
            <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        </link>
        <joint name="left_motor" type="continuous"><parent link="base_link"/><child link="left_wheel"/><origin xyz="0 0.15 0.08"/><axis xyz="0 1 0"/></joint>

        <link name="right_wheel">
            <visual><geometry><cylinder radius="0.08" length="0.05"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/><material name="wheel_dark"/></visual>
            <collision><geometry><cylinder radius="0.08" length="0.05"/></geometry><origin xyz="0 0 0" rpy="1.5708 0 0"/></collision>
            <inertial><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
        </link>
        <joint name="right_motor" type="continuous"><parent link="base_link"/><child link="right_wheel"/><origin xyz="0 -0.15 0.08"/><axis xyz="0 1 0"/></joint>

        <link name="front_caster">
            <visual><geometry><sphere radius="0.04"/></geometry><origin xyz="0 0 0"/><material name="grey"><color rgba="0.4 0.4 0.4 1"/></material></visual>
            <collision><geometry><sphere radius="0.04"/></geometry><origin xyz="0 0 0"/></collision>
        </link>
        <joint name="front_caster_joint" type="fixed"><parent link="base_link"/><child link="front_caster"/><origin xyz="0.15 0 0.04"/></joint>
        
        <link name="rear_caster">
            <visual><geometry><sphere radius="0.04"/></geometry><origin xyz="0 0 0"/><material name="grey"/></visual>
            <collision><geometry><sphere radius="0.04"/></geometry><origin xyz="0 0 0"/></collision>
        </link>
        <joint name="rear_caster_joint" type="fixed"><parent link="base_link"/><child link="rear_caster"/><origin xyz="-0.15 0 0.04"/></joint>
    </robot>
    """
    with open("sketch_bot_v5.urdf", "w") as f:
        f.write(urdf_content)
    return "sketch_bot_v5.urdf"

# --- UPDATED: CAMERA PIPELINE WITH YOLO ANNOTATIONS ---
def capture_aisle_scan(robot_id, joint_map, scan_number, session_id="live"):
    """Snaps pictures with asymmetrical FOV for Live AI Inference."""
    print(f"[VISION] Snapping Scan #{scan_number} (4 Cameras)...")
    
    cameras = [
        ("Left_Low", "cam_L_low_joint", math.pi/2),
        ("Left_High", "cam_L_high_joint", math.pi/2),
        ("Right_Low", "cam_R_low_joint", -math.pi/2),
        ("Right_High", "cam_R_high_joint", -math.pi/2)
    ]
    
    _, orn = p.getBasePositionAndOrientation(robot_id)
    _, _, robot_yaw = p.getEulerFromQuaternion(orn)
    
    for cam_name, joint_name, angle_offset in cameras:
        cam_state = p.getLinkState(robot_id, joint_map[joint_name])
        cam_pos = cam_state[0]
        
        look_yaw = robot_yaw + angle_offset
        target_pos = [
            cam_pos[0] + math.cos(look_yaw),
            cam_pos[1] + math.sin(look_yaw),
            cam_pos[2]
        ]
        
        view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=target_pos, cameraUpVector=[0, 0, 1])
        
        cam_fov = 73.73  
        proj_matrix = p.computeProjectionMatrixFOV(fov=cam_fov, aspect=1.33, nearVal=0.01, farVal=3.0)
        
        # FAST RENDERING: We removed the ER_SEGMENTATION_MASK flag to speed up the simulation!
        width, height, rgb_img, _, _ = p.getCameraImage(
            480, 360, 
            viewMatrix=view_matrix, 
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL 
        )
        
        # 1. Save the Image for the AI to analyze
        np_img = np.array(rgb_img, dtype=np.uint8).reshape((height, width, 4))
        filename_prefix = f"scans/run_{session_id}_scan{scan_number}_{cam_name}"
        plt.imsave(f"{filename_prefix}.png", np_img)

def run_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.resetDebugVisualizerCamera(cameraDistance=6.5, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0.9, 0, 0])
    p.loadURDF("plane.urdf")
    
    build_store_layout()
    
    route_waypoints = [
        [-0.0475, -3.5], 
        [-0.0475, 3.7],  
        [0.9, 4.5],  
        [1.8650, 3.7],  
        [1.8650, -3.7]  
    ]
    draw_route(route_waypoints)
    
    urdf_path = generate_differential_urdf()
    start_orientation = p.getQuaternionFromEuler([0, 0, 1.5708])
    robot_id = p.loadURDF(urdf_path, basePosition=[route_waypoints[0][0], route_waypoints[0][1], 0.02], baseOrientation=start_orientation)
    
    joint_map = {p.getJointInfo(robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(robot_id))}
    p.changeDynamics(robot_id, joint_map["front_caster_joint"], lateralFriction=0.0)
    p.changeDynamics(robot_id, joint_map["rear_caster_joint"], lateralFriction=0.0)
    
    print("[SYSTEM] Autonomous Routing & Smart 1-Meter Vision Engine Engaged.")
    
    current_waypoint_idx = 1
    scan_count = 1
    
    # ODOMETRY TRACKING
    last_scan_pos = [route_waypoints[0][0], route_waypoints[0][1]]
    return_aisle_reset = False
    
    try:
        while p.isConnected():
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            curr_x, curr_y = pos[0], pos[1]
            _, _, yaw = p.getEulerFromQuaternion(orn)
            
            # --- ODOMETRY SNAPPING FOR RETURN AISLE ---
            if current_waypoint_idx == 4 and not return_aisle_reset:
                if curr_y <= 3.5:
                    last_scan_pos = [curr_x, 3.5] 
                    return_aisle_reset = True
            
            # --- THE OPTIMIZED 1.0 METER CAMERA TRIGGER ---
            distance_since_last_scan = math.sqrt((curr_x - last_scan_pos[0])**2 + (curr_y - last_scan_pos[1])**2)
            
            in_scanning_zone = -2.9 <= curr_y <= 2.9
            
            if distance_since_last_scan >= 1.0:
                if in_scanning_zone:
                    p.setJointMotorControl2(robot_id, joint_map["left_motor"], p.VELOCITY_CONTROL, targetVelocity=0, force=300)
                    p.setJointMotorControl2(robot_id, joint_map["right_motor"], p.VELOCITY_CONTROL, targetVelocity=0, force=300)
                    p.stepSimulation() 
                    
                    capture_aisle_scan(robot_id, joint_map, scan_count)
                    scan_count += 1
                
                last_scan_pos = [curr_x, curr_y]
            
            # --- DRIVING LOGIC ---
            if current_waypoint_idx < len(route_waypoints):
                target_x, target_y = route_waypoints[current_waypoint_idx]
                
                dx = target_x - curr_x
                dy = target_y - curr_y
                distance = math.sqrt(dx**2 + dy**2)
                target_angle = math.atan2(dy, dx)
                
                angle_error = target_angle - yaw
                while angle_error > math.pi: angle_error -= 2*math.pi
                while angle_error < -math.pi: angle_error += 2*math.pi
                
                # --- FASTER DRIVING LOGIC ---
                # Doubled the turn multiplier and max turn speed
                v_turn = 8.0 * angle_error 
                v_turn = max(min(v_turn, 4.0), -4.0)
                
                if distance < 0.2:
                    current_waypoint_idx += 1
                else:
                    if abs(angle_error) > 0.2:
                        v_drive = 2.0  # Increased speed while turning (was 1.0)
                    else:
                        v_drive = 8.0  # Doubled top straight-line speed (was 4.0)
                
                # Increased motor force from 150 to 300 to ensure fast acceleration
                p.setJointMotorControl2(robot_id, joint_map["left_motor"], p.VELOCITY_CONTROL, targetVelocity=(v_drive - v_turn), force=300)
                p.setJointMotorControl2(robot_id, joint_map["right_motor"], p.VELOCITY_CONTROL, targetVelocity=(v_drive + v_turn), force=300)
            else:
                p.setJointMotorControl2(robot_id, joint_map["left_motor"], p.VELOCITY_CONTROL, targetVelocity=0, force=300)
                p.setJointMotorControl2(robot_id, joint_map["right_motor"], p.VELOCITY_CONTROL, targetVelocity=0, force=300)

            p.stepSimulation()
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        if os.path.exists("sketch_bot_v5.urdf"):
            os.remove("sketch_bot_v5.urdf")

if __name__ == "__main__":
    run_simulation()
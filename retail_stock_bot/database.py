import sqlite3
import glob
import os

# ==========================================
# 1. READ INVENTORY FROM ASSETS
# ==========================================
asset_files = sorted(glob.glob("assets/*.urdf"))
if not asset_files:
    print("[ERROR] No assets found. Run build_assets.py first.")
    exit()

# Extract the clean names (e.g., "00_Red_Box")
product_list = [os.path.basename(f).replace(".urdf", "") for f in asset_files]

# ==========================================
# 2. INITIALIZE SQLITE DATABASE
# ==========================================
db_name = "retail_store.db"
conn = sqlite3.connect(db_name)
cursor = conn.cursor()

print(f"[SYSTEM] Building SQLite Database: {db_name}")

# Drop tables if they already exist so we can start fresh
cursor.execute("DROP TABLE IF EXISTS inventory_scans")
cursor.execute("DROP TABLE IF EXISTS products")

# --- TABLE 1: PRODUCTS ---
cursor.execute("""
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    max_capacity INTEGER NOT NULL
)
""")

# --- TABLE 2: INVENTORY SCANS ---
cursor.execute("""
CREATE TABLE inventory_scans (
    scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    product_id INTEGER,
    detected_count INTEGER NOT NULL,
    missing_stock INTEGER NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
)
""")

# ==========================================
# 3. PRE-POPULATE PRODUCT DATA
# ==========================================
# We know Max Capacity is strictly 6 (3 shelves * 2 deep) for every 1-meter bay
MAX_CAPACITY = 36

for class_id, name in enumerate(product_list):
    cursor.execute("""
    INSERT INTO products (product_id, product_name, max_capacity)
    VALUES (?, ?, ?)
    """, (class_id, name, MAX_CAPACITY))

conn.commit()
conn.close()

print("[SYSTEM] Database built successfully!")
print(f"[SYSTEM] Loaded {len(product_list)} products with a strict max capacity of {MAX_CAPACITY}.")
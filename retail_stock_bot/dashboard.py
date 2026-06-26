from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import sqlite3
import os
import io
import csv
from dotenv import load_dotenv

# LangChain & AI Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# Import the Groq Connector for Llama-3
from langchain_groq import ChatGroq

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Securely load API keys from the hidden .env file
load_dotenv()

app = FastAPI()

# --- Bulletproof Database Finder ---
def find_database():
    """Finds the REAL database, ignoring any accidental blank ones."""
    paths_to_check = [
        "../retail_store.db",  # Look in the parent folder first
        "retail_store.db",     # Look in the current folder
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            try:
                # Test if it's the real DB by looking for our table
                conn = sqlite3.connect(path)
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM products LIMIT 1")
                conn.close()
                return os.path.abspath(path) # Found the real one!
            except sqlite3.OperationalError:
                # This is the empty ghost database, ignore it!
                conn.close()
                continue
                
    return os.path.abspath("retail_store.db") # Fallback

DB_PATH = find_database()

# --- SQLAlchemy on Windows hates backslashes. We must convert them to forward slashes! ---
formatted_db_path = DB_PATH.replace('\\', '/')

# Connect LangChain to our SQLite Database
db = SQLDatabase.from_uri(f"sqlite:///{formatted_db_path}")

# Switched to Meta's updated Llama 3.3 70B model via Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# --- FIX: Strict Prompt Engineering for Accurate SQL ---
CUSTOM_INSTRUCTIONS = """You are an expert AI Store Manager for a retail dashboard.
You are connected to a SQLite database with two tables: 'products' and 'inventory_scans'.

CRITICAL RULE FOR ACCURACY:
The 'inventory_scans' table contains HISTORICAL data of every scan the robot has ever done. 
Whenever a user asks about the "current" stock, "missing" items, or status of a product, you MUST ONLY look at the MOST RECENT scan for each product.
To find the most recent scan, you MUST join the tables using this exact logic:
AND i.scan_id = (SELECT MAX(scan_id) FROM inventory_scans WHERE product_id = p.product_id)

Do not sum up historical missing stock. Only return data based on the latest scan IDs.
Be friendly and concise in your final answer!
"""

# Create the SQL Agent (The "Brain") with our custom rules attached
agent_executor = create_sql_agent(
    llm, 
    db=db, 
    agent_type="zero-shot-react-description", 
    verbose=True,
    prefix=CUSTOM_INSTRUCTIONS
)

class ChatRequest(BaseModel):
    message: str

# ==========================================
# 2. DATA PROCESSING FUNCTIONS
# ==========================================
def get_inventory_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = """
    SELECT 
        p.product_name,
        p.max_capacity,
        i.detected_count,
        i.missing_stock,
        i.timestamp
    FROM products p
    LEFT JOIN inventory_scans i 
        ON p.product_id = i.product_id 
        AND i.scan_id = (
            SELECT MAX(scan_id) 
            FROM inventory_scans 
            WHERE product_id = p.product_id
        )
    ORDER BY p.product_id
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    inventory = []
    for row in rows:
        inventory.append({
            "name": row[0],
            "capacity": row[1],
            "stock": row[2] if row[2] is not None else "Awaiting Scan...",
            "missing": row[3],
            "timestamp": row[4] if row[4] else "No Data"
        })
    return inventory

# ==========================================
# 3. FASTAPI ENDPOINTS
# ==========================================
@app.get("/api/export")
def export_csv():
    """Generates a CSV file of the current inventory and downloads it to the user."""
    data = get_inventory_data()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Product Name", "Total Capacity", "Current Stock", "Items Missing", "Last Scanned"])
    
    for row in data:
        writer.writerow([row["name"], row["capacity"], row["stock"], row["missing"], row["timestamp"]])
        
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]), 
        media_type="text/csv", 
        headers={"Content-Disposition": "attachment; filename=inventory_report.csv"}
    )

@app.post("/api/chat")
def chat_with_bot(req: ChatRequest):
    """Takes user text, writes SQL, and returns the AI's answer."""
    
    # Gracefully catch missing Groq API keys
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "YOUR_GROQ_API_KEY_HERE":
        return {"reply": "⚠️ **Error:** Please add your real Groq API key to your hidden .env file!"}
        
    try:
        response = agent_executor.invoke({"input": req.message})
        return {"reply": response["output"]}
    except Exception as e:
        # Print exact error to terminal so we can debug it
        error_msg = str(e)
        print(f"\n[CHATBOT ERROR] {error_msg}\n")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Renders the beautiful HTML/JS frontend."""
    inventory = get_inventory_data()
    total_products = len(inventory)
    total_missing = sum(item["missing"] for item in inventory if isinstance(item["missing"], int))
    
    # Building the HTML string dynamically
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Retail Dashboard</title>
        <style>
            :root {{ --primary: #2563eb; --bg: #f8fafc; --text: #1e293b; --border: #e2e8f0; }}
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); display: flex; height: 100vh; }}
            
            /* Sidebar Chatbot */
            .sidebar {{ width: 350px; background: white; border-right: 1px solid var(--border); display: flex; flex-direction: column; }}
            .chat-header {{ padding: 20px; background: var(--primary); color: white; font-weight: bold; font-size: 1.2rem; }}
            .chat-history {{ flex-grow: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 15px; }}
            .message {{ padding: 12px; border-radius: 8px; max-width: 85%; line-height: 1.4; }}
            .msg-user {{ background: #eff6ff; align-self: flex-end; border-bottom-right-radius: 0; }}
            .msg-ai {{ background: #f1f5f9; align-self: flex-start; border-bottom-left-radius: 0; }}
            .chat-input-area {{ padding: 20px; border-top: 1px solid var(--border); display: flex; gap: 10px; }}
            input[type="text"] {{ flex-grow: 1; padding: 10px; border: 1px solid var(--border); border-radius: 6px; outline: none; }}
            button {{ background: var(--primary); color: white; border: none; padding: 10px 15px; border-radius: 6px; cursor: pointer; font-weight: bold; }}
            button:hover {{ opacity: 0.9; }}
            
            /* Main Content */
            .main-content {{ flex-grow: 1; padding: 40px; overflow-y: auto; }}
            .header-bar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }}
            .metrics {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; border: 1px solid var(--border); width: 250px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
            .card h3 {{ margin: 0 0 10px 0; color: #64748b; font-size: 1rem; }}
            .card p {{ margin: 0; font-size: 2rem; font-weight: bold; }}
            .alert-text {{ color: #ef4444; }}
            .good-text {{ color: #10b981; }}
            
            table {{ width: 100%; background: white; border-radius: 12px; overflow: hidden; border-collapse: collapse; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
            th, td {{ padding: 15px 20px; text-align: left; border-bottom: 1px solid var(--border); }}
            th {{ background: #f1f5f9; font-weight: 600; color: #475569; text-transform: uppercase; font-size: 0.85rem; }}
            .action-btn {{ background: #10b981; text-decoration: none; display: inline-block; }}
        </style>
    </head>
    <body>

        <div class="sidebar">
            <div class="chat-header">🤖 AI Store Manager</div>
            <div class="chat-history" id="chatHistory">
                <div class="message msg-ai">Hello! I am connected to your live inventory database. Ask me anything!</div>
            </div>
            <div class="chat-input-area">
                <input type="text" id="chatInput" placeholder="e.g., What items are missing?">
                <button onclick="sendMessage()">Ask</button>
            </div>
        </div>

        <div class="main-content">
            <div class="header-bar">
                <h1 style="margin: 0; color: #0f172a;">Live Inventory Control</h1>
                <div>
                    <a href="/" class="action-btn" style="background: var(--primary); padding: 10px 20px; color: white; border-radius: 6px; margin-right: 10px;">🔄 Refresh Table</a>
                    <a href="/api/export" class="action-btn" style="padding: 10px 20px; color: white; border-radius: 6px;">📥 Export CSV Report</a>
                </div>
            </div>
            
            <div class="metrics">
                <div class="card">
                    <h3>Tracked Products</h3>
                    <p>{total_products}</p>
                </div>
                <div class="card">
                    <h3>Total Missing Units</h3>
                    <p class="{'alert-text' if total_missing > 0 else 'good-text'}">{total_missing}</p>
                </div>
            </div>

            <table>
                <tr>
                    <th>Product Name</th>
                    <th>Capacity</th>
                    <th>Current Stock</th>
                    <th>Status</th>
                    <th>Last Scanned</th>
                </tr>
                {"".join(f'''
                <tr>
                    <td>{row["name"]}</td>
                    <td>{row["capacity"]}</td>
                    <td>{row["stock"]}</td>
                    <td class="{'alert-text' if isinstance(row["missing"], int) and row["missing"] > 0 else 'good-text' if row["missing"] == 0 else ''}">
                        {f"{row['missing']} Missing!" if isinstance(row['missing'], int) and row['missing'] > 0 else "Fully Stocked" if row['missing'] == 0 else "Awaiting Scan"}
                    </td>
                    <td style="color: #64748b; font-size: 0.9rem;">{row["timestamp"]}</td>
                </tr>
                ''' for row in inventory)}
            </table>
        </div>

        <script>
            async function sendMessage() {{
                const inputField = document.getElementById('chatInput');
                const chatHistory = document.getElementById('chatHistory');
                const userText = inputField.value.trim();
                
                if (!userText) return;
                
                // Add user message to UI
                chatHistory.innerHTML += `<div class="message msg-user">${{userText}}</div>`;
                inputField.value = '';
                chatHistory.scrollTop = chatHistory.scrollHeight;
                
                // Add loading indicator
                const loadingId = "loading-" + Date.now();
                chatHistory.innerHTML += `<div id="${{loadingId}}" class="message msg-ai" style="opacity: 0.5;">Thinking...</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;

                try {{
                    const response = await fetch('/api/chat', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ message: userText }})
                    }});
                    const data = await response.json();
                    
                    if (response.ok) {{
                        document.getElementById(loadingId).outerHTML = `<div class="message msg-ai">${{data.reply}}</div>`;
                    }} else {{
                        document.getElementById(loadingId).outerHTML = `<div class="message msg-ai alert-text"><b>System Error:</b><br>${{data.detail}}</div>`;
                    }}
                    
                }} catch (error) {{
                    document.getElementById(loadingId).outerHTML = `<div class="message msg-ai alert-text">Error connecting to AI. Check terminal.</div>`;
                }}
                chatHistory.scrollTop = chatHistory.scrollHeight;
            }}
            
            // Allow pressing Enter to send
            document.getElementById('chatInput').addEventListener('keypress', function (e) {{
                if (e.key === 'Enter') sendMessage();
            }});
        </script>
    </body>
    </html>
    """
    return html_content
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
import cv2
import time

from database import get_connection
import ai_engine

app = FastAPI(title="AI Document Locker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads/docs", exist_ok=True)
os.makedirs("uploads/fingerprints", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def startup_event():
    ai_engine.load_ai_models()

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

@app.get("/dashboard")
def read_dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/register")
def read_register():
    return FileResponse("static/register.html")

@app.get("/access")
def read_access():
    return FileResponse("static/access.html")

@app.get("/history")
def read_history():
    return FileResponse("static/history.html")

@app.get("/register_center")
def read_register_center():
    return FileResponse("static/register_center.html")

# ======================= CENTER & DEVICE APIS =======================

@app.post("/api/register_center")
def register_center(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO service_centers (name, email, password_hash) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        success = True
        msg = "Center successfully registered."
    except sqlite3.IntegrityError:
        success = False
        msg = "Email already exists."
    conn.close()
    return {"success": success, "msg": msg}

@app.post("/api/login")
def login(email: str = Form(...), password: str = Form(...)):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name FROM service_centers WHERE email = ? AND password_hash = ?", (email, password))
    user = c.fetchone()
    conn.close()
    if user:
        return {"success": True, "center_id": user["id"], "center_name": user["name"]}
    return {"success": False, "msg": "Invalid credentials"}

@app.post("/api/add_device")
def add_device(center_id: int = Form(...), name: str = Form(...), ip: str = Form(...)):
    conn = get_connection()
    c = conn.cursor()
    c.execute("INSERT INTO devices (center_id, name, ip_address) VALUES (?, ?, ?)", (center_id, name, ip))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/get_devices/{center_id}")
def get_devices(center_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM devices WHERE center_id = ?", (center_id,))
    devices = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"success": True, "devices": devices}

@app.delete("/api/remove_device/{device_id}")
def remove_device(device_id: int):
    conn = get_connection()
    c = conn.cursor()
    # Detach logs instead of deleting them to preserve history
    c.execute("UPDATE access_logs SET device_id = NULL WHERE device_id = ?", (device_id,))
    c.execute("DELETE FROM devices WHERE id = ?", (device_id,))
    conn.commit()
    conn.close()
    return {"success": True}

# ======================= AI HARDWARE INTERAACTION =======================

@app.post("/api/scan_registration")
def scan_registration(ip: str = Form(...)):
    """
    Trigger 3-scan voting loop. Blocks until hardware returns images or timeouts.
    """
    b_val, g_val, best_img = ai_engine.execute_unified_voting(ip, scans=3)
    if best_img is None:
        return {"success": False, "msg": "Failed to communicate with sensor or capture fingerprint."}
        
    # Save temp raw image for registration form
    temp_path = f"uploads/fingerprints/temp_{int(time.time())}.png"
    cv2.imwrite(temp_path, best_img)
    
    # Send display completion message
    ai_engine.poll_esp32_single_scan(ip, 8080, custom_msg="Scan Complete!")
    
    return {
        "success": True, 
        "blood_group": b_val, 
        "gender": g_val,
        "temp_fingerprint_path": temp_path
    }

# ======================= USER REGISTRATION =======================

from typing import List

@app.post("/api/register_user")
def register_user(
    name: str = Form(...),
    email: str = Form(""),
    phone: str = Form(""),
    address: str = Form(""),
    blood_group: str = Form(...),
    gender: str = Form(...),
    temp_fingerprint_path: str = Form(...),
    doc_names: List[str] = Form(default=[]),
    doc_files: List[UploadFile] = File(default=[])
):
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Move temp fingerprint to permanent location
    final_fp_path = temp_fingerprint_path.replace("temp_", "user_")
    if os.path.exists(temp_fingerprint_path):
        os.rename(temp_fingerprint_path, final_fp_path)
    else:
        return {"success": False, "msg": "Fingerprint temp file missing! Did you scan?"}
        
    # 2. Database User insertion
    c.execute('''
        INSERT INTO users (name, email, phone, address, blood_group, gender, fingerprint_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, email, phone, address, blood_group, gender, final_fp_path))
    user_id = c.lastrowid
    
    # 3. Documents (Dynamic)
    for index, file_obj in enumerate(doc_files):
        if file_obj and file_obj.filename:
            doc_type = doc_names[index] if index < len(doc_names) else f"Document_{index+1}"
            file_path = f"uploads/docs/{user_id}_{doc_type}_{file_obj.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_obj.file, buffer)
            c.execute("INSERT INTO documents (user_id, document_type, file_path) VALUES (?, ?, ?)", 
                      (user_id, doc_type, file_path))
                      
    conn.commit()
    conn.close()
    return {"success": True, "msg": "User successfully registered!"}

# ======================= ACCESS DOCS FLOW =======================

def load_all_descriptors():
    # Cache all feature descriptors mapping user_id -> desc
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, fingerprint_path FROM users")
    rows = c.fetchall()
    conn.close()
    
    desc_map = {}
    for r in rows:
        path = r["fingerprint_path"]
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, desc = ai_engine.extract_features(img)
                desc_map[r["id"]] = desc
    return desc_map

@app.post("/api/access_docs")
def access_docs(ip: str = Form(...), center_id: int = Form(...), device_id: int = Form(...)):
    # 1. Poll Device specifically once
    img = ai_engine.poll_esp32_single_scan(ip, 8080, custom_msg="Authenticating..")
    if img is None:
        return {"success": False, "msg": "Device unreachable or scan failed."}
        
    # 2. Match Fingerprint
    desc_map = load_all_descriptors()
    matched_id = ai_engine.compare_raw_image_to_descriptors(img, desc_map)
    
    if matched_id:
        ai_engine.poll_esp32_single_scan(ip, 8080, custom_msg="Access Granted!")
        
        # Pull User info
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id = ?", (matched_id,))
        user = dict(c.fetchone())
        
        c.execute("SELECT * FROM documents WHERE user_id = ?", (matched_id,))
        docs = [dict(row) for row in c.fetchall()]
        
        # Log History
        c.execute("INSERT INTO access_logs (user_id, center_id, device_id, action_type, details) VALUES (?, ?, ?, ?, ?)", 
                  (matched_id, center_id, device_id, 'PROFILE_ACCESS', 'Authenticated via SIFT Fingerprint'))
        conn.commit()
        conn.close()
        
        return {"success": True, "user": user, "documents": docs}
    else:
        ai_engine.poll_esp32_single_scan(ip, 8080, custom_msg="Access Denied")
        return {"success": False, "msg": "No matching user found in database."}

@app.get("/api/download_doc")
def download_doc(path: str, user_id: int = -1, center_id: int = -1, device_id: int = -1):
    if os.path.exists(path):
        if user_id != -1 and center_id != -1 and device_id != -1:
            conn = get_connection()
            c = conn.cursor()
            doc_type = path.split('_')[-2] if '_' in path else 'Document'
            c.execute("INSERT INTO access_logs (user_id, center_id, device_id, action_type, details) VALUES (?, ?, ?, ?, ?)", 
                      (user_id, center_id, device_id, 'DOWNLOADED_DOC', doc_type))
            conn.commit()
            conn.close()
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File missing")

@app.delete("/api/delete_doc/{doc_id}")
def delete_doc(doc_id: int, user_id: int, center_id: int, device_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT file_path, document_type FROM documents WHERE id = ?", (doc_id,))
    doc = c.fetchone()
    if doc:
        if os.path.exists(doc['file_path']):
            os.remove(doc['file_path'])
        c.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        c.execute("INSERT INTO access_logs (user_id, center_id, device_id, action_type, details) VALUES (?, ?, ?, ?, ?)", 
                  (user_id, center_id, device_id, 'DELETED_DOC', doc['document_type']))
        conn.commit()
    conn.close()
    return {"success": True}

@app.post("/api/upload_doc")
def upload_doc(
    user_id: int = Form(...), center_id: int = Form(...), device_id: int = Form(...),
    doc_name: str = Form(...), doc_file: UploadFile = File(...)
):
    conn = get_connection()
    c = conn.cursor()
    if doc_file and doc_file.filename:
        file_path = f"uploads/docs/{user_id}_{doc_name}_{doc_file.filename}"
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(doc_file.file, buffer)
        
        c.execute("INSERT INTO documents (user_id, document_type, file_path) VALUES (?, ?, ?)", 
                  (user_id, doc_name, file_path))
        c.execute("INSERT INTO access_logs (user_id, center_id, device_id, action_type, details) VALUES (?, ?, ?, ?, ?)", 
                  (user_id, center_id, device_id, 'UPLOADED_DOC', doc_name))
        conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/history")
def get_history():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        SELECT a.id, u.name as user_name, c.name as center_name, 
               COALESCE(d.name, 'Removed Device') as device_name, 
               a.action_type, a.details, a.timestamp 
        FROM access_logs a
        JOIN users u ON a.user_id = u.id
        JOIN service_centers c ON a.center_id = c.id
        LEFT JOIN devices d ON a.device_id = d.id
        ORDER BY a.timestamp DESC
    ''')
    logs = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"success": True, "logs": logs}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

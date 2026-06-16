import sqlite3
import os

DB_PATH = "database.db"

def get_connection():
    # Enforces foreign key constraints mapping
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = 1")
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Service Centers
    c.execute('''
        CREATE TABLE IF NOT EXISTS service_centers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    # 2. Devices
    c.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            center_id INTEGER,
            ip_address TEXT,
            name TEXT,
            FOREIGN KEY(center_id) REFERENCES service_centers(id)
        )
    ''')
    
    # 3. Users (Hold biometric & AI metadata)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            blood_group TEXT,
            gender TEXT,
            fingerprint_path TEXT 
        )
    ''')
    
    # 4. Documents
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            document_type TEXT,
            file_path TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # 5. Access Logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            center_id INTEGER,
            device_id INTEGER,
            action_type TEXT DEFAULT 'PROFILE_ACCESS',
            details TEXT DEFAULT '',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(center_id) REFERENCES service_centers(id),
            FOREIGN KEY(device_id) REFERENCES devices(id)
        )
    ''')
    
    conn.commit()
    
    # Seed a default admin if none exists
    c.execute("SELECT COUNT(*) FROM service_centers")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO service_centers (name, email, password_hash) VALUES (?, ?, ?)", 
                  ("Default Service Center", "admin@service.com", "admin")) # In a real app we'd hash the password
        conn.commit()
        
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized.")

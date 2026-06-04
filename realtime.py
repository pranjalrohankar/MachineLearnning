# backend/app.py
from flask import Flask, jsonify, request
import sqlite3
import datetime
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("attacks.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS attack_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            attack_type TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Fetch recent attack logs
@app.route("/api/logs", methods=["GET"])
def get_logs():
    conn = sqlite3.connect("attacks.db")
    c = conn.cursor()
    c.execute("SELECT timestamp, attack_type FROM attack_logs ORDER BY id DESC LIMIT 10")
    logs = [{"timestamp": row[0], "attack_type": row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify({"logs": logs})

# Simulate real-time attack detection
@app.route("/api/scan", methods=["POST"])
def scan_system():
    attack_types = ["DDoS", "Brute Force", "SQL Injection", "Phishing", "Normal"]
    detected_attack = random.choice(attack_types)

    # Store in database
    conn = sqlite3.connect("attacks.db")
    c = conn.cursor()
    c.execute("INSERT INTO attack_logs (timestamp, attack_type) VALUES (?, ?)", 
              (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), detected_attack))
    conn.commit()
    conn.close()

    return jsonify({"message": f"Scan Complete: {detected_attack} detected!"})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=8000)

from flask import Flask, jsonify, request, render_template, session, redirect, url_for, flash, g
from flask_cors import CORS
from threading import Thread
from scapy.all import sniff, IP, TCP, UDP, conf, get_if_list, ARP, Ether
from collections import defaultdict
import time
import joblib
import warnings
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import logging
from urllib.parse import urlparse
import re
from sklearn.linear_model import LogisticRegression
from functools import wraps
import sqlite3
import hashlib
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
from flask_login import LoginManager, login_required
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import base64
import io
from scipy.io import arff
import json
import random
import whois
import requests
from bs4 import BeautifulSoup
import absl.logging
import google.generativeai as genai
import threading
import ipaddress
from threading import Lock
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.info("Starting application...")

# Suppress gRPC and absl logs
os.environ["GRPC_VERBOSITY"] = "ERROR"
absl.logging.set_verbosity(absl.logging.ERROR)

# Configure Gemini API
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, '.env'))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB0JgjoBhJvpXMw7Ti5kJRUEdDPAnXcMQE")
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your_secret_key")
CORS(app)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================== ARP SPOOFING DETECTION ==================
# Load the pre-trained ARP spoofing model
arp_model = joblib.load(os.path.join(BASE_DIR, 'arp_spoofing_model.pkl'))

# Feature names expected by the ARP model
arp_feature_names = [
    'protocol', 'ip_version', 'bidirectional_packets', 'bidirectional_bytes',
    'src2dst_packets', 'src2dst_bytes', 'dst2src_packets', 'dst2src_bytes',
    'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
    'bidirectional_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms',
    'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms'
]

# Data structures to track network flows and alerts
flow_stats = defaultdict(lambda: {
    'packet_count': 0,
    'byte_count': 0,
    'packet_sizes': [],
    'timestamps': [],
    'src_packets': 0,
    'src_bytes': 0,
    'dst_packets': 0,
    'dst_bytes': 0
})

arp_alerts = []
arp_sniffing_thread = None
arp_analysis_thread = None
arp_sniffing_active = False
target_network = None
arp_thread_lock = Lock()

# ================== SYN FLOOD DETECTION ==================
# Global variables for SYN flood detection
syn_counts = defaultdict(int)
syn_ack_counts = defaultdict(int)
syn_detection_active = False
syn_flood_alerts = []
syn_target_ip = ""

# ================== PORT SCAN DETECTION ==================
# Port Scan Detection variables
connection_stats = defaultdict(lambda: {
    'count': 0,
    'duration': 0,
    'src_bytes': 0,
    'dst_bytes': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'hot': 0,
    'num_failed_logins': 0,
    'logged_in': 0,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'last_timestamp': None
})

alerts = []
portscan_monitoring_active = False
current_interface = None
TIME_WINDOW = 60  # seconds

# ================== MODEL CONFIGURATION ==================
# Model accuracies and descriptions
model_accuracies = {
    'Phishing Detection - Gradient Boosting': 96.2,
    'Malware Detection - Random Forest': 93.5,
    'XSS Detection - Gradient Boosting': 94.8,
    
    'DoS Detection - Gradient Boosting': 96.8,
    'DoS Detection - KNN': 95.3,
    'SYN Flood Detection': 92.0,
    'Port Scan Detection': 91.5,
    'ARP Spoofing Detection': 90.2
}

model_descriptions = {
    'Phishing Detection - Gradient Boosting': "Analyzes 79 URL and website features to detect phishing attempts with high accuracy.",
    'Malware Detection - Random Forest': "Detects various malware attack types by analyzing network packet features.",
    'XSS Detection - Gradient Boosting': "Identifies potential Cross-Site Scripting attacks in text inputs.",
   
    'DoS Detection - Gradient Boosting': "Alternative DoS detection model with slightly different characteristics.",
    'DoS Detection - KNN': "K-Nearest Neighbors model for DoS detection, useful for comparative analysis.",
    'SYN Flood Detection': "Monitors TCP SYN packets to detect potential SYN flood attacks in real-time.",
    'Port Scan Detection': "Identifies port scanning activities by analyzing connection patterns.",
    'ARP Spoofing Detection': "Detects ARP spoofing attacks by analyzing network traffic patterns."
}

# UPLOAD_FOLDER configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

models = {}
label_encoders = {}
scaler = None
# ================== MODEL LOADING ==================
try:
    # Load Port Scan model files
    portscan_model = joblib.load(os.path.join(BASE_DIR, 'portscan_model.pkl'))
    portscan_scaler = joblib.load(os.path.join(BASE_DIR, 'portscan_scaler.pkl'))
    portscan_features = joblib.load(os.path.join(BASE_DIR, 'portscan_features.pkl'))
    print("✅ Port Scan model loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠️ Port Scan model files not found. Port Scan detection will be disabled. Error: {e}")
    portscan_model = None
    portscan_scaler = None
    portscan_features = None
except Exception as e:
    print(f"❌ Error loading Port Scan model: {e}")
    portscan_model = None
    portscan_scaler = None
    portscan_features = None

# Load the pre-trained phishing model
phishing_model = joblib.load(os.path.join(BASE_DIR, 'phishing_model.pkl'))

def map_to_main_category(attack_name):
    """Maps an attack type to its main category"""
    malware_categories = {
        'Denial-of-Service (DoS)': ['neptune', 'smurf', 'pod', 'teardrop', 'land'],
        'Remote-to-Local (R2L) Attacks': ['ftp_write', 'guess_passwd', 'imap', 'phf', 'spy', 'multihop', 'worm'],
        'User-to-Root (U2R) Attacks': ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'sqlattack', 'xlock'],
        'Probing (Reconnaissance)': ['satan', 'nmap', 'ipsweep', 'portsweep'],
        'Malware': ['back', 'rootkit', 'warezclient', 'warezmaster', 'xterm', 'httptunnel'],
        'normal': ['normal']
    }

    for category, attacks in malware_categories.items():
        if attack_name in attacks:
            return category
    return 'unknown'

# Load models
try:
    phishing_model = joblib.load(os.path.join(BASE_DIR, 'phishing_model.pkl'))
    malware_model = joblib.load(os.path.join(BASE_DIR, "malware_model.pkl"))
    malware_scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    malware_encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
    malware_label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    syn_flood_model = joblib.load(os.path.join(BASE_DIR, "syn_flood_model.pkl"))
    syn_flood_scaler = joblib.load(os.path.join(BASE_DIR, "syn_flood_scaler.pkl"))
    
    with open(os.path.join(BASE_DIR, "feature_names.json"), "r") as f:
        expected_feature_names = json.load(f)
    with open(os.path.join(BASE_DIR, "class_mapping.json"), "r") as f:
        malware_class_mapping = json.load(f)
    
    print("✅ All models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading model files: {e}")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ================== ARP SPOOFING FUNCTIONS ==================
def is_in_target_network(ip):
    """Check if an IP address belongs to the target network"""
    if target_network is None:
        return True  # Monitor all networks if none specified
    return ipaddress.ip_address(ip) in ipaddress.ip_network(target_network)

def arp_packet_handler(packet):
    """Process each network packet for ARP spoofing detection"""
    with arp_thread_lock:
        if ARP in packet:
            handle_arp_packet(packet)
            return
        
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            
            # Only process packets within target network
            if not (is_in_target_network(src_ip) or is_in_target_network(dst_ip)):
                return
                
            flow_key = (src_ip, dst_ip)
            
            flow_stats[flow_key]['packet_count'] += 1
            flow_stats[flow_key]['byte_count'] += len(packet)
            flow_stats[flow_key]['packet_sizes'].append(len(packet))
            flow_stats[flow_key]['timestamps'].append(time.time())
            flow_stats[flow_key]['src_packets'] += 1
            flow_stats[flow_key]['src_bytes'] += len(packet)

def handle_arp_packet(packet):
    """Handle ARP packets specifically"""
    global arp_alerts
    
    arp = packet[ARP]
    if arp.op == 2:  # ARP reply
        # Check if source IP is in target network
        if not is_in_target_network(arp.psrc):
            return
            
        features = {
            'protocol': 2054,
            'ip_version': 4,
            'bidirectional_packets': 1,
            'bidirectional_bytes': len(packet),
            'src2dst_packets': 1,
            'src2dst_bytes': len(packet),
            'dst2src_packets': 0,
            'dst2src_bytes': 0,
            'bidirectional_min_ps': len(packet),
            'bidirectional_mean_ps': len(packet),
            'bidirectional_stddev_ps': 0,
            'bidirectional_max_ps': len(packet),
            'bidirectional_min_piat_ms': 0,
            'bidirectional_mean_piat_ms': 0,
            'bidirectional_stddev_piat_ms': 0,
            'bidirectional_max_piat_ms': 0
        }
        
        df = pd.DataFrame([features], columns=arp_feature_names)
        prediction = arp_model.predict(df)
        proba = arp_model.predict_proba(df)[0][1]
        
        if prediction[0] == 1 and proba > 0.8:
            alert = {
                'type': 'ARP Spoofing Detected',
                'src_ip': arp.psrc,
                'dst_ip': 'Broadcast',
                'mac_address': arp.hwsrc,
                'probability': float(proba),
                'packets': 1,
                'bytes': len(packet),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'message': f"Suspicious ARP reply from {arp.psrc} ({arp.hwsrc}) claiming to be on network {target_network}"
            }
            arp_alerts.append(alert)

def analyze_flows():
    """Analyze all current flows for ARP spoofing"""
    global flow_stats, arp_alerts
    
    with arp_thread_lock:
        for flow_key, flow_data in flow_stats.items():
            features = calculate_flow_features(flow_data)
            if features:
                df = pd.DataFrame([features], columns=arp_feature_names)
                prediction = arp_model.predict(df)
                proba = arp_model.predict_proba(df)[0][1]
                
                if prediction[0] == 1 and proba > 0.8:
                    src_ip, dst_ip = flow_key
                    alert = {
                        'type': 'Flow Analysis',
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'probability': float(proba),
                        'packets': flow_data['packet_count'],
                        'bytes': flow_data['byte_count'],
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'message': f"Suspicious network flow detected between {src_ip} and {dst_ip}"
                    }
                    arp_alerts.append(alert)
        
        flow_stats.clear()

def calculate_flow_features(flow_data):
    """Calculate features for a network flow"""
    if len(flow_data['timestamps']) < 2:
        return None
    
    features = {
        'protocol': 6,  # Assuming TCP
        'ip_version': 4,  # IPv4
        'bidirectional_packets': flow_data['packet_count'],
        'bidirectional_bytes': flow_data['byte_count'],
        'src2dst_packets': flow_data['src_packets'],
        'src2dst_bytes': flow_data['src_bytes'],
        'dst2src_packets': flow_data['dst_packets'],
        'dst2src_bytes': flow_data['dst_bytes'],
        'bidirectional_min_ps': min(flow_data['packet_sizes']) if flow_data['packet_sizes'] else 0,
        'bidirectional_mean_ps': np.mean(flow_data['packet_sizes']) if flow_data['packet_sizes'] else 0,
        'bidirectional_stddev_ps': np.std(flow_data['packet_sizes']) if len(flow_data['packet_sizes']) > 1 else 0,
        'bidirectional_max_ps': max(flow_data['packet_sizes']) if flow_data['packet_sizes'] else 0,
        'bidirectional_min_piat_ms': min([(flow_data['timestamps'][i] - flow_data['timestamps'][i-1]) * 1000 
                                   for i in range(1, len(flow_data['timestamps']))]) 
                                   if len(flow_data['timestamps']) > 1 else 0,
        'bidirectional_mean_piat_ms': np.mean([(flow_data['timestamps'][i] - flow_data['timestamps'][i-1]) * 1000 
                                           for i in range(1, len(flow_data['timestamps']))])
                                           if len(flow_data['timestamps']) > 1 else 0,
        'bidirectional_stddev_piat_ms': np.std([(flow_data['timestamps'][i] - flow_data['timestamps'][i-1]) * 1000 
                                        for i in range(1, len(flow_data['timestamps']))])
                                        if len(flow_data['timestamps']) > 2 else 0,
        'bidirectional_max_piat_ms': max([(flow_data['timestamps'][i] - flow_data['timestamps'][i-1]) * 1000 
                                       for i in range(1, len(flow_data['timestamps']))])
                                       if len(flow_data['timestamps']) > 1 else 0
    }
    
    return features

def start_arp_sniffing():
    """Start network sniffing in a background thread for ARP detection"""
    global arp_sniffing_active, arp_sniffing_thread, arp_alerts
    
    arp_sniffing_active = True
    try:
        sniff(prn=arp_packet_handler, store=0, stop_filter=lambda x: not arp_sniffing_active)
    except Exception as e:
        logger.error(f"ARP Sniffing Error: {e}")
        arp_alerts.append({
            'type': 'Sniffing Error',
            'src_ip': 'N/A',
            'dst_ip': 'N/A',
            'mac_address': 'N/A',
            'probability': 0.0,
            'packets': 0,
            'bytes': 0,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'message': f"Failed to start packet sniffing: {str(e)}. Try running as root (sudo)."
        })
        arp_sniffing_active = False

def stop_arp_sniffing():
    """Stop ARP network sniffing and clean up threads"""
    global arp_sniffing_active, arp_sniffing_thread, arp_analysis_thread
    
    arp_sniffing_active = False
    
    # Wait for threads to finish if they exist
    if arp_sniffing_thread and arp_sniffing_thread.is_alive():
        arp_sniffing_thread.join(timeout=2)
    
    if arp_analysis_thread and arp_analysis_thread.is_alive():
        arp_analysis_thread.join(timeout=2)
    
    # Reset thread references
    arp_sniffing_thread = None
    arp_analysis_thread = None

def arp_background_analysis():
    """Background thread for periodic ARP analysis"""
    while arp_sniffing_active:
        analyze_flows()
        time.sleep(10)





# ================== SYN FLOOD DETECTION FUNCTIONS ==================
def extract_syn_features(packet):
    """Extract SYN flood features with more detailed metrics"""
    src_ip = packet[IP].src
    
    # Update counts based on packet type
    if packet[TCP].flags == 'S':  # SYN packet
        syn_counts[src_ip] += 1
    elif packet[TCP].flags == 'SA':  # SYN-ACK packet
        syn_ack_counts[src_ip] += 1

    # Calculate advanced metrics
    total_syn = sum(syn_counts.values())
    unique_sources = len(syn_counts)
    syn_rate = syn_counts.get(src_ip, 0)
    syn_ack_ratio = syn_ack_counts.get(src_ip, 0.1) / max(1, syn_counts.get(src_ip, 1))
    avg_packet_size = sum(len(p) for p in syn_counts) / max(1, total_syn) if total_syn > 0 else 0

    return {
        "syn_rate": syn_rate,
        "total_syn": total_syn,
        "unique_sources": unique_sources,
        "syn_ack_ratio": syn_ack_ratio,
        "avg_packet_size": avg_packet_size,
        "current_time": time.time()
    }

def detect_syn_flood(packet):
    """Enhanced SYN flood detection with better thresholds"""
    global syn_flood_alerts
    
    if not (IP in packet and TCP in packet) or packet[IP].dst != syn_target_ip:
        return

    features = extract_syn_features(packet)
    
    # Dynamic thresholds based on traffic patterns
    syn_threshold = 100  # packets per second
    ratio_threshold = 0.2  # SYN-ACK/SYN ratio
    
    if (features['syn_rate'] > syn_threshold and 
        features['syn_ack_ratio'] < ratio_threshold):
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "src_ip": packet[IP].src,
            "syn_rate": features['syn_rate'],
            "syn_ack_ratio": f"{features['syn_ack_ratio']:.2f}",
            "total_syn": features['total_syn'],
            "unique_sources": features['unique_sources'],
            "confidence": "High" if features['syn_rate'] > 150 else "Medium"
        }
        
        # Prevent duplicate alerts for same source
        existing_alerts = [a for a in syn_flood_alerts if a['src_ip'] == alert['src_ip']]
        if not existing_alerts or (
            datetime.now() - datetime.fromisoformat(existing_alerts[-1]['timestamp'])
        ).total_seconds() > 10:
            syn_flood_alerts.append(alert)
            if len(syn_flood_alerts) > 20:  # Keep more alerts
                syn_flood_alerts.pop(0)

def reset_syn_counts():
    """Periodically reset counts while detection is active"""
    while syn_detection_active:
        time.sleep(1)  # Reset interval
        syn_counts.clear()
        syn_ack_counts.clear()

def start_syn_sniffing():
    """Start SYN flood detection with better error handling"""
    global syn_detection_active
    syn_detection_active = True
    
    try:
        # Start background thread to reset counts
        Thread(target=reset_syn_counts, daemon=True).start()
        
        # Start sniffing with BPF filter
        sniff(
            filter=f"tcp and dst host {syn_target_ip}",
            prn=detect_syn_flood,
            store=0,
            timeout=30  # Timeout to check detection_active
        )
    except Exception as e:
        print(f"SYN sniffing error: {e}")
    finally:
        syn_detection_active = False

# ================== PORT SCAN DETECTION FUNCTIONS ==================
def preprocess_features(features):
    """Preprocess features to match training data format"""
    if portscan_model is None or portscan_scaler is None or portscan_features is None:
        print("Port Scan model not loaded - skipping preprocessing")
        return None
        
    try:
        df = pd.DataFrame([features], columns=portscan_features['feature_names'])
        return portscan_scaler.transform(df)
    except Exception as e:
        print(f"Feature preprocessing error: {e}")
        return None

def portscan_packet_handler(packet):
    """Handle incoming packets and extract features"""
    extract_portscan_features(packet)

def extract_portscan_features(packet):
    """Extract relevant features from network packet"""
    try:
        features = {}
        
        if IP in packet:
            features['src_ip'] = packet[IP].src
            features['dst_ip'] = packet[IP].dst
            features['protocol'] = packet[IP].proto
            
            if TCP in packet:
                features['src_port'] = packet[TCP].sport
                features['dst_port'] = packet[TCP].dport
                features['flags'] = packet[TCP].flags
            elif UDP in packet:
                features['src_port'] = packet[UDP].sport
                features['dst_port'] = packet[UDP].dport
                features['flags'] = 'UDP'
        
        # Update connection stats
        if 'src_ip' in features and 'dst_ip' in features:
            conn_key = (features['src_ip'], features['dst_ip'])
            stats = connection_stats[conn_key]
            
            stats['count'] += 1
            current_time = time.time()
            
            if stats['last_timestamp']:
                stats['duration'] += current_time - stats['last_timestamp']
            stats['last_timestamp'] = current_time
            
            if TCP in packet and 'S' in str(packet[TCP].flags):
                stats['hot'] += 1
        
        return features
    except Exception as e:
        print(f"Packet processing error: {e}")
        return {}

def analyze_connections():
    """Analyze accumulated connection data for port scan patterns"""
    global alerts
    
    if portscan_model is None:
        return  # Skip if model isn't loaded
        
    for (src_ip, dst_ip), stats in connection_stats.items():
        if stats['last_timestamp'] is None:
            continue
            
        model_features = {
            'feature_0': 0,  # Placeholder for protocol type
            'feature_1': stats['count'],
            'feature_2': stats['duration'],
            # ... (rest of your feature mapping)
        }
        
        processed_features = preprocess_features(model_features)
        if processed_features is None:
            continue
        
        try:
            prediction = portscan_model.predict(processed_features)
            probability = portscan_model.predict_proba(processed_features)[0][1]
            
            if prediction[0] == 1 and probability > 0.9:
                alert = {
                    'source_ip': src_ip,
                    'target_ip': dst_ip,
                    'probability': probability,
                    'connection_count': stats['count'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'interface': current_interface
                }
                alerts.append(alert)
        except Exception as e:
            print(f"Prediction error: {e}")

def run_portscan_detection(interface):
    """Run the detection on specified interface"""
    global portscan_monitoring_active
    while portscan_monitoring_active:
        try:
            sniff(iface=interface, prn=portscan_packet_handler, timeout=TIME_WINDOW)
            analyze_connections()
            
            # Clear old connection data
            current_time = time.time()
            old_keys = [k for k in connection_stats 
                       if connection_stats[k]['last_timestamp'] and 
                       (current_time - connection_stats[k]['last_timestamp'] > TIME_WINDOW)]
            for k in old_keys:
                del connection_stats[k]
                
        except Exception as e:
            print(f"Monitoring error: {e}")
            alerts.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': f"Interface {interface} error: {str(e)}. Try running as root (sudo)."
            })
            portscan_monitoring_active = False

# ================== HELPER FUNCTIONS ==================
def generate_chart(results):
    try:
        plt.switch_backend('Agg')  # Important for non-GUI environments
        
        # Prepare data - combine all categories
        categories = ['Normal', 'Malware', 'Other Attacks']
        values = [
            results['normal'],
            sum(results['malware'].values()),
            sum(results['attacks'].values())
        ]
        colors = ['#4CAF50', '#FF5252', '#FFC107']  # Green, Red, Yellow
        
        # Create plot with larger size
        plt.figure(figsize=(8, 6))
        bars = plt.bar(categories, values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if value > 0
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom',
                        fontsize=12, fontweight='bold')
        
        # Customize the plot
        plt.title('Traffic Distribution', fontsize=14, pad=20)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to BytesIO buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Encode image to base64
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_img}"
        
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None 

def generate_accuracy_chart(accuracies):
    """ Generate Bar Chart for Model Accuracies """
    plt.figure(figsize=(10, 5))
    models = list(accuracies.keys())
    accuracy_values = list(accuracies.values())
    colors = ["#" + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(len(models))]
    
    bars = plt.bar(models, accuracy_values, color=colors)
    plt.xlabel("Models", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Model Accuracy Comparison", fontsize=16)
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add accuracy values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom')
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    img_buffer.seek(0)
    encoded_image = base64.b64encode(img_buffer.read()).decode("utf-8")
    plt.close()
    return f"data:image/png;base64,{encoded_image}"

# Database setup
def get_db():
    try:
        if 'db' not in g:
            g.db = sqlite3.connect(os.path.join(BASE_DIR, 'user.db'))
            g.db.row_factory = sqlite3.Row
        return g.db
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

# Error handling
@app.errorhandler(500)
def internal_error(error):
    app.logger.error('Internal Server Error: %s', error)
    return jsonify({
        'status': 'error',
        'message': 'Internal server error, please try again later.'
    }), 500

def hash_password(password):
    """Hashes the password using scrypt"""
    salt = os.urandom(16)  # Generate a random salt
    hashed = hashlib.scrypt(
        password.encode('utf-8'),  # Encode password to bytes
        salt=salt,
        n=16384,  # CPU/memory cost factor (adjustable)
        r=8,  # Block size
        p=1,  # Parallelization factor
        dklen=64  # Length of the derived key
    )
    return f"scrypt:n=16384,r=8,p=1${salt.hex()}${hashed.hex()}"

# User class for DB operations
class User:
    @staticmethod
    def create_user(fullname, username, email, password):
        hashed_password = hash_password(password)
        db = get_db()
        db.execute(
            'INSERT INTO user (fullname, username, email, password) VALUES (?, ?, ?, ?)',
            (fullname, username, email, hashed_password)
        )
        db.commit()

    @staticmethod
    def get_user_by_email(email):
        db = get_db()
        return db.execute(
            'SELECT * FROM user WHERE email = ?',
            (email,)
        ).fetchone()

    @staticmethod
    def get_user_by_username(username):
        db = get_db()
        return db.execute(
            'SELECT * FROM user WHERE username = ?',
            (username,)
        ).fetchone()

    @staticmethod
    def check_password(user, password):
        return hashlib.scrypt(
            password.encode('utf-8'),
            salt=bytes.fromhex(user['password'].split('$')[1]),
            n=16384,
            r=8,
            p=1,
            dklen=64
        ).hex() == user['password'].split('$')[2]

# Create table if not exists (run once to create the table)
def create_table():
    with app.app_context():  # Ensures application context is active
        db = get_db()
        # Create the table if it doesn't exist
        db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)')
        db.commit()

# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Decorator for login-required routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Database configuration
DATABASE = os.path.join(BASE_DIR, 'feedback.db')

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            rating INTEGER NOT NULL DEFAULT 0
        )
    ''')
    # Check if the 'rating' column exists, and add it if it doesn't
    cursor = conn.execute("PRAGMA table_info(feedback)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'rating' not in columns:
        conn.execute('ALTER TABLE feedback ADD COLUMN rating INTEGER NOT NULL DEFAULT 0')
    conn.commit()
    conn.close()

def initialize_models():
    global models, label_encoders, scaler, model_accuracies
    try:
        # Load and preprocess data
        data = pd.read_csv(os.path.join(BASE_DIR, 'fs_new validation project.csv'), header=None)
        columns = [
            'protocol', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'label'
        ]
        data.columns = columns

        # Encode categorical variables
        categorical_cols = ['protocol', 'service', 'flag']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        
        # Map attack types
        data['label'] = data['label'].apply(map_attack_type)

        # Split data and train models
        X = data.drop('label', axis=1)
        y = data['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'KNN': KNeighborsClassifier(),
            
            'Gradient Boosting': GradientBoostingClassifier()
        }
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            # Calculate and store accuracy
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred) * 100
            model_accuracies[name] = accuracy
        
        logger.info("Models initialized successfully.")
        logger.info(f"Model accuracies: {model_accuracies}")
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
        raise

def map_attack_type(attack):
    dos_attacks = ['neptune', 'smurf', 'pod', 'teardrop', 'land', 'back', 'apache2', 'udpstorm', 'processtable', 'mailbomb']
   
    if attack in dos_attacks:
        return 'DoS'
    
    else:
        return 'Normal'

# Initialize and train the model
def initialize_model():
    data = {
        "input": [
            '<script>alert("XSS Attack")</script>',
            '<img src="x" onerror="alert(\'XSS\')">',
            '<a href="javascript:alert(\'XSS\')">Click me</a>',
            'Normal text without malicious content',
            'Another safe input with no XSS tags',
            '<div onmouseover="alert(\'XSS\')">Hover me</div>',
            '<iframe src="javascript:alert(\'XSS\')"></iframe>',
            'Click here for <b>bold</b> text (safe)',
            '<body onload="alert(\'XSS\')">',
            'Safe plain text message',
        ],
        "label": [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["input"])
    y = df["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and store accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    model_accuracies['XSS Detection'] = accuracy
    
    return model, vectorizer

# Initialize model and vectorizer
model, vectorizer = initialize_model()

# List of expected features (must match the model's training features)
expected_features = [
    'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens', 'nb_at', 
    'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore', 'nb_tilde', 
    'nb_percent', 'nb_slash', 'nb_star', 'nb_colon', 'nb_comma', 
    'nb_semicolumn', 'nb_dollar', 'nb_space', 'nb_www', 'nb_com', 
    'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url', 
    'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain', 
    'abnormal_subdomain', 'nb_subdomains', 'prefix_suffix', 'random_domain', 
    'shortening_service', 'path_extension', 'nb_redirection', 
    'nb_external_redirection', 'length_words_raw', 'char_repeat', 
    'shortest_words_raw', 'shortest_word_host', 'shortest_word_path', 
    'longest_words_raw', 'longest_word_host', 'longest_word_path', 
    'avg_words_raw', 'avg_word_host', 'avg_word_path', 'phish_hints', 
    'domain_in_brand', 'brand_in_subdomain', 'brand_in_path', 'suspecious_tld', 
    'statistical_report', 'nb_hyperlinks', 'ratio_intHyperlinks', 
    'ratio_extHyperlinks', 'ratio_nullHyperlinks', 'nb_extCSS', 
    'ratio_intRedirection', 'ratio_extRedirection', 'ratio_intErrors', 
    'ratio_extErrors', 'login_form', 'external_favicon', 'links_in_tags', 
    'submit_email', 'ratio_intMedia', 'ratio_extMedia', 'sfh', 'iframe', 
    'popup_window', 'safe_anchor', 'onmouseover', 'right_clic', 'empty_title', 
    'domain_in_title', 'domain_with_copyright', 'whois_registered_domain', 
    'domain_registration_length', 'domain_age', 'web_traffic', 'dns_record', 
    'google_index', 'page_rank'
]

def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    # Basic URL features
    features['length_url'] = len(url)
    features['length_hostname'] = len(domain)
    features['ip'] = 1 if re.match(r'(\d{1,3}\.){3}\d{1,3}', domain) else 0
    features['nb_dots'] = url.count('.')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_qm'] = url.count('?')
    features['nb_and'] = url.count('&')
    features['nb_or'] = url.count('|')
    features['nb_eq'] = url.count('=')
    features['nb_underscore'] = url.count('_')
    features['nb_tilde'] = url.count('~')
    features['nb_percent'] = url.count('%')
    features['nb_slash'] = url.count('/')
    features['nb_star'] = url.count('*')
    features['nb_colon'] = url.count(':')
    features['nb_comma'] = url.count(',')
    features['nb_semicolumn'] = url.count(';')
    features['nb_dollar'] = url.count('$')
    features['nb_space'] = url.count(' ')
    features['nb_www'] = 1 if 'www' in url else 0
    features['nb_com'] = 1 if '.com' in url else 0
    features['nb_dslash'] = url.count('//')
    features['http_in_path'] = 1 if 'http' in url else 0
    features['https_token'] = 1 if 'https' in url else 0
    features['ratio_digits_url'] = len(re.findall(r'\d', url)) / len(url) if len(url) > 0 else 0

    # Domain-based features
    features['tld_in_path'] = 1 if re.search(r'\.[a-z]{2,}$', parsed_url.path) else 0
    features['tld_in_subdomain'] = 1 if re.search(r'\.[a-z]{2,}$', parsed_url.netloc) else 0
    features['nb_subdomains'] = len(parsed_url.netloc.split('.')) - 1
    features['shortening_service'] = 1 if any(service in url for service in ['bit.ly', 'goo.gl', 'tinyurl.com']) else 0
    features['abnormal_subdomain'] = 1 if len(parsed_url.netloc.split('.')) > 3 else 0

    # Keyword-based features
    features['phish_hints'] = 1 if any(word in url for word in ['login', 'signin', 'account', 'verify']) else 0

    # WHOIS Information
    try:
        domain_info = whois.whois(domain)
        features['whois_registered_domain'] = 1 if domain_info.domain_name else 0
        if domain_info.creation_date and domain_info.expiration_date:
            creation_date = domain_info.creation_date[0] if isinstance(domain_info.creation_date, list) else domain_info.creation_date
            expiration_date = domain_info.expiration_date[0] if isinstance(domain_info.expiration_date, list) else domain_info.expiration_date
            features['domain_age'] = (datetime.now() - creation_date).days
            features['domain_registration_length'] = (expiration_date - creation_date).days
        else:
            features['domain_age'] = 0
            features['domain_registration_length'] = 0
    except:
        features['whois_registered_domain'] = 0
        features['domain_age'] = 0
        features['domain_registration_length'] = 0

    # Web content features
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        features['domain_in_title'] = 1 if domain.lower() in title.lower() else 0
        features['empty_title'] = 1 if not title.strip() else 0
        features['domain_with_copyright'] = 1 if '©' in soup.get_text() else 0
        features['external_favicon'] = 1 if soup.find("link", rel="icon") and domain not in soup.find("link", rel="icon")['href'] else 0
    except:
        features['domain_in_title'] = 0
        features['empty_title'] = 1
        features['domain_with_copyright'] = 0
        features['external_favicon'] = 0

    # Fill missing values for features that are not extracted
    for feature_name in expected_features:
        if feature_name not in features:
            features[feature_name] = 0

    # Convert features to DataFrame
    features_df = pd.DataFrame([features])
    return features_df

# ================== ROUTES ==================
@app.route('/')
def welcome():
    return render_template('welcomepage.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            fullname = request.form.get('fullname')
            email = request.form.get('email')
            username = request.form.get('username')
            password = request.form.get('password')

            # Check if user already exists
            if User.get_user_by_email(email):
                return jsonify({'message': 'Email already registered!'}), 400

            if User.get_user_by_username(username):
                return jsonify({'message': 'Username already taken!'}), 400

            # Create new user
            User.create_user(fullname, username, email, password)

            # Set session
            session['user_id'] = username  # Save username as session info

            return jsonify({
                'message': f'Welcome, {username}!',
                'redirect': True
            }), 200

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return jsonify({'message': 'Registration failed!'}), 500

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')

            user = User.get_user_by_email(email)
            
            if not user or not User.check_password(user, password):
                return jsonify({
                    'message': 'Invalid email or password!,you have to signup '
                }), 401

            session['user_id'] = user['username']

            return jsonify({
                'message': f'Welcome back, {user["fullname"]}!',
                'redirect': True
            }), 200

        except Exception as e:
            logger.error(f"Login error: {e}")
            return jsonify({'message': 'Login failed!'}), 500

    return render_template('login.html')

@app.route('/home')
@login_required
def home():
    # Generate accuracy chart
    accuracy_chart = generate_accuracy_chart(model_accuracies)
    return render_template('model1.html', accuracy_chart=accuracy_chart, model_accuracies=model_accuracies)

@app.route('/dos_detection')
@login_required
def dos_detection():
    return render_template('dos_detection.html')

@app.route('/phishing_detection')
@login_required
def phishing_detection():
    return render_template('phis.html')

@app.route('/xss_detection')
@login_required
def xss_detection():
    return render_template('xss_detection.html')

@app.route('/predict')
@login_required
def predict_mal():
    return render_template('mal.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/predict_phishing', methods=['POST'])
def predict_phishing():
    try:
        data = request.json
        url = data.get('url', '').strip()

        # Validate the URL
        if not url:
            return jsonify({"error": "URL is required"}), 400

        if not url.startswith(('http://', 'https://')):
            return jsonify({"error": "Invalid URL format. URL must start with http:// or https://"}), 400

        logger.info(f"Processing URL: {url}")

        # Extract features
        try:
            input_features = extract_features(url)
            logger.info(f"Extracted Features: {input_features.to_dict()}")
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return jsonify({"error": "Failed to extract features from the URL"}), 500

        # Ensure the features match the expected format
        try:
            input_features = input_features.reindex(columns=expected_features, fill_value=0)
            logger.info(f"Reindexed Features: {input_features.to_dict()}")
        except Exception as e:
            logger.error(f"Error reindexing features: {str(e)}")
            return jsonify({"error": "Failed to prepare features for prediction"}), 500

        # Predict using the model
        try:
            prediction = phishing_model.predict(input_features)
            result = "Phishing" if prediction[0] == 1 else "Safe"
            logger.info(f"Prediction successful. Result: {result}")
            return jsonify({"result": result})
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({"error": "Failed to make a prediction"}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/predict_xss', methods=['POST'])
def predict_xss():
    try:
        data = request.json
        input_text = data.get('text', '')
        
        # Transform input and predict
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        
        # Generate response
        if prediction == 1:
            result = "Potential XSS attack detected!"
            status = "danger"
        else:
            result = "No XSS detected."
            status = "safe"
            
        return jsonify({
            'result': result,
            'status': status
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_dos', methods=['POST'])
def predict_dos():
    try:
        user_input = request.get_json()
        logger.debug(f"Received input: {user_input}")

        numeric_fields = ['src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
                          'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
                          'su_attempted', 'num_root', 'num_file_creations']

        for field in numeric_fields:
            user_input[field] = float(user_input.get(field, 0.0))

        sample_data = pd.DataFrame([user_input])

        for col, le in label_encoders.items():
            if sample_data[col].iloc[0] not in le.classes_:
                le.classes_ = np.append(le.classes_, sample_data[col].iloc[0])
            sample_data[col] = le.transform(sample_data[col])

        sample_data_scaled = scaler.transform(sample_data)
        results = {}

        for name, model in models.items():
            prediction = model.predict(sample_data_scaled)[0]
            results[name] = {'prediction': prediction}
            logger.debug(f"{name} prediction: {prediction}")

        # Generate HTML for prediction results
        result_html = "<h2>Prediction Results:</h2>"
        for model_name, result in results.items():
            result_html += f"<div class='model-prediction'><h3>{model_name}</h3>"
            result_html += f"<p>Prediction: <strong>{result['prediction']}</strong></p>"
            result_html += f"<p>Model Accuracy: <strong>{model_accuracies.get(model_name, 'N/A')}%</strong></p></div>"

        return jsonify({'status': 'success', 'result_html': result_html})

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'status': 'error', 'error': f"Prediction failed: {str(e)}"}), 500

def detect_attack_patterns(user_input):
    patterns = []
    thresholds = {
        'src_bytes': 1000000,
        'dst_bytes': 1000000,
        'num_failed_logins': 5,
        'num_compromised': 0,
        'root_shell': 1,
        'su_attempted': 1
    }
    
    for feature, threshold in thresholds.items():
        if user_input.get(feature, 0) > threshold:
            patterns.append(f"Suspicious {feature.replace('_', ' ')}: {user_input[feature]}")
    
    return patterns if patterns else ["No suspicious patterns detected"]

@app.route('/predict_malware', methods=['POST'])
def predict_malware():
    """ Predict Malware Type from Uploaded File """
    try:
        if malware_model is None or malware_scaler is None or malware_class_mapping is None:
            return jsonify({'error': 'Malware model, scaler, or class mapping not loaded properly.'})

        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the file into a DataFrame
        df = pd.read_csv(file_path, header=None)
        print("Original DataFrame:\n", df.head())  # Debugging

        # Store the actual labels (last column)
        actual_labels = df.iloc[:, -1].astype(str).str.lower().tolist()
        
        # Drop the last column (attack type) to exclude it from prediction
        df_features = df.iloc[:, :-1]  # Exclude last column
        print("Features Only DataFrame:\n", df_features.head())  # Debugging

        # Ensure encoders are loaded
        if malware_encoders is None:
            return jsonify({'error': 'Encoders not loaded. Please check model files.'})

        # Encode categorical columns
        categorical_columns = ['protocol', 'service', 'flag']
        for idx, col in enumerate(categorical_columns):
            if idx in df_features.columns:  # Ensure column exists
                if col in malware_encoders:
                    df_features[idx] = malware_encoders[col].transform(df_features[idx])  # Apply encoding
                else:
                    return jsonify({'error': f'Missing encoder for column: {col}'})

        print("Encoded Features DataFrame:\n", df_features.head())  # Debugging

        # Scale the features
        X_scaled = malware_scaler.transform(df_features)

        # Make predictions
        predicted_classes = malware_model.predict(X_scaled)
        predicted_labels = [malware_class_mapping.get(str(cls), "Unknown") for cls in predicted_classes]
        
        # Define malware types (only these will be categorized as 'malware')
        malware_types = [
            'back', 'rootkit', 'warezclient', 'warezmaster', 
            'xterm', 'httptunnel', 'worm', 'sqlattack'
        ]
        
        results = {
            'malware': {},
            'attacks': {},
            'normal': 0
        }
        
        for pred, actual in zip(predicted_labels, actual_labels):
            label = pred.lower()
            actual_label = actual.lower()
            
            if actual_label == 'normal':
                results['normal'] += 1
            elif label in malware_types:
                results['malware'][label] = results['malware'].get(label, 0) + 1
            else:
                # For other attacks, use the predicted label (or actual if preferred)
                results['attacks'][label] = results['attacks'].get(label, 0) + 1
        
        chart_url = generate_chart(results)

        return jsonify({
            "success": True,
            "results": results,
            "chart": chart_url  # Removed model_accuracy
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/predict_packet', methods=['POST'])
def predict_packet():
    """ Predict Malware Type from Single Packet Data """
    try:
        # Validate model components
        if not all([malware_model, malware_scaler, malware_class_mapping, malware_encoders]):
            return jsonify({'error': 'Model components not loaded properly'}), 500

        data = request.json
        packet_data = data.get('packet_data', '').strip()
        
        if not packet_data:
            return jsonify({'error': 'Packet data is missing'}), 400

        # Split and validate input
        raw_values = [x.strip() for x in packet_data.split(',')]
        
        # Expected features (adjust based on your model)
        expected_features = 16  # Change this number based on your actual model
        if len(raw_values) != expected_features:
            return jsonify({
                'error': f'Expected {expected_features} comma-separated values (got {len(raw_values)})',
                'example': 'tcp,http,SF,100,200,0,0,0,0,0,1,0,0,0,0,0'
            }), 400

        # Create DataFrame with proper column names (adjust based on your model)
        feature_names = [f'feature_{i}' for i in range(expected_features)]
        df = pd.DataFrame([raw_values], columns=feature_names)

        # First check for specific malware patterns (enhanced)
        malware_result = detect_specific_malware(df)
        if malware_result:
            return jsonify({
                "success": True,
                "prediction": malware_result,
                "mitigation": get_mitigation_advice(malware_result['category'])
            })

        # If no specific malware detected, proceed with model prediction
        
        # Convert all features to numeric where possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                # Handle categorical features (like protocol, service, flag)
                if col in malware_encoders:
                    try:
                        df[col] = malware_encoders[col].transform(df[col])
                    except:
                        # If value not in encoder, use unknown category
                        df[col] = 0  # Or use a specific 'unknown' code

        # Scale the features
        X_scaled = malware_scaler.transform(df)

        # Make prediction
        predicted_class = malware_model.predict(X_scaled)[0]
        predicted_label = malware_class_mapping.get(str(predicted_class), "unknown").lower()

        # Enhanced classification logic with more malware types
        if predicted_label == 'normal':
            result = {
                "type": "normal",
                "category": "normal",
                "label": "NORMAL",
                "confidence": 99,
                "indicators": ["No malicious patterns detected"] 
            }
        elif predicted_label in ['neptune', 'smurf', 'pod', 'teardrop', 'land']:
            result = {
                "type": "attack",
                "category": "dos",
                "label": predicted_label.upper(),
                "confidence": 96,
                "indicators": [
                    "High packet rate",
                    "Suspicious traffic patterns",
                    "Abnormal packet size distribution"
                ]
            }
        elif predicted_label in ['satan', 'ipsweep', 'nmap', 'portsweep']:
            result = {
                "type": "attack",
                "category": "probe",
                "label": predicted_label.upper(),
                "confidence": 92,
                "indicators": [
                    "Port scanning activity",
                    "Network reconnaissance",
                    "Multiple connection attempts"
                ]
            }
        elif predicted_label in ['warezclient', 'warezmaster']:
            result = {
                "type": "malware",
                "category": "warez",
                "label": predicted_label.upper(),
                "confidence": 89,
                "indicators": [
                    "FTP-related anomalies",
                    "Unauthorized data transfer",
                    "Suspicious authentication patterns"
                ]
            }
        elif predicted_label in ['rootkit', 'back', 'httptunnel']:
            result = {
                "type": "malware",
                "category": "stealth",
                "label": predicted_label.upper(),
                "confidence": 91,
                "indicators": [
                    "Stealthy connection patterns",
                    "Abnormal service behavior",
                    "Possible backdoor activity"
                ]
            }
        elif predicted_label in ['sqlattack', 'xterm', 'loadmodule']:
            result = {
                "type": "malware",
                "category": "exploit",
                "label": predicted_label.upper(),
                "confidence": 88,
                "indicators": [
                    "Privilege escalation attempts",
                    "Suspicious command patterns",
                    "Exploit-like behavior"
                ]
            }
        elif predicted_label == 'worm':
            result = {
                "type": "malware",
                "category": "worm",
                "label": "WORM",
                "confidence": 90,
                "indicators": [
                    "Self-replicating patterns",
                    "Rapid spread behavior",
                    "Network propagation attempts"
                ]
            }
        else:  # Other malware types
            result = {
                "type": "malware",
                "category": predicted_label,
                "label": predicted_label.upper(),
                "confidence": 85,
                "indicators": get_attack_indicators(predicted_label, df)
            }

        return jsonify({
            "success": True,
            "prediction": result,
            "mitigation": get_mitigation_advice(result['category'])
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def detect_specific_malware(df):
    """ Enhanced malware detection with rules for specific malware types """
    try:
        # Convert to numeric where needed - updated to handle deprecation warning
        for col in df.select_dtypes(include=['object']):
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # If conversion fails, leave as string or handle differently
                pass

        # Rest of your detection logic remains the same...
        # ROOTKIT detection - enhanced patterns
        if (df['feature_2'].iloc[0] in ['SF', 'S0', 'S1'] and  # stealthy flags
            df['feature_3'].iloc[0] > 0 and     # duration
            df['feature_4'].iloc[0] > 0 and     # src_bytes
            df['feature_5'].iloc[0] == 0 and    # dst_bytes
            df['feature_10'].iloc[0] > 0.2 and  # dst_host_srv_count
            df['feature_11'].iloc[0] < 0.1):    # dst_host_same_srv_rate
            return {
                "type": "malware",
                "category": "rootkit",
                "label": "ROOTKIT",
                "confidence": 94,
                "indicators": [
                    "Stealthy connection patterns",
                    "Low byte transfer with service count",
                    "Possible backdoor activity",
                    "Abnormal same service rate"
                ]
            }

        # WAREZ detection - enhanced patterns
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_1'].iloc[0] in ['ftp_data', 'ftp', 'http', 'http_443'] and
            df['feature_4'].iloc[0] > 0 and      # src_bytes
            df['feature_5'].iloc[0] == 0 and     # dst_bytes
            df['feature_9'].iloc[0] == 0):       # logged_in
            return {
                "type": "malware",
                "category": "warez",
                "label": "WAREZMASTER" if df['feature_4'].iloc[0] > 100 else "WAREZCLIENT",
                "confidence": 90,
                "indicators": [
                    "FTP/HTTP data channel activity",
                    "Suspicious byte patterns",
                    "Unauthenticated access",
                    "Potential illegal file transfer"
                ]
            }

        # BACK detection - backdoor patterns
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_2'].iloc[0] in ['S1', 'SF'] and
            df['feature_3'].iloc[0] > 1.0 and    # duration
            df['feature_4'].iloc[0] > 0 and      # src_bytes
            df['feature_5'].iloc[0] == 0 and     # dst_bytes
            df['feature_7'].iloc[0] > 0.5):      # wrong_fragment
            return {
                "type": "malware",
                "category": "backdoor",
                "label": "BACK",
                "confidence": 92,
                "indicators": [
                    "Long-lived connection",
                    "Wrong fragments detected",
                    "Possible covert channel",
                    "Backdoor-like behavior"
                ]
            }

        # HTTPTUNNEL detection
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_1'].iloc[0] == 'http' and
            df['feature_4'].iloc[0] > 100 and    # src_bytes
            df['feature_5'].iloc[0] > 100 and    # dst_bytes
            df['feature_6'].iloc[0] > 0.5):      # land
            return {
                "type": "malware",
                "category": "tunnel",
                "label": "HTTPTUNNEL",
                "confidence": 88,
                "indicators": [
                    "HTTP tunnel patterns",
                    "Bi-directional similar traffic",
                    "Possible data exfiltration"
                ]
            }

        # SQL attack detection
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_1'].iloc[0] in ['sql_net', 'oracle'] and
            df['feature_4'].iloc[0] > 50 and     # src_bytes
            df['feature_5'].iloc[0] > 50 and     # dst_bytes
            df['feature_8'].iloc[0] > 0.7):      # urgent
            return {
                "type": "malware",
                "category": "sqlinjection",
                "label": "SQLATTACK",
                "confidence": 89,
                "indicators": [
                    "Database service activity",
                    "Urgent flag set",
                    "Possible SQL injection attempt"
                ]
            }

        # XTERM detection - enhanced
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_4'].iloc[0] in range(3, 6) and  # src_bytes
            df['feature_5'].iloc[0] in range(1, 4) and  # dst_bytes
            df['feature_2'].iloc[0] == 'SF' and         # flag
            df['feature_12'].iloc[0] > 0.5):           # dst_host_diff_srv_rate
            return {
                "type": "malware",
                "category": "privilege_escalation",
                "label": "XTERM",
                "confidence": 87,
                "indicators": [
                    "Suspicious terminal activity",
                    "Unusual byte patterns",
                    "Potential exploit attempt",
                    "Abnormal diff service rate"
                ]
            }

        # WORM detection
        if (df['feature_0'].iloc[0] == 'tcp' and
            df['feature_3'].iloc[0] == 0 and     # duration
            df['feature_4'].iloc[0] > 0 and      # src_bytes
            df['feature_5'].iloc[0] > 0 and      # dst_bytes
            df['feature_13'].iloc[0] > 0.8):     # dst_host_srv_diff_host_rate
            return {
                "type": "malware",
                "category": "worm",
                "label": "WORM",
                "confidence": 91,
                "indicators": [
                    "Rapid connection attempts",
                    "Bi-directional traffic",
                    "High diff host rate",
                    "Worm-like propagation"
                ]
            }

        return None

    except Exception as e:
        print(f"Error in malware detection: {str(e)}")
        return None


def get_mitigation_advice(category):
    """ Return mitigation strategies with more specific malware advice """
    advice = {
        "warez": [
            "Immediately block FTP/HTTP service on affected ports",
            "Reset all credentials for affected services",
            "Scan for unauthorized file transfers",
            "Check for installed warez clients/servers"
        ],
        "rootkit": [
            "Isolate affected systems immediately",
            "Perform memory forensics analysis",
            "Check for kernel module modifications",
            "Reinstall operating system from clean media"
        ],
        "backdoor": [
            "Identify and remove persistence mechanisms",
            "Rotate all credentials and keys",
            "Check for unusual cron jobs/services",
            "Scan for reverse shell connections"
        ],
        "tunnel": [
            "Inspect HTTP traffic for tunneling patterns",
            "Block suspicious HTTP headers",
            "Implement HTTP payload inspection",
            "Monitor for data exfiltration"
        ],
        "sqlinjection": [
            "Patch database management systems",
            "Implement SQL injection filters",
            "Review database access logs",
            "Enable parameterized queries"
        ],
        "worm": [
            "Isolate infected systems from network",
            "Patch all vulnerable services",
            "Implement egress filtering",
            "Update antivirus signatures"
        ],
        "privilege_escalation": [
            "Audit sudo permissions",
            "Check for setuid binaries",
            "Review recent privilege changes",
            "Monitor for suspicious process creation"
        ],
        "dos": [
            "Implement rate limiting on network services",
            "Configure SYN flood protection",
            "Contact your ISP about the source IP",
            "Enable DDoS mitigation services"
        ],
        "probe": [
            "Block scanning IP addresses",
            "Implement port knocking",
            "Reduce ICMP responses",
            "Enable intrusion detection systems"
        ],
        "normal": [
            "No action required",
            "Continue normal monitoring"
        ],
        "default": [
            "Isolate affected systems",
            "Review relevant security logs",
            "Update firewall and IDS rules",
            "Conduct thorough malware scan"
        ]
    }
    return advice.get(category.lower(), advice['default'])


def get_attack_indicators(label, df):
    """ Return more detailed indicators for detected threats """
    indicators = {
        "warezmaster": [
            f"FTP activity (feature_1: {df['feature_1'].iloc[0]})",
            f"High source bytes (feature_4: {df['feature_4'].iloc[0]})",
            "Unauthenticated access",
            "Potential file transfer"
        ],
        "warezclient": [
            f"FTP/HTTP activity (feature_1: {df['feature_1'].iloc[0]})",
            f"Source bytes (feature_4: {df['feature_4'].iloc[0]})",
            "Client-side patterns",
            "Possible download activity"
        ],
        "rootkit": [
            f"Stealth traffic (feature_2: {df['feature_2'].iloc[0]})",
            f"Abnormal calls (feature_7: {df['feature_7'].iloc[0]})",
            "Low visibility patterns",
            "Possible kernel-level activity"
        ],
        "back": [
            f"Long connection (feature_3: {df['feature_3'].iloc[0]})",
            f"Wrong fragments (feature_7: {df['feature_7'].iloc[0]})",
            "Backdoor-like behavior",
            "Covert channel indicators"
        ],
        "httptunnel": [
            f"HTTP traffic (feature_1: {df['feature_1'].iloc[0]})",
            f"Bi-directional bytes (feature_4: {df['feature_4'].iloc[0]}/{df['feature_5'].iloc[0]})",
            "Tunneling patterns",
            "Possible data exfiltration"
        ],
        "sqlattack": [
            f"Database service (feature_1: {df['feature_1'].iloc[0]})",
            f"Urgent flag set (feature_8: {df['feature_8'].iloc[0]})",
            "SQL injection patterns",
            "Database exploitation attempts"
        ],
        "xterm": [
            f"Terminal activity (feature_4: {df['feature_4'].iloc[0]})",
            f"Abnormal service rate (feature_12: {df['feature_12'].iloc[0]})",
            "Privilege escalation patterns",
            "Exploit-like behavior"
        ],
        "worm": [
            f"Rapid connections (feature_13: {df['feature_13'].iloc[0]})",
            "Self-replicating patterns",
            "Network propagation attempts",
            "Multiple host targeting"
        ],
        "dos": [
            f"High packet rate (feature_4: {df['feature_4'].iloc[0]})",
            f"SYN flood patterns (feature_10: {df['feature_10'].iloc[0]})",
            "Traffic volume anomalies",
            "Service disruption patterns"
        ],
        "default": [
            "Suspicious network patterns detected",
            f"Model prediction confidence: {df.get('confidence', 85)}%",
            "Multiple anomaly indicators",
            "Requires further investigation"
        ]
    }
    return indicators.get(label.lower(), indicators['default'])
# ================== ARP SPOOFING ROUTES ==================
@app.route('/arp_detection')
@login_required
def arp_detection():
    return render_template('arp.html')

# ================== ARP SPOOFING ROUTES ==================
@app.route('/api/arp/set_network', methods=['POST'])
def set_arp_network():
    global target_network
    data = request.get_json()
    
    if not data or 'network' not in data:
        return jsonify({'status': 'error', 'message': 'Network parameter missing'}), 400
        
    try:
        # Validate network format
        target_network = data['network']
        ipaddress.ip_network(target_network)  # This validates the CIDR format
        
        return jsonify({
            'status': 'success', 
            'message': f'Monitoring network: {target_network}'
        })
    except ValueError as e:
        return jsonify({
            'status': 'error', 
            'message': f'Invalid network format: {str(e)}. Use CIDR notation like 192.168.1.0/24'
        }), 400

@app.route('/api/arp/start', methods=['POST'])  # Changed from /api/start
def start_arp_detection():
    global arp_sniffing_thread, arp_sniffing_active
    
    if not target_network:
        return jsonify({
            'status': 'error',
            'message': 'Network not configured. Set network first.'
        }), 400
        
    if arp_sniffing_active:
        return jsonify({
            'status': 'info',
            'message': 'ARP detection already running'
        })
        
    try:
        arp_sniffing_active = True
        arp_sniffing_thread = Thread(target=start_arp_sniffing)
        arp_sniffing_thread.daemon = True
        arp_sniffing_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'ARP detection started',
            'network': target_network
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start: {str(e)}'
        }), 500

@app.route('/api/arp/stop', methods=['POST'])
def stop_arp_detection():
    global arp_sniffing_active
    
    arp_sniffing_active = False
    
    # Wait for thread to finish if it exists
    if 'arp_sniffing_thread' in globals() and arp_sniffing_thread.is_alive():
        arp_sniffing_thread.join(timeout=2)
    
    return jsonify({
        'status': 'success',
        'message': 'ARP detection stopped'
    })

@app.route('/api/arp/alerts', methods=['GET'])
def get_arp_alerts():
    try:
        return jsonify({
            'status': 'success',
            'alerts': arp_alerts,
            'count': len(arp_alerts)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get alerts: {str(e)}'
        }), 500

@app.route('/api/arp/clear', methods=['POST'])
def clear_arp_alerts():
    global arp_alerts
    
    try:
        arp_alerts = []  # Clear the alerts list
        return jsonify({
            'status': 'success',
            'message': 'ARP alerts cleared',
            'cleared_count': len(arp_alerts)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to clear alerts: {str(e)}'
        }), 500

@app.route('/api/arp/status', methods=['GET'])
def get_arp_status():
    return jsonify({
        'sniffing_active': arp_sniffing_active,
        'target_network': target_network,
        'threads_alive': {
            'sniffing': arp_sniffing_thread.is_alive() if arp_sniffing_thread else False,
            'analysis': arp_analysis_thread.is_alive() if arp_analysis_thread else False
        },
        'alert_count': len(arp_alerts)
    })
# ================== SYN FLOOD ROUTES ==================
@app.route('/syn_flood')
@login_required
def syn_flood_detection():
    return render_template('sys.html')

@app.route('/start_syn_detection', methods=['POST'])
def start_syn_detection():
    """Start SYN flood detection with validation"""
    global syn_target_ip, syn_detection_active
    
    if syn_detection_active:
        return jsonify({"warning": "Detection already running"}), 200
    
    syn_target_ip = request.json.get('ip')
    if not syn_target_ip:
        return jsonify({"error": "Target IP required"}), 400
    
    try:
        Thread(target=start_syn_sniffing, daemon=True).start()
        return jsonify({
            "status": "SYN flood detection started",
            "target": syn_target_ip
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to start detection: {str(e)}"}), 500

@app.route('/stop_syn_detection', methods=['POST'])
def stop_syn_detection():
    """Stop SYN flood detection"""
    global syn_detection_active
    syn_detection_active = False
    return jsonify({"status": "SYN flood detection stopped"})

@app.route('/get_syn_alerts')
def get_syn_alerts():
    """Get SYN flood alerts with statistics"""
    return jsonify({
        "active": syn_detection_active,
        "target_ip": syn_target_ip,
        "total_syn_packets": sum(syn_counts.values()),
        "unique_sources": len(syn_counts),
        "alerts": syn_flood_alerts[-20:]  # Return last 20 alerts
    })

# ================== PORT SCAN ROUTES ==================
@app.route('/portscan')
def portscan_detection():
    return render_template('portscan.html')

@app.route('/api/interfaces')
def get_interfaces():
    """Return list of available network interfaces"""
    try:
        interfaces = get_if_list()
        if not interfaces:
            return jsonify({"error": "No network interfaces found"}), 404
        return jsonify(interfaces)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/start_monitoring')
def start_monitoring():
    """Start monitoring on selected interface"""
    global portscan_monitoring_active, current_interface, monitoring_thread
    
    if portscan_model is None:
        return jsonify({"error": "Port Scan detection is not available (model not loaded)"}), 503
        
    interface = request.args.get('interface')
    
    if not interface or interface not in get_if_list():
        return jsonify({"error": "Invalid interface"}), 400
        
    if not portscan_monitoring_active:
        try:
            portscan_monitoring_active = True
            current_interface = interface
            monitoring_thread = threading.Thread(target=run_portscan_detection, args=(interface,))
            monitoring_thread.daemon = True
            monitoring_thread.start()
            return jsonify({"status": f"Monitoring started on {interface}"})
        except Exception as e:
            portscan_monitoring_active = False
            return jsonify({"error": f"Failed to start monitoring: {str(e)}"}), 500
    return jsonify({"status": "Already monitoring"})

@app.route('/api/stop_monitoring')
def stop_monitoring():
    """Stop active monitoring"""
    global portscan_monitoring_active, monitoring_thread
    
    portscan_monitoring_active = False
    if monitoring_thread and monitoring_thread.is_alive():
        monitoring_thread.join(timeout=1)  # Wait for clean shutdown
        
    return jsonify({"status": "Monitoring stopped"})

@app.route('/api/status')
def get_status():
    """Get current monitoring status"""
    return jsonify({
        "active": portscan_monitoring_active,
        "interface": current_interface,
        "alert_count": len(alerts),
        "model_loaded": portscan_model is not None
    })

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify(alerts[-100:])  # Return last 100 alerts

# ================== CARD ROUTES ==================
@app.route('/doscard')
def doscard():
    return render_template('doscard.html', model_accuracies=model_accuracies)

@app.route('/phishingcard')
def phishingcard():
    return render_template('phishingcard.html', model_accuracies=model_accuracies)

@app.route('/xsscard')
def xsscard():
    return render_template('xsscard.html', model_accuracies=model_accuracies)

@app.route('/malwarecard')
def malwarecard():
    return render_template('malwarecard.html', model_accuracies=model_accuracies)

@app.route('/random_forest')
def random_forest():
    return render_template('random_forest.html', accuracy=model_accuracies.get('Random Forest', 'N/A'))

@app.route('/gradient_boosting')
def gradient_boosting():
    return render_template('gradient_boosting.html', accuracy=model_accuracies.get('Gradient Boosting', 'N/A'))

@app.route('/knn')
def knn():
    return render_template('knn.html', accuracy=model_accuracies.get('KNN', 'N/A'))

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/performance')
@login_required
def performance():
    """Route to display model performance metrics"""
    try:
        # Generate accuracy chart
        accuracy_chart = generate_accuracy_chart(model_accuracies)
        
        return render_template('performance.html',
                            accuracy_chart=accuracy_chart,
                            model_accuracies=model_accuracies,
                            model_descriptions=model_descriptions)
    except Exception as e:
        logger.error(f"Error loading performance page: {str(e)}")
        flash('Error loading performance metrics', 'error')
        return redirect(url_for('home'))

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    rating = request.form['rating']

    # Insert feedback into the database
    conn = get_db_connection()
    conn.execute('INSERT INTO feedback (name, email, message, rating) VALUES (?, ?, ?, ?)',
                 (name, email, message, rating))
    conn.commit()
    conn.close()
    
    return redirect(url_for('home'))

# ================== GEMINI CHATBOT ROUTES ==================
@app.route('/gemini')
def gemini():
    return render_template('gemini.html')

@app.route('/api/gemini', methods=['POST'])
def gemini_chat():
    try:
        # Get the user's message from the request
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Send the message to the Gemini API
        response = gemini_model.generate_content(user_message)

        # Return the Gemini API's response
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    with app.app_context():
        get_db().execute('''CREATE TABLE IF NOT EXISTS users
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                             fullname TEXT NOT NULL,
                             username TEXT UNIQUE NOT NULL,
                             email TEXT UNIQUE NOT NULL,
                             password TEXT NOT NULL)''')
        get_db().commit()
    logger.info("Initializing models...")
    initialize_models()  # Ensure this is called before running the app
        
    app.run(host='127.0.0.1', port=8080, debug=True)
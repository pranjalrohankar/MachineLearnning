import pandas as pd
import json

# Load dataset (update file path if needed)
df = pd.read_csv("fs_new validation project.csv", header=None)

# Assign column names manually
df.columns = [
    "protocol",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "attack_type"
]

# Extract feature names (excluding 'attack_type')
feature_names = df.columns[:-1].tolist()

# Save feature names to a file
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)  # Use the extracted feature names
    # Load feature names
with open("feature_names.json", "r") as f:
    expected_feature_names = json.load(f)

print("✅ Feature names saved correctly:", feature_names)
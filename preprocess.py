import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = "cybersecurity_intrusion_data.csv"

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# -----------------------------
# Drop columns not used for training
# -----------------------------
# session_id is an identifier
# attack_detected is a label (NOT used in unsupervised training)
columns_to_drop = ["session_id", "attack_detected"]
df_features = df.drop(columns=columns_to_drop)

# -----------------------------
# Separate categorical & numeric columns
# -----------------------------
categorical_cols = [
    "protocol_type",
    "encryption_used",
    "browser_type"
]

numeric_cols = [
    col for col in df_features.columns
    if col not in categorical_cols
]

# -----------------------------
# One-hot encode categorical features
# -----------------------------
df_encoded = pd.get_dummies(
    df_features,
    columns=categorical_cols,
    drop_first=True
)

# -----------------------------
# Scale numeric features
# -----------------------------
scaler = StandardScaler()
df_encoded[numeric_cols] = scaler.fit_transform(
    df_encoded[numeric_cols]
)

print("Processed shape:", df_encoded.shape)

# -----------------------------
# Save processed data
# -----------------------------
OUTPUT_PATH = "data/processed_logins.csv"
df_encoded.to_csv(OUTPUT_PATH, index=False)

print(f"Preprocessed data saved to {OUTPUT_PATH}")

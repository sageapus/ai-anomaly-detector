# -------------------------------
# Anomaly Detection on Preprocessed Login Data
# -------------------------------

import pandas as pd
from sklearn.ensemble import IsolationForest

# 1️⃣ Load preprocessed data
# Data is already one-hot encoded and scaled from preprocess.py
df = pd.read_csv('data/processed_logins.csv')

print("Loaded data shape:", df.shape)
print("Features:", df.columns.tolist())

# 2️⃣ Use all features (already preprocessed and scaled)
X = df.values

# 3️⃣ Train Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.02,  # adjust % of expected anomalies
    random_state=42
)
iso_forest.fit(X)

# 4️⃣ Predict anomalies
df['anomaly'] = iso_forest.predict(X)  # -1 = anomaly, 1 = normal
df['is_suspicious'] = df['anomaly'] == -1

# 5️⃣ Summary
print("\nAnomaly Detection Results:")
print("Total sessions:", len(df))
print("Suspicious sessions flagged:", df['is_suspicious'].sum())

# 6️⃣ Inspect suspicious sessions
print("\nSample suspicious sessions:")
suspicious_sessions = df[df['is_suspicious']]
print(suspicious_sessions.head())

# 7️⃣ Save report
df.to_csv('data/suspicious_sessions_report.csv', index=False)
print("\nReport saved to 'data/suspicious_sessions_report.csv'")

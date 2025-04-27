import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

def detect_anomalies(df):
    """Detect anomalies using Isolation Forest and DBSCAN."""
    features = df.select_dtypes(include=['float', 'int']).values

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_labels = iso_forest.fit_predict(features)
    df["isolation_forest_anomaly"] = iso_labels == -1

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(features)
    df["dbscan_anomaly"] = db_labels == -1

    # Combine both anomaly flags
    df["is_anomaly"] = df["isolation_forest_anomaly"] | df["dbscan_anomaly"]
    return df[df["is_anomaly"]]

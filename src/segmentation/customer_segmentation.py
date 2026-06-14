import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ---------------------------------
# Prepare Feature Matrix
# ---------------------------------

def prepare_features(df: pd.DataFrame, feature_columns):

    print("\nPreparing features for segmentation...")

    X = df[feature_columns].copy()

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print("Feature scaling completed")

    return X_scaled, scaler


# ---------------------------------
# Train K-Means Model
# ---------------------------------

def train_kmeans(X_scaled, n_clusters=4):

    print(f"\nTraining K-Means model with {n_clusters} clusters...")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    print("K-Means training completed")

    return model


# ---------------------------------
# Assign Customer Segments
# ---------------------------------

def assign_segments(df: pd.DataFrame, model, X_scaled):

    print("\nAssigning customer segments...")

    df = df.copy()

    df["Segment"] = model.predict(X_scaled)

    return df


# ---------------------------------
# Segment Statistics
# ---------------------------------

def segment_statistics(df: pd.DataFrame):

    print("\n===== SEGMENT STATISTICS =====")

    stats = df.groupby("Segment").agg({

        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "CLV": "mean"

    }).round(2)

    print(stats)

    return stats


# ---------------------------------
# Add Business Labels
# ---------------------------------

def label_segments(df: pd.DataFrame):

    print("\nLabelling segments...")

    segment_map = {
        0: "Low Value Customers",
        1: "Potential Customers",
        2: "Loyal Customers",
        3: "VIP Customers"
    }

    df["SegmentLabel"] = df["Segment"].map(segment_map)

    return df


# ---------------------------------
# Segment Distribution
# ---------------------------------

def segment_summary(df: pd.DataFrame):

    print("\n===== CUSTOMER SEGMENT SUMMARY =====")

    summary = df["SegmentLabel"].value_counts()

    print(summary)

    return summary


# ---------------------------------
# Save Segmentation Results
# ---------------------------------

def save_segments(df: pd.DataFrame):

    path = "data/processed/customer_segments.csv"

    df.to_csv(path, index=False)

    print(f"\nCustomer segmentation saved to {path}")
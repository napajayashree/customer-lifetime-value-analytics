import pandas as pd
import os


# ---------------------------------
# Create Directory if Missing
# ---------------------------------

def ensure_directory(path):

    if not os.path.exists(path):

        os.makedirs(path)

        print(f"Created directory: {path}")


# ---------------------------------
# Save DataFrame
# ---------------------------------

def save_dataframe(df: pd.DataFrame, path):

    ensure_directory(os.path.dirname(path))

    df.to_csv(path, index=False)

    print(f"Data saved to {path}")


# ---------------------------------
# Load DataFrame
# ---------------------------------

def load_dataframe(path):

    if not os.path.exists(path):

        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    print(f"Loaded dataset from {path}")

    return df


# ---------------------------------
# Basic Data Summary
# ---------------------------------

def dataframe_summary(df: pd.DataFrame):

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())


# ---------------------------------
# Feature / Target Split
# ---------------------------------

def split_features_target(df, target_column):

    X = df.drop(columns=[target_column])

    y = df[target_column]

    return X, y
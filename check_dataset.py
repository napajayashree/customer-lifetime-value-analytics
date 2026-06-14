import pandas as pd

path = "data/raw/online_retail.xlsx"

df = pd.read_excel(path)

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nFIRST ROWS")
print(df.head())

print("\nMISSING VALUES")
print(df.isnull().sum())
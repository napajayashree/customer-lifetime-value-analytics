import pandas as pd
import pickle

model = pickle.load(open("models/ml_models/randomforest.pkl","rb"))

features = [
"Recency",
"Frequency",
"Monetary",
"PurchaseFrequency",
"AvgBasketSize",
"AvgPurchaseValue",
"CustomerLifetime"
]

importance = model.feature_importances_

df = pd.DataFrame({
"Feature": features,
"Importance": importance
}).sort_values("Importance", ascending=False)

print(df)
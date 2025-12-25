import pandas as pd

df = pd.read_csv("data/heart_disease_data.csv")

# 70% reference (historique)
reference = df.sample(frac=0.7, random_state=42)

# 30% current (production simulée)
current = df.drop(reference.index)

reference.to_csv("monitoring/data/reference_data.csv", index=False)
current.to_csv("monitoring/data/current_data.csv", index=False)

print("✅ Reference & Current datasets créés")

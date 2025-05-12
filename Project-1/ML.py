import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load your FIFA dataset
df = pd.read_csv("cleaned_fifa_data.csv")  # Use your cleaned version

# ---- Simulate Win Rate (if you don't have it) ----
# Example: simulate based on Overall, Potential, Stamina
df['WinRate'] = (df['Overall'] * 0.5 + df['Potential'] * 0.3 + df['Stamina'] * 0.2) / 100

# Features to train on
features = ['Overall', 'Potential', 'Stamina', 'SprintSpeed', 'Strength']
df = df.dropna(subset=features + ['WinRate'])  # Drop rows with missing values

X = df[features]
y = df['WinRate']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and data
with open("winrate_model.pkl", "wb") as f:
    pickle.dump(model, f)

df.to_pickle("fifa_data.pkl")  # Save full dataset too

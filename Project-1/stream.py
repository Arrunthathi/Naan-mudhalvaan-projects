import streamlit as st
import pandas as pd
import pickle

# Load model and data
@st.cache_data
def load_model():
    with open("winrate_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_pickle("fifa_data.pkl")

model = load_model()
df = load_data()

st.title("⚽ FIFA Player Summary + Win Rate Predictor")

# Dropdown to select player
player_names = sorted(df['Name'].unique())
player_name = st.selectbox("Select a Player", player_names)

# Lookup player
player_row = df[df['Name'] == player_name]

if not player_row.empty:
    st.subheader("🔍 Player Summary")

    # Define which columns are "mandatory" to display
    mandatory_cols = [
        'Name', 'Age', 'Nationality', 'Club', 'Position', 'Overall', 'Potential',
        'Value', 'Wage', 'Preferred Foot', 'International Reputation'
    ]

    display_row = player_row[mandatory_cols].transpose()
    st.dataframe(display_row, use_container_width=True)

    # Prepare features for prediction
    features = ['Overall', 'Potential', 'Stamina', 'SprintSpeed', 'Strength']
    X_input = player_row[features]

    # Predict win rate
    predicted_rate = model.predict(X_input)[0] * 100  # Convert to %
    st.subheader("🎯 Predicted Win Rate")
    st.success(f"{predicted_rate:.2f}%")
else:
    st.error("❌ Player not found.")

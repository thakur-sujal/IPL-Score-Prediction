import streamlit as st
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Load trained model ---
model = load_model("ipl_score_model.keras")

# --- Load label encoders ---
with open("team_encoder.pkl", "rb") as f:
    team_encoder = pickle.load(f)

with open("venue_encoder.pkl", "rb") as f:
    venue_encoder = pickle.load(f)

# --- Define teams and venues (for UI only) ---
teams = team_encoder.classes_.tolist()
venues = venue_encoder.classes_.tolist()

# --- Streamlit App UI ---
st.set_page_config(
    page_title="🏏 IPL Score Predictor", page_icon="🏏", layout="centered"
)
st.title("🏏 IPL Score Prediction App")
st.markdown("Predict the final score based on match situation and venue!")

st.markdown("---")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("🏏 Batting Team", teams)
with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", teams)

venue = st.selectbox("🏟 Venue", venues)

bat_team_encoded = team_encoder.transform([batting_team])[0]
bowl_team_encoded = team_encoder.transform([bowling_team])[0]
venue_encoded = venue_encoder.transform([venue])[0]

st.markdown("### 📊 Current Match Stats")

runs = st.number_input("Current Runs Scored", min_value=0, step=1)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)

col3, col4 = st.columns(2)
with col3:
    completed_overs = st.number_input(
        "Completed Overs", min_value=0, max_value=19, step=1
    )
with col4:
    balls = st.number_input("Balls in Current Over", min_value=0, max_value=5, step=1)

overs = completed_overs + balls / 6.0

# --- Automatically calculate and show Current Run Rate ---
if overs > 0:
    crr = runs / overs
else:
    crr = 0.0
st.metric(label="📈 Current Run Rate", value=f"{crr:.2f} runs/over")

# --- Control inputs for last 5 overs ---
if overs < 5:
    st.warning("⚠️ Last 5 overs data disabled: match is before 5th over.")
    runs_last_5 = 0
    wickets_last_5 = 0
else:
    runs_last_5 = st.number_input("Runs Scored in Last 5 Overs", min_value=0, step=1)
    wickets_last_5 = st.number_input(
        "Wickets Lost in Last 5 Overs", min_value=0, max_value=10, step=1
    )

# --- Predict button ---
if st.button("🚀 Predict Final Score"):

    if wickets == 10:
        st.error("❌ Prediction unavailable: The batting team is all out.")
    else:
        input_features = np.array(
            [
                [
                    bat_team_encoded,
                    bowl_team_encoded,
                    venue_encoded,
                    runs,
                    wickets,
                    overs,
                    runs_last_5,
                    wickets_last_5,
                ]
            ]
        )
        predicted_score = model.predict(input_features)[0][0]
        st.success(f"🎯 Predicted Final Score: **{predicted_score:.0f}**")

st.markdown("---")
st.markdown("<small>Built by SUJAL SURYANSH HARDIK</small>", unsafe_allow_html=True)

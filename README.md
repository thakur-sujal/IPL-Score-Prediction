# 🏏 IPL Score Prediction App

This project is a deep learning-based web app built using **Keras**, **Streamlit**, and **Scikit-learn** that predicts the **final score of a T20 match (IPL)** based on match conditions, team stats, and venue.

---

## 🔍 Project Overview

The application predicts the total score that a team is likely to score in an IPL match given:

- Batting team
- Bowling team
- Venue
- Current score
- Wickets fallen
- Overs completed
- Runs and wickets in the last 5 overs

---

## 📂 Project Structure

```
📁 IPL-Score-Prediction/
├── app.py                    # Streamlit web app for prediction
├── ipl_score_prediction.py   # Model training and evaluation script
├── ipl.csv                   # Dataset used for training
├── venv/                     # (Auto-created) Virtual environment (optional)
├── ipl_score_model.keras     # (Auto-created) Trained Keras model after running training script
├── team_encoder.pkl          # (Auto-created) Label encoder for teams
├── venue_encoder.pkl         # (Auto-created) Label encoder for venues
├── terminal.txt              # (Optional) Output logs or console prints
```

> **NOTE:** The main files required to start the project are:
>
> - `app.py`
> - `ipl_score_prediction.py`
> - `ipl.csv`
>
> Other files (`.keras`, `.pkl`, `venv/`) are generated automatically when you run the scripts.

---

## 🚀 How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/your-thakur-sujal/IPL-Score-Prediction.git
cd IPL-Score-Prediction
```

### 2. Install the dependencies

Make sure you have Python 3.7+ installed. Then:

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Sample <code>requirements.txt</code></summary>

```text
streamlit
numpy
pandas
scikit-learn
matplotlib
keras
tensorflow
```

</details>

### 3. Run the training script (only once)

```bash
python ipl_score_prediction.py
```

This will:

- Train the model using the dataset
- Save the trained model and encoders

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Dataset

- The dataset `ipl.csv` contains over-by-over data from IPL T20 matches.
- Unnecessary columns are dropped (`batsman`, `bowler`, etc.) before model training.
- Categorical features are encoded using `LabelEncoder`.

---

## ✅ Features

- Real-time final score prediction
- Dynamic input for current stats
- Auto-disable for last 5 overs data if < 5 overs
- Run rate calculation
- All out check with proper messaging

---

## 📌 Future Improvements

- Win/loss prediction for second innings
- Target-based scoring comparison
- Advanced visualizations
- Match summary export

---

## 👨‍💻 Authors

- Sujal Meena

---

## 🙌 Acknowledgements

- IPL Dataset inspired by [GeeksforGeeks ](www.geeksforgeeks.org)
- Built with using Python and Streamlit

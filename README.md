# 🎧 Streamlit Playlist Generator (Mood-Based)

Generate music playlists based on your mood and audio preferences using machine learning!

---

## 🧠 Overview

This app uses a pre-trained mood classification model and multiple `NearestNeighbors` models to recommend songs from a Spotify dataset. The mood is predicted using a `.keras` model, and recommendations are pulled from pre-trained mood-specific models (`.pkl`), all CPU-optimized.

---

## 🌟 Features

- Predicts top 2 moods based on user input
- Retrieves songs in a 70:30 mood split
- Uses Spotify audio features: `danceability`, `energy`, `speechiness`, `tempo`, `acousticness`, and `valence`
- Clean UI with Streamlit sliders and formatted playlist output
- Runs entirely on CPU (no GPU required)

---

## 🗂️ Contents

- `app.py` – Streamlit application
- `mood_decider_float32_CPU.keras` – Pre-trained mood classifier
- `model_<mood>.pkl` – NearestNeighbors models for each mood
- `MusicMoodFinal.parquet` – Optimized Spotify audio feature dataset
- `requirements.txt` – Python dependencies

---

## 💻 How It Works

1. **User Input**: App collects audio preferences through sliders (e.g., danceability, tempo).
2. **Mood Prediction**: A `.keras` model predicts the top 2 probable moods.
3. **Recommendations**: Songs are retrieved using k-NN (from respective mood models), with a 70:30 split.
4. **Display**: Personalized playlist is shown in a ranked list.

---

## 🖼 UI Preview

- Sliders for feature input
- Number of recommendations selector
- Stylish playlist section showing artist and track title

---

## 📌 Requirements

- Python 3.8+
- Streamlit
- scikit-learn
- pandas
- numpy
- tensorflow (for `.keras` model)

---

## 🚀 Deployment Ready

- All models are pre-trained and saved
- App is CPU-only (no CUDA/GPU required)
- Can be hosted on Streamlit Cloud, Render, or any cloud VM with Python



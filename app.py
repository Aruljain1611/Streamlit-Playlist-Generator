import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


@st.cache_resource
def load_mood_model():
    return tf.keras.models.load_model('mood_decider_float32_CPU.keras')

mood_model = load_mood_model()

@st.cache_resource
def load_nn_models():
    return {
        'Happy': joblib.load('model_happy.pkl'),
        'Sad': joblib.load('model_sad.pkl'),
        'Energetic': joblib.load('model_energetic.pkl'),
        'Calm': joblib.load('model_calm.pkl')
    }

nn_models = load_nn_models()

@st.cache_resource
def load_scaler_model():
    return joblib.load('input_scaler.pkl')

scaler = load_scaler_model()

def load_songs_df():
    return pd.read_parquet('MusicMoodFinal.parquet')

songs_df = load_songs_df()

st.title("🎵 Mood-Based Playlist Generator")
st.markdown("**Enter your song preferences below:**")


danceability = st.slider("Danceability",0.0, 1.0, 0.5, help="How suitable a track is for dancing based on tempo, rhythm, and beat strength. Higher = more danceable.")
energy = st.slider("Energy",0.0, 1.0, 0.5,help="Represents intensity and activity. High energy means loud, fast, and noisy.")
speechiness = st.slider("Speechiness",0.0, 1.0, 0.3,help="Detects the presence of spoken words. Values closer to 1.0 indicate more talking (e.g., podcasts or rap).")
tempo = st.number_input("Tempo (BPM)",50.0, 200.0, 100.0,help="The speed of the track in Beats Per Minute. Typical pop is 90-120 BPM.")
acoustic_intensity = st.number_input("Acoustic Intensity (dB)",-60.0, 0.0, -5.0,help="Perceived loudness. Closer to 0 is louder; lower values are softer or more acoustic.")
mood_score = st.slider("Mood Score",0.0, 1.0, 0.5,help="Overall emotional vibe. 0 is sad/serious, 1 is happy/uplifting.")
count = st.number_input("Number of Recommendations", min_value = 5, max_value = 50, value=10)

if st.button("Generate Playlist"):
    input_vector = np.array([[danceability, energy, speechiness, tempo, acoustic_intensity, mood_score]])
    input_vector_scaled = scaler.transform(input_vector)
    proba = mood_model.predict(input_vector_scaled)[0]
    print(proba)
    mood_labels = ['Calm','Energetic','Happy','Sad']
    top2_indices = np.argsort(proba)[::-1][:2]
    mood1, mood2 = [mood_labels[i] for i in top2_indices]
    conf1, conf2 = proba[top2_indices]
    print(mood1,mood2)

    st.success(f"Top Moods: {mood1.capitalize()} & {mood2.capitalize()}")
    
    num_mood1 = int(0.7 * count)
    num_mood2 = count - num_mood1

    def get_recommendations(mood, count_mood):
        nn_model = nn_models[mood]
        mood_songs = songs_df[songs_df['Mood'] == mood].reset_index(drop=True)
        distances, indices = nn_model.kneighbors(input_vector, n_neighbors=min(count_mood, len(mood_songs)))
        return mood_songs.iloc[indices[0]]


    songs_1 = get_recommendations(mood1, num_mood1)
    songs_2 = get_recommendations(mood2, num_mood2)

    final_playlist = pd.concat([songs_1, songs_2], ignore_index=True)

    st.subheader("🎧 Your Personalized Playlist")
    for i, row in final_playlist.iterrows():
        st.write(f"{i+1}. **{row['name']}** — *{row['artists']}*")


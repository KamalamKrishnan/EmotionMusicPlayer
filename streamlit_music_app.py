import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import tempfile
import pygame
import threading
import time
from mutagen.mp3 import MP3

# Initialize pygame mixer
pygame.mixer.init()

# Emotion-to-song mapping
emotion_to_song = {
    'happy': 'music/happy.mp3',
    'sad': 'music/sad.mp3',
    'neutral': 'music/neutral.mp3'
}

# State variables
current_emotion = None
last_played_time = 0
emotion_window = []
current_song_name = ""
song_duration = 0
start_time = 0

# Function to play music


def play_emotion_song(emotion):
    global current_emotion, last_played_time, current_song_name, song_duration, start_time

    if emotion not in emotion_to_song:
        return

    if emotion != current_emotion or time.time() - last_played_time > 15:
        current_emotion = emotion
        last_played_time = time.time()
        song_path = emotion_to_song[emotion]
        current_song_name = song_path.split("/")[-1]

        audio = MP3(song_path)
        song_duration = int(audio.info.length)
        start_time = time.time()

        def play_song():
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()

        threading.Thread(target=play_song, daemon=True).start()


# Streamlit UI
st.set_page_config(page_title="Emotion Music Player", layout="centered")
st.title("🎵 Emotion-Based Music Player")

FRAME_WINDOW = st.image([])
progress_placeholder = st.empty()

run = st.checkbox('Start Webcam')
cap = None

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to read from webcam")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_detected = False

        try:
            result = DeepFace.analyze(
                frame_rgb, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]

            if 'dominant_emotion' in result:
                dominant_emotion = result['dominant_emotion']

                if dominant_emotion in emotion_to_song:
                    emotion_window.append(dominant_emotion)
                    if len(emotion_window) > 10:
                        emotion_window.pop(0)

                    smoothed_emotion = max(
                        set(emotion_window), key=emotion_window.count)
                    play_emotion_song(smoothed_emotion)
                    cv2.putText(frame_rgb, f"Emotion: {smoothed_emotion}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    face_detected = True
        except:
            pass

        if not face_detected:
            cv2.putText(frame_rgb, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        FRAME_WINDOW.image(frame_rgb)

        # Progress bar for music
        if song_duration > 0:
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / song_duration)
            progress_placeholder.progress(
                progress, text=f"Now Playing: {current_song_name}")

    cap.release()
else:
    if cap:
        cap.release()
    st.info("Turn on the checkbox above to start the webcam.")

import cv2
from deepface import DeepFace
import threading
import time
import pygame
import os
from collections import deque

# Initialize pygame
pygame.mixer.init()

# Emotion-song map
emotion_to_song = {
    'happy': 'music/happy.mp3',
    'sad': 'music/sad.mp3',
    'neutral': 'music/neutral.mp3'
}

current_emotion = None
last_played_time = 0
current_song_name = ""
song_duration = 1

# Emotion smoothing queue
emotion_queue = deque(maxlen=10)

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get song duration


def get_song_duration(path):
    try:
        from mutagen.mp3 import MP3
        audio = MP3(path)
        return int(audio.info.length)
    except:
        return 30

# Play emotion-based song


def play_emotion_song(emotion):
    global current_emotion, last_played_time, current_song_name, song_duration

    if emotion not in emotion_to_song:
        return

    if emotion != current_emotion or time.time() - last_played_time > 15:
        current_emotion = emotion
        last_played_time = time.time()
        song_path = emotion_to_song[emotion]
        current_song_name = os.path.basename(song_path)
        song_duration = get_song_duration(song_path)

        def play():
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()

        threading.Thread(target=play, daemon=True).start()


# Webcam
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_frame = cv2.resize(frame, (640, 480))

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_detected = False
    stable_emotion = None

    if len(faces) > 0:
        if frame_count % 5 == 0:  # Analyze every 5 frames
            try:
                result = DeepFace.analyze(
                    small_frame, actions=['emotion'],
                    enforce_detection=False
                )
                result = result[0] if isinstance(result, list) else result
                dominant_emotion = result.get("dominant_emotion", "")

                # Optional: check confidence
                if result["emotion"].get(dominant_emotion, 0) >= 50:  # ≥50% confidence
                    emotion_queue.append(dominant_emotion)
            except Exception as e:
                print("DeepFace error:", e)

        if emotion_queue:
            # Use most common recent emotion
            stable_emotion = max(set(emotion_queue), key=emotion_queue.count)
            if stable_emotion in emotion_to_song:
                display_text = f"Emotion: {stable_emotion}"
                play_emotion_song(stable_emotion)
                cv2.putText(frame, display_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                emotion_detected = True
    else:
        emotion_queue.clear()  # Clear queue if no face detected

    if not emotion_detected:
        cv2.putText(frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Progress bar
    if pygame.mixer.music.get_busy():
        pos_ms = pygame.mixer.music.get_pos()
        elapsed = pos_ms // 1000
        progress = int((elapsed / song_duration) * 500)
        cv2.rectangle(frame, (10, 70), (510, 100), (200, 200, 200), 2)
        cv2.rectangle(frame, (10, 70), (10 + progress, 100), (0, 128, 255), -1)
        cv2.putText(frame, f"{elapsed}s / {song_duration}s", (520, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Show window
    cv2.imshow("Emotion-Based Music Player", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

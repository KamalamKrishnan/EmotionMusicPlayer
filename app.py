import cv2
from deepface import DeepFace
import threading
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Supported emotions and their songs
emotion_to_song = {
    'happy': 'music/happy.mp3',
    'sad': 'music/sad.mp3',
    'neutral': 'music/neutral.mp3'
}

# Track last played
current_emotion = None
last_played_time = 0
current_song_name = ""


def play_emotion_song(emotion):
    global current_emotion, last_played_time, current_song_name

    if emotion not in emotion_to_song:
        return  # Ignore unsupported emotions

    # Play only if new emotion or 15s passed
    if emotion != current_emotion or time.time() - last_played_time > 15:
        current_emotion = emotion
        last_played_time = time.time()
        song_path = emotion_to_song[emotion]
        current_song_name = song_path.split("/")[-1]

        def play_song():
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()

        threading.Thread(target=play_song, daemon=True).start()


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze face for emotion
        result = DeepFace.analyze(
            frame, actions=['emotion'], enforce_detection=False)
        detected_emotion = result[0]['dominant_emotion']

        # Only allow supported emotions
        if detected_emotion in emotion_to_song:
            play_emotion_song(detected_emotion)

            # Show detected emotion
            display_text = f"Emotion: {detected_emotion.upper()}"
            cv2.putText(frame, display_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Show currently playing track
            song_text = f" Now Playing: {current_song_name}"
            cv2.putText(frame, song_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            # Don't show anything if emotion is unsupported (e.g. angry)
            pass

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Emotion-Based Music Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

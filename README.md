# 🎧 Emotion-Based Music Player using OpenCV + DeepFace

Play songs that match your **real-time facial emotions**!  
This AI-powered music player uses your **webcam** to detect emotions like **Happy**, **Sad**, or **Neutral**, and plays appropriate music automatically.

---

## 😊 Emotions Supported

- 😀 **Happy** → Plays upbeat songs  
- 😐 **Neutral** → Plays calm songs  
- 😢 **Sad** → Plays emotional tracks  

> Your face decides the vibe! 🎶

---

## 🔥 Features

- 🎥 Real-time face detection via webcam  
- 🧠 Emotion detection using **DeepFace** (or **FER**)  
- 🎵 Automatically plays emotion-matched songs  
- ⏸️ Auto-pause when **no face detected**  
- 📊 Song progress bar  
- 📈 Emotion confidence level display  
- 🚨 "No face detected" alert  

---

## 📦 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt

Or manually install:

pip install opencv-python deepface pygame numpy

import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
from keras.models import load_model
from gtts import gTTS
import pygame
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import tempfile

# Khởi tạo pygame cho âm thanh
pygame.mixer.init()

# Load mô hình
model = load_model("C:/Users/DELL/Desktop/code/models/emotion_cnn.h5")

# Danh sách nhãn biểu cảm và câu nói tiếng Việt
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']
emotion_speech = {
    'Angry': "Đừng giận dữ như vậy mà. Hãy ngồi xuống và hít thở thật sâu!",
    'Fear': "Bạn đang sợ điều gì à?",
    'Happy': "Có vẻ bạn có một ngày tuyệt vời nhỉ. Hãy lan tỏa nó tới mọi người nào!",
    'Sad': "Đừng buồn nhé. Có mình đây rồi.",
    'Surprise': "Ồ, có gì làm bạn ngạc nhiên vậy?",
}

# Trạng thái âm thanh
audio_enabled = True
last_speak_time = 0
speak_interval = 5

def play_audio(text):
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(temp_fd)  # Đóng file descriptor để tránh lỗi trên Windows
    try:
        tts = gTTS(text=text, lang='vi')
        tts.save(temp_path)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    finally:
        os.remove(temp_path)  # Xóa file sau khi phát xong


# Hàm phát giọng nói bằng luồng riêng để tránh lag
def speak(text):
    if audio_enabled:
        threading.Thread(target=play_audio, args=(text,), daemon=True).start()

# Hàm dự đoán biểu cảm
def predict_emotion(face_image):
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=(0, -1))
    predictions = model.predict(face_image)
    return emotion_labels[np.argmax(predictions)]

# Hàm cập nhật video
def update_frame():
    global last_speak_time
    ret, frame = cap.read()
    if not ret:
        return

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    current_time = time.time()

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face)

        if emotion in emotion_speech and (current_time - last_speak_time >= speak_interval):
            speak(emotion_speech[emotion])
            last_speak_time = current_time

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk
    root.after(10, update_frame)

# Hàm bật/tắt âm thanh
def toggle_audio():
    global audio_enabled
    audio_enabled = not audio_enabled
    audio_button.config(text="🔊 ON" if audio_enabled else "🔇 OFF")

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Emotion Recognition")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

audio_button = Button(root, text="🔊 ON", command=toggle_audio, font=("Arial", 12))
audio_button.pack()

# Mở webcam
cap = cv2.VideoCapture(0)
update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()

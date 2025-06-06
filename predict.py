import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np

# Load mô hình đã train
model = load_model("C:/Users/DELL/Desktop/code/models/emotion_cnn.h5")

# Danh sách ánh xạ nhãn thành tên biểu cảm
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

def predict_emotion(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tại đường dẫn: {image_path}")
    image = cv2.resize(image, (48, 48)) / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    prediction = model.predict(image)
    emotion_label = np.argmax(prediction)
    return emotion_label

# Đường dẫn ảnh
image_path = "C:/Users/DELL/Desktop/code/data/test/happy/PrivateTest_95094.jpg"

# Dự đoán biểu cảm
emotion_index = predict_emotion(image_path)
emotion_name = emotion_labels[emotion_index]

# Hiển thị biểu cảm dưới dạng chữ
print("Biểu cảm dự đoán là:", emotion_name)

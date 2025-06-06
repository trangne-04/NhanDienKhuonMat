import tensorflow as tf
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import load_model

# Đường dẫn đến thư mục chứa ảnh test
test_dir = "C:/Users/DELL/Desktop/code/data/test"

# Danh sách nhãn cảm xúc
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

# Load mô hình đã huấn luyện
model = load_model("C:/Users/DELL/Desktop/code/models/emotion_mobilenet.h5")

# Load dữ liệu ảnh từ thư mục test
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

# Lấy ảnh và nhãn thực tế
X_test = np.concatenate([x for x, y in test_dataset], axis=0)  # Ảnh test
y_test = np.concatenate([y for x, y in test_dataset], axis=0)  # Nhãn thật

# Chuyển ảnh grayscale (48,48,1) thành RGB (48,48,3)
X_test = np.repeat(X_test, 3, axis=-1)

# Chuẩn hóa ảnh về [0,1]
X_test = X_test / 255.0  

# Dự đoán nhãn từ mô hình
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"🔹 Accuracy của mô hình: {accuracy * 100:.2f}%")

# In báo cáo chi tiết (Precision, Recall, F1-score)
print(classification_report(y_test, y_pred, target_names=emotion_labels))

# Hiển thị Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Chuẩn hóa để tạo ma trận tỷ lệ (phần trăm)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 🔹 Đổi màu `cmap` thành "YlGnBu" (hoặc màu khác bạn thích)
custom_cmap = "YlGnBu"  # Thử đổi thành "coolwarm", "magma", "viridis" nếu muốn

# Vẽ biểu đồ Confusion Matrix theo số lượng
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap=custom_cmap, xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MobileNet (Số lượng)")
plt.show()

# Vẽ biểu đồ Confusion Matrix theo phần trăm (%)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap=custom_cmap, xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("MobileNet (Tỷ lệ %)")
plt.show()

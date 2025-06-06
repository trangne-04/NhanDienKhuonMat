import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.models import load_model

# Đường dẫn đến thư mục chứa ảnh test
test_dir = r"C:/Users/DELL/Desktop/code/data/test"

# Danh sách nhãn theo mô hình đã huấn luyện
emotion_labels = ['angry', 'fear', 'happy', 'sad', 'surprise']

# Load mô hình đã huấn luyện
model = load_model("C:/Users/DELL/Desktop/code/models/emotion_cnn.h5")
print("\nMô hình đã được load thành công!")

# Load dữ liệu ảnh từ thư mục test
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    image_size=(48, 48),
    color_mode="grayscale",  # Nếu lỗi, đổi thành "rgb"
    batch_size=32,
    shuffle=False  # Không shuffle để giữ nguyên thứ tự ảnh
)

# Kiểm tra thứ tự nhãn của TensorFlow dataset
class_names = test_dataset.class_names
print("\nThứ tự nhãn trong TensorFlow dataset:", class_names)

# Đảm bảo thứ tự nhãn của dataset khớp với mô hình
if sorted(class_names) != sorted(emotion_labels):
    raise ValueError("Thứ tự nhãn của TensorFlow dataset không khớp với mô hình!")

# Lấy dữ liệu ảnh và nhãn thực tế
X_test = np.concatenate([x for x, y in test_dataset], axis=0)
y_test = np.concatenate([y for x, y in test_dataset], axis=0)

# Kiểm tra batch đầu tiên có bị lỗi không
for images, labels in test_dataset.take(1):
    print("\nKiểm tra batch đầu tiên:")
    print("Batch shape:", images.shape)  
    print("Labels:", labels.numpy())  

# Chuẩn hóa ảnh về [0,1]
X_test = X_test / 255.0  

# Dự đoán nhãn từ mô hình
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Tính độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Accuracy của mô hình: {accuracy * 100:.2f}%")

# In báo cáo chi tiết
print("\n Báo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=emotion_labels))

# Hiển thị Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(emotion_labels)))
print("\n Confusion Matrix:")
print(cm)

# Kiểm tra các lỗi cụ thể trong dự đoán
misclassified_counts = np.bincount(y_test[y_test != y_pred], minlength=len(emotion_labels))
print("\n Số lượng ảnh bị dự đoán sai theo từng nhãn:")
for label, count in zip(emotion_labels, misclassified_counts):
    print(f"{label}: {count} ảnh")

# Chuẩn hóa để tạo ma trận tỷ lệ
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Vẽ biểu đồ Confusion Matrix theo số lượng
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("CNN (Số lượng)", fontsize=14)
plt.show()

# Vẽ biểu đồ Confusion Matrix theo phần trăm
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("CNN (Tỷ lệ %)", fontsize=14)
plt.show()

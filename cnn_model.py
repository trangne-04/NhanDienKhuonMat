import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn đến thư mục dữ liệu
data_dir = "C:/Users/DELL/Desktop/code/data/train"

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    # Chuyển đổi ảnh sang grayscale nếu chưa phải grayscale
    if len(image.shape) == 3:  # Nếu ảnh có 3 kênh màu (RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Giảm nhiễu bằng Gaussian Blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Cân bằng histogram để cải thiện độ tương phản
    image = cv2.equalizeHist(image)
    
    # Thay đổi kích thước ảnh về 48x48
    image = cv2.resize(image, (48, 48))
    
    return image

# Hàm load dữ liệu
def load_data(data_dir):
    X = []
    y = []

    label_mapping = {label: idx for idx, label in enumerate(sorted(os.listdir(data_dir)))}
    print("Thứ tự nhãn khi train:", label_mapping)  # In ra để kiểm tra


    # Duyệt qua các thư mục trong data_dir
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if not os.path.isdir(folder_path):
            continue
        label_id = label_mapping[label]  # Chuyển tên thư mục thành số nhãn
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)  # Đọc ảnh
            if image is None:  # Bỏ qua ảnh lỗi
                continue
            image = preprocess_image(image)  # Áp dụng tiền xử lý
            X.append(image)
            y.append(label_id)

    # Chuyển danh sách thành numpy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y

# Load dữ liệu
X, y = load_data(data_dir)

# Chuẩn hóa dữ liệu (ảnh grayscale từ 0-255 về 0-1)
X = X / 255.0

# Thêm kênh màu (grayscale chỉ có 1 kênh)
X = np.expand_dims(X, axis=-1)

# One-hot encode nhãn
y = to_categorical(y, num_classes=5)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tăng cường dữ liệu
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Xây dựng mô hình CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 6 lớp cho 6 biểu cảm
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình với dữ liệu tăng cường
model.fit(
    data_gen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=70
)

# Tạo thư mục lưu mô hình nếu chưa tồn tại
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

# Lưu mô hình đã huấn luyện
model.save(os.path.join(model_dir, "emotion_cnn.h5"))
print("Mô hình đã được lưu thành công!")
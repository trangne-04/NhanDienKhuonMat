import os
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục dữ liệu
data_dir = "C:/Users/DELL/Desktop/code/data/train"

# Hàm tiền xử lý ảnh
def preprocess_image(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.equalizeHist(image)
    image = cv2.resize(image, (48, 48))
    return image

# Hàm load dữ liệu
def load_data(data_dir):
    X, y = [] , []
    label_mapping = {'angry': 0, 'fear': 1, 'happy': 2, 'sad': 3, 'surprise': 4}
    
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if not os.path.isdir(folder_path):
            continue
        label_id = label_mapping[label]
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = preprocess_image(image)
            X.append(image)
            y.append(label_id)
    
    X = np.array(X) / 255.0  # Chuẩn hóa
    X = np.expand_dims(X, axis=-1)  # Thêm kênh màu
    y = to_categorical(y, num_classes=6)
    return X, y

# Load dữ liệu
X, y = load_data(data_dir)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Tăng cường dữ liệu
data_gen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Load mô hình MobileNetV2 (dùng input shape 48x48x3, nhưng ảnh gốc grayscale nên phải chuyển thành 3 kênh)
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights=None)

# Thêm các lớp fully connected
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(6, activation='softmax')(x)

# Tạo mô hình mới
model = Model(inputs=base_model.input, outputs=output_layer)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(
    data_gen.flow(np.repeat(X_train, 3, axis=-1), y_train, batch_size=64),
    validation_data=(np.repeat(X_val, 3, axis=-1), y_val),
    epochs=20
)

# Lưu mô hình
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "emotion_mobilenet.h5"))
print("Mô hình MobileNetV2 đã được lưu thành công!")

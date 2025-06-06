import os
import matplotlib.pyplot as plt
import numpy as np

# Đường dẫn thư mục dữ liệu train & test
train_dir = "C:/Users/DELL/Desktop/code/data/train"
test_dir = "C:/Users/DELL/Desktop/code/data/test"

# Hàm đếm số lượng ảnh trong từng thư mục nhãn
def count_images(data_dir):
    label_counts = {}
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):  # Chỉ xét thư mục (bỏ qua file rác)
            label_counts[label] = len(os.listdir(folder_path))  # Đếm số ảnh trong thư mục
    return label_counts

# Đếm số lượng ảnh trong từng nhãn của tập train & test
train_counts = count_images(train_dir)
test_counts = count_images(test_dir)

# Gộp tổng số lượng ảnh của cả train & test
all_labels = sorted(set(train_counts.keys()).union(set(test_counts.keys())))
total_counts = [train_counts.get(label, 0) + test_counts.get(label, 0) for label in all_labels]

# Chọn màu xanh nhạt từ colormap 'Blues'
light_blue = plt.cm.Blues(0.6)  # Giá trị 0.6 để có màu xanh nhạt

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 6))
plt.bar(all_labels, total_counts, color=light_blue)
plt.xlabel("Biểu cảm", fontsize=12)
plt.ylabel("Số lượng ảnh", fontsize=12)
plt.title("Mức độ phân tán tập trung của dữ liệu FER2013", fontsize=14)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Hiển thị số lượng trên từng cột
for i, count in enumerate(total_counts):
    plt.text(i, count + 5, str(count), ha='center', fontsize=10, fontweight='bold')

plt.show()

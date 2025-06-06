import cv2
import os

# Đường dẫn lưu dữ liệu
base_dir = "C:/Users/DELL/Desktop/code/datapython realtime_prediction.py"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Danh sách nhãn biểu cảm
labels = ["angry", "fear", "happy", "sad", "surprise"]

# Tạo các thư mục nếu chưa tồn tại
for label in labels:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

# Hàm thu thập dữ liệu
def collect_data(label, num_samples=100):
    cap = cv2.VideoCapture(0)  # Mở webcam
    count = 0

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc hình ảnh từ webcam!")
            break

        # Hiển thị khung hình và hướng dẫn
        cv2.imshow("Collecting data - Press 'q' to quit", frame)
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tiền xử lý: Giảm nhiễu Gaussian
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Tiền xử lý: Cân bằng histogram
        gray = cv2.equalizeHist(gray)
        
        # Resize ảnh về 48x48
        resized = cv2.resize(gray, (48, 48))
        
        # Chia dữ liệu thành train và test
        if count < int(num_samples * 0.8):  # 80% cho train
            save_path = os.path.join(train_dir, label, f"{label}_{count}.jpg")
        else:  # 20% cho test
            save_path = os.path.join(test_dir, label, f"{label}_{count}.jpg")

        cv2.imwrite(save_path, resized)
        count += 1

        print(f"Đã lưu (sau tiền xử lý): {save_path}")

        # Thoát nếu bấm phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Thu thập dữ liệu cho từng biểu cảm
for label in labels:
    print(f"Bắt đầu thu thập dữ liệu cho biểu cảm: {label}")
    num_samples = int(input(f"Nhập số lượng mẫu cho biểu cảm '{label}': "))
    collect_data(label, num_samples)

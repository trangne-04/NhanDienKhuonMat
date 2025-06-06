🧠Hệ Thống Nhận Diện Khuôn Mặt & Cảm Xúc Thời Gian Thực
Hệ thống nhận diện khuôn mặt và phân tích cảm xúc theo thời gian thực sử dụng OpenCV, CNN, MobileNet và tập dữ liệu FER2013. Dự án hỗ trợ huấn luyện, đánh giá độ chính xác, dự đoán theo thời gian thực và trực quan hóa đặc trưng khuôn mặt bằng t-SNE.

📁 Cấu Trúc Thư Mục Mã Nguồn (src/)
Tên tệp Python	Mô tả chức năng
accuracy.py	Tính độ chính xác của mô hình CNN
accuracy_mobilenet.py	Tính độ chính xác của mô hình MobileNet
cnn_model.py	Xây dựng và huấn luyện mô hình CNN với dữ liệu FER2013
mobilenet_model.py	Xây dựng và huấn luyện mô hình MobileNet để nhận diện khuôn mặt
collect_data.py	Thu thập và xử lý dữ liệu khuôn mặt để huấn luyện
predict.py	Dự đoán khuôn mặt hoặc cảm xúc từ ảnh tĩnh
realtime_prediction.py	Nhận diện khuôn mặt và cảm xúc theo thời gian thực (sử dụng webcam)
t-sne.py	Trực quan hóa đặc trưng khuôn mặt bằng thuật toán t-SNE

🛠️ Hướng Dẫn Sử Dụng
1. Cài Đặt Môi Trường
bash
Copy
Edit
pip install -r requirements.txt
Các thư viện thường dùng:

text
Copy
Edit
opencv-python
tensorflow / keras
numpy
matplotlib
scikit-learn
2. Huấn Luyện Mô Hình
CNN với FER2013:

bash
Copy
Edit
python cnn_model.py
MobileNet:

bash
Copy
Edit
python mobilenet_model.py
Gợi ý: Dữ liệu huấn luyện được chuẩn bị trong collect_data.py.

3. Đánh Giá Mô Hình
bash
Copy
Edit
python accuracy.py
python accuracy_mobilenet.py
4. Dự Đoán
Ảnh tĩnh:

bash
Copy
Edit
python predict.py --image path_to_image.jpg
Thời gian thực với webcam:

bash
Copy
Edit
python realtime_prediction.py
5. Trực Quan Hóa Đặc Trưng (t-SNE)
bash
Copy
Edit
python t-sne.py
Xuất biểu đồ 2D thể hiện sự phân tách của các đặc trưng khuôn mặt.

📌 Ghi Chú
Thư mục models/ nên chứa các mô hình đã huấn luyện (*.h5, *.pb, v.v.)

Tập dữ liệu FER2013: https://www.kaggle.com/datasets/msambare/fer2013

📄 Giấy Phép
Dự án được thực hiện bởi Nguyễn Thị Trang-CNTT 1501

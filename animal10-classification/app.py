import gradio as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
from cnns import AlexNet
from argparse import ArgumentParser

# ==========================================
# 1. CẤU HÌNH MÔ HÌNH (Thay đổi theo model của bạn)
# ==========================================
def get_args():
    parser = ArgumentParser(description= "CNN testing")
    parser.add_argument("--images_size", type= int, default = 224, help = "Image size")
    parser.add_argument("--image_dir", type = str )
    parser.add_argument("--checkpoint", type= str, default = "./trained_model/best_cnn.pt")
    args = parser.parse_args()
    return args
# Định nghĩa 10 lớp của bộ dữ liệu Animal 10
# (Hãy đảm bảo thứ tự này trùng với thứ tự lúc bạn train mô hình)
LABELS = [
    "Butterfly (Bướm)",
    "Cat (Mèo)",
    "Chicken (Gà)",
    "Cow (Bò)",
    "Dog (Chó)",
    "Elephant (Voi)",
    "Horse (Ngựa)",
    "Sheep (Cừu)",
    "Spyder (Nhện)",
    "Squirrel (Sóc)",
]
args = get_args()
# Khởi tạo kiến trúc mô hình của bạn ở đây
# Ví dụ: model = MyCustomCNN() hoặc sử dụng torchvision models có sẵn
# Ở đây giả định bạn dùng một mô hình ResNet hoặc Custom đã có:
# model = torch.load('path_to_architecture') 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device  = torch.device("cpu")
print(f"Using device: {device}")
model = AlexNet(num_classes=10).to(device)
if args.checkpoint :
    checkpoint = torch.load(args.checkpoint, map_location= "cpu")
    model.load_state_dict(checkpoint["model"])
# Tạm thời load weights (Bạn thay thế bằng code load model thực tế của bạn nhé)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 10) # 10 classes

# # LOAD TRỌNG SỐ BẠN ĐÃ HUẤN LUYỆN
# # model.load_state_dict(torch.load("animal10_model.pth", map_location=device))
# model.to(device)
model.eval()

# Pipeline tiền xử lý ảnh (phải giống lúc bạn train)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], 
#         std=[0.229, 0.224, 0.225]
#     )
# ])
transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.images_size, args.images_size) )
       
    ])

# ==========================================
# 2. HÀM DỰ ĐOÁN (PREDICT FUNCTION)
# ==========================================
def predict_animal(image):
    if image is None:
        return None
    
    # Tiền xử lý ảnh đầu vào
    # img_t = transform(image).unsqueeze(0).to(device)
    image = transform(image).to(device)
    img_t = image[None,:,:,:]
    # Dự đoán không tính gradient
    with torch.no_grad():
        outputs = model(img_t)
        # Tính xác suất bằng hàm Softmax
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Trả về một dictionary chứa {Tên_Lớp: Xác_Suất} để Gradio tự vẽ biểu đồ tỉ lệ
    return {LABELS[i]: float(probabilities[i]) for i in range(10)}

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN WEB INTERACTIVE (GRADIO UI)
# ==========================================

# Tùy chỉnh CSS để giao diện trông hiện đại, mượt mà và thu hút hơn
custom_css = """
body { background-color: #f0f4f8; }
.gradio-container { max-width: 850px !important; margin: 40px auto !important; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); background: white; padding: 20px; }
h1 { color: #1e293b; text-align: center; font-weight: 800 !important; }
.feedback { font-size: 1.1rem; }
"""

# Tạo giao diện Block cao cấp của Gradio
with nn.Blocks(css=custom_css, title="Animal 10 Classification Demo") as demo:
    nn.Markdown(
        """
        # 🐾 Animal 10 Image Classifier Demo
        ### Tải lên một hình ảnh động vật bất kỳ để nhận diện loài và xem xác suất phân tích từ mô hình AI.
        """
    )
    
    with nn.Row():
        with nn.Column(scale=1):
            # Khung upload ảnh mượt mà, hỗ trợ kéo thả và cả webcam nếu muốn
            input_img = nn.Image(type="pil", label="Hình ảnh đầu vào")
            clear_btn = nn.Button("Xóa ảnh", variant="stop")
            submit_btn = nn.Button("Dự đoán loài vật 🚀", variant="primary")
            
        with nn.Column(scale=1):
            # Khung hiển thị kết quả xác suất dạng biểu đồ thanh (bar) trực quan, tự sắp xếp từ cao đến thấp
            output_label = nn.Label(num_top_classes=5, label="Kết quả dự đoán (Top 5)")
            
    # Thiết lập các sự kiện tương tác (Interactive)
    submit_btn.click(fn=predict_animal, inputs=input_img, outputs=output_label)
    clear_btn.click(fn=lambda: (None, None), inputs=None, outputs=[input_img, output_label])
    
    # Tính năng tương tác cao: Thả ảnh vào tự động chạy dự đoán luôn không cần bấm nút nếu thích
    input_img.change(fn=predict_animal, inputs=input_img, outputs=output_label)

# Khởi chạy local
if __name__ == "__main__":
    demo.launch()
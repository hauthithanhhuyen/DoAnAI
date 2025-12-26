import cv2
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from fer import FER
import numpy as np
import base64
import os
from datetime import datetime

app = Flask(__name__)

# --- 1. CẤU HÌNH DATABASE ---
database_url = os.environ.get('DATABASE_URL')

if database_url:
    # Cấu hình cho Render (PostgreSQL)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # Cấu hình cho máy cá nhân (SQLite)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. ĐỊNH NGHĨA MODEL ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    emotion = db.Column(db.String(100)) 
    confidence = db.Column(db.Float)
    ip_address = db.Column(db.String(50))
    username = db.Column(db.String(100)) 

# Tạo bảng
with app.app_context():
    try:
        db.create_all()
        print("✅ KẾT NỐI DATABASE THÀNH CÔNG!")
    except Exception as e:
        print(f"⚠️ LỖI KẾT NỐI DB: {e}")

# --- 3. KHỞI TẠO AI (QUAN TRỌNG: mtcnn=False) ---
# Dùng mtcnn=False để sử dụng Haar Cascade (Nhẹ hơn nhiều, phù hợp Render Free)
detector = FER(mtcnn=False) 

emotion_dict = {
    "happy": "Hạnh phúc 😊", "sad": "Buồn 😢", "angry": "Giận dữ 😡",
    "surprise": "Ngạc nhiên 😲", "fear": "Sợ hãi 😨", "disgust": "Ghê tởm 🤢", "neutral": "Bình thường 😐"
}

# --- 4. HÀM XỬ LÝ ẢNH THÔNG MINH ---
def analyze_emotion(frame, ip_addr, user_name="Khách vãng lai"):
    try:
        # MẸO TỐI ƯU: Thu nhỏ ảnh xuống còn 40% để AI chạy nhanh
        scale_factor = 0.4
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        
        # Nhận diện trên ảnh nhỏ
        results = detector.detect_emotions(small_frame)

        if results:
            # Lấy cảm xúc cao nhất
            top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
            score = results[0]["emotions"][top_emotion]
            vn_label = emotion_dict.get(top_emotion, top_emotion)
            
            # Lấy tọa độ từ ảnh nhỏ
            (x, y, w, h) = results[0]["box"] 

            # PHÓNG TO TỌA ĐỘ LẠI (Để vẽ đúng lên ảnh gốc)
            real_box = [
                int(x / scale_factor),
                int(y / scale_factor),
                int(w / scale_factor),
                int(h / scale_factor)
            ]

            # Lưu vào Database nếu độ tin cậy > 50%
            if score > 0.5:
                try:
                    # Kiểm tra xem có vừa lưu chưa để tránh spam DB (tùy chọn)
                    # Ở đây lưu luôn cho đơn giản
                    new_record = History(
                        emotion=vn_label,
                        confidence=score,
                        ip_address=ip_addr,
                        username=user_name
                    )
                    db.session.add(new_record)
                    db.session.commit()
                    # Print log ngắn gọn để không rác console
                    print(f"💾 Saved: {vn_label} ({score:.2f})")
                except Exception as e:
                    print(f"DB Error: {e}")

            return {
                'has_face': True, 
                'emotion': vn_label, 
                'box': real_box, 
                'score': score
            }
        else:
            return {'has_face': False, 'emotion': 'Không tìm thấy mặt'}
            
    except Exception as e:
        print(f"AI Error: {e}")
        return {'has_face': False, 'emotion': 'Lỗi xử lý'}

# --- 5. CÁC ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        records = History.query.order_by(History.timestamp.desc()).limit(50).all()
        return render_template('history.html', records=records)
    except Exception as e:
        return f"<h3>Lỗi đọc dữ liệu: {e}</h3>"

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data: 
            return jsonify({'status': 'error', 'msg': 'No image'})

        user_name = data.get('username', 'Khách vãng lai')
        
        # Giải mã ảnh Base64 từ JS gửi về
        image_data = data['image'].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'status': 'error', 'msg': 'Invalid image'})

        # Gọi hàm phân tích
        result = analyze_emotion(frame, request.remote_addr, user_name)
        return jsonify(result)
        
    except Exception as e:
        print(f"Process Error: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files: 
            return jsonify({'error': 'No file'})
        file = request.files['file']
        
        user_name = request.form.get('username', 'Khách vãng lai')
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        result = analyze_emotion(frame, request.remote_addr, user_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

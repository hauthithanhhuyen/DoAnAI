import cv2
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from fer import FER
import numpy as np
import base64
import os
from datetime import datetime

app = Flask(__name__)

# --- CẤU HÌNH DATABASE THÔNG MINH ---
# 1. Kiểm tra xem có đang chạy trên Render không? (Render sẽ cung cấp biến DATABASE_URL)
database_url = os.environ.get('DATABASE_URL')

if database_url:
    # --- CẤU HÌNH CHO RENDER (PostgreSQL) ---
    # Sửa lỗi nhỏ của Render (nó trả về postgres:// nhưng thư viện cần postgresql://)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    # --- CẤU HÌNH CHO MÁY CÁ NHÂN (SQLite) ---
    # Dùng SQLite cho nhẹ, không cần cài SQL Server phức tạp
    # Nó sẽ tự tạo ra file 'history.db' ngay trong thư mục code
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///history.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- ĐỊNH NGHĨA MODEL (BẢNG DỮ LIỆU) ---
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # PostgreSQL và SQLite lưu tiếng Việt tốt với String thường, không cần NVARCHAR
    emotion = db.Column(db.String(100)) 
    confidence = db.Column(db.Float)
    ip_address = db.Column(db.String(50))
    username = db.Column(db.String(100)) 

# Tự động tạo bảng dữ liệu
with app.app_context():
    try:
        db.create_all()
        print("✅ KẾT NỐI DATABASE THÀNH CÔNG!")
    except Exception as e:
        print(f"⚠️ LỖI KẾT NỐI DB: {e}")

# Khởi tạo mô hình AI (MTCNN)
detector = FER(mtcnn=True) 

emotion_dict = {
    "happy": "Hạnh phúc 😊", "sad": "Buồn 😢", "angry": "Giận dữ 😡",
    "surprise": "Ngạc nhiên 😲", "fear": "Sợ hãi 😨", "disgust": "Ghê tởm 🤢", "neutral": "Bình thường 😐"
}

# --- CÁC HÀM XỬ LÝ (GIỮ NGUYÊN NHƯ CŨ) ---
def analyze_emotion(frame, ip_addr, user_name="Khách vãng lai"):
    try:
        results = detector.detect_emotions(frame)
        if results:
            top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
            score = results[0]["emotions"][top_emotion]
            vn_label = emotion_dict.get(top_emotion, top_emotion)
            box = results[0]["box"] 

            # Chỉ lưu nếu độ tin cậy > 50%
            if score > 0.5:
                try:
                    new_record = History(
                        emotion=vn_label,
                        confidence=score,
                        ip_address=ip_addr,
                        username=user_name
                    )
                    db.session.add(new_record)
                    db.session.commit()
                    print(f"💾 Đã lưu: {user_name} - {vn_label}")
                except Exception as e:
                    print(f"Lỗi lưu DB: {e}")

            return {'has_face': True, 'emotion': vn_label, 'box': box, 'score': score}
        else:
            return {'has_face': False, 'emotion': 'Không tìm thấy mặt'}
    except Exception as e:
        print(f"AI Error: {e}")
        return {'has_face': False, 'emotion': 'Lỗi nhận diện'}

# --- CÁC ROUTES ---

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
        if not data or 'image' not in data: return jsonify({'status': 'error'})

        user_name = data.get('username', 'Khách vãng lai')
        image_data = data['image'].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = analyze_emotion(frame, request.remote_addr, user_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error'}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'})
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
    # Chạy trên tất cả các IP để Ngrok hoặc điện thoại có thể truy cập
    app.run(host='0.0.0.0', port=5000, debug=True)
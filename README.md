# 🚀 Hướng Dẫn Cài Đặt & Sử Dụng Dự Án

## 1. Yêu Cầu Hệ Thống
- Python **3.9+**
- Git

---

## 2. Cài Đặt Môi Trường

### a. Clone repository
```bash
git clone <URL_CỦA_REPOSITORY_CỦA_BẠN>
cd <TÊN_THƯ_MỤC_PROJECT>
```

### b. Tạo và kích hoạt môi trường ảo
Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### c. Cài đặt thư viện cần thiết
```bash
pip install -r requirements.txt
```

## 3. Hướng Dẫn Sử Dụng
Tất cả các script phải được chạy từ thư mục gốc của dự án.

### a. Tiền xử lý dữ liệu
```bash
# Cấp quyền thực thi (chỉ cần thực hiện một lần)
chmod +x scripts/preprocess.sh

# Chạy tiền xử lý dữ liệu
./scripts/preprocess.sh healthcare_stroke
```

### b. Huấn Luyện & Đánh Giá Mô Hình
```bash
chmod +x scripts/run_exps.sh ${tên thí nghiệm}
```
ví dụ chạy baseline:

```bash
./scripts/run_exps.sh xgboost_baseline
./scripts/run_exps.sh tabtransformer_baseline
```

### c. Đánh giá trên tập test
```bash
chmod +x scripts/evaluate.sh
./scripts/evaluate.sh xgboost_tuned
```
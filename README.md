# 🚀 Hướng Dẫn Cài Đặt & Sử Dụng Dự Án

## A. Uni-tabular data

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

## B. Multimodal data ()
Đây là quy trình để chạy các thí nghiệm multi-modal mới, kết hợp dữ liệu dạng bảng và dữ liệu hình ảnh.

### 1. Tiền xử lý Dữ liệu

Bước này chỉ cần chạy **một lần** cho mỗi bộ dữ liệu. Nó sẽ xử lý các giá trị thiếu, mã hóa các cột hạng mục, và chia dữ liệu thành các tập `train`, `validation`, và `test`.

**Cú pháp:**

```bash
./scripts/preprocess_multimodal.sh <tên_config_data>
```
Ví dụ
```bash
./scripts/preprocess_preprocess_multimodal.sh multimodal_stroke_v1
```
Kết quả sẽ được lưu vào thư mục `data/processed/multimodal_stroke/`.

### 2. Huấn luyện Mô hình

Sau khi đã tiền xử lý, bạn có thể chạy các thí nghiệm huấn luyện. Script sẽ tự động tạo một thư mục kết quả duy nhất dựa trên tên của các file cấu hình.

**Cú pháp:**
```bash
./scripts/run_multi_modal_exp.sh \
  --model configs/models/<tên_config_model>.yaml \
  --data configs/data/<tên_config_data>.yaml \
  --trainer configs/trainers/<tên_config_trainer>.yaml
```

Ví dụ (Chạy thí nghiệm Fusion Transformer):
```bash
./scripts/run_multi_modal_exp.sh \
  --model configs/models/fusion_vit_tabtransformer.yaml \
  --data configs/data/multimodal_stroke_v1.yaml \
  --trainer configs/trainers/adamw_bce_sqrt.yaml
```

Kết quả huấn luyện, bao gồm checkpoint của mô hình tốt nhất, sẽ được lưu tại results/experiments/<tên_thí_nghiệm>/.

### 3. Đánh giá trên Tập Test

```bash
./scripts/evaluate_multimodal.sh \
  --model configs/models/<tên_config_model>.yaml \
  --data configs/data/<tên_config_data>.yaml \
  --trainer configs/trainers/<tên_config_trainer>.yaml
```

Ví dụ (Đánh giá mô hình Fusion Transformer đã huấn luyện):

```bash
./scripts/evaluate_multimodal.sh \
  --model configs/models/fusion_vit_tabtransformer.yaml \
  --data configs/data/multimodal_stroke_v1.yaml \
  --trainer configs/trainers/adamw_bce_sqrt.yaml
```
Kết quả đánh giá cuối cùng sẽ được lưu trong thư mục con test_evaluation bên trong thư mục thí nghiệm tương ứng.

## C. Uni-Image data ()
Đây là quy trình để chạy các thí nghiệm với dữ liệu chỉ Image

### 1. Tiền xử lý Dữ liệu


**Cú pháp:**

```bash
./scripts/preprocess_image.sh <tên_config_data>
```
Ví dụ
```bash
./scripts/preprocess_image.sh image_only_stroke
```
Kết quả sẽ được lưu vào thư mục `data/processed/image_only_stroke/`.

### 2. Huấn luyện Mô hình

Sau khi đã tiền xử lý, bạn có thể chạy các thí nghiệm huấn luyện. Script sẽ tự động tạo một thư mục kết quả duy nhất dựa trên tên của các file cấu hình.

**Cú pháp:**
```bash
./scripts/run_image_exp.sh \
  --model configs/models/<tên_config_model>.yaml \
  --data configs/data/<tên_config_data>.yaml \
  --trainer configs/trainers/<tên_config_trainer>.yaml
```

Ví dụ (Chạy thí nghiệm ViT):
```bash
./scripts/run_image_exp.sh \
  --model configs/models/vit_classifier.yaml \
  --data configs/data/image_only_stroke.yaml \
  --trainer configs/trainers/adamw_bce_sqrt.yaml
```

Kết quả huấn luyện, bao gồm checkpoint của mô hình tốt nhất, sẽ được lưu tại results/experiments/<tên_thí_nghiệm>/.

### 3. Đánh giá trên Tập Test

```bash
./scripts/evaluate_image.sh \
  --model configs/models/<tên_config_model>.yaml \
  --data configs/data/<tên_config_data>.yaml \
  --trainer configs/trainers/<tên_config_trainer>.yaml
```

Ví dụ (Đánh giá mô hình Fusion Transformer đã huấn luyện):

```bash
./scripts/evaluate_image.sh \
  --model configs/models/vit_classifier.yaml \
  --data configs/data/image_only_stroke.yaml \
  --trainer configs/trainers/adamw_bce_sqrt.yaml
```
Kết quả đánh giá cuối cùng sẽ được lưu trong thư mục con test_evaluation bên trong thư mục thí nghiệm tương ứng.
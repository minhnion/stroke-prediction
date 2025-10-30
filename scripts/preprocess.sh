set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của file cấu hình dataset."
  echo "Ví dụ: $0 healthcare_stroke"
  exit 1
fi

DATASET_NAME=$1
CONFIG_FILE="configs/data/${DATASET_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Lỗi: Không tìm thấy file cấu hình tại đường dẫn: ${CONFIG_FILE}"
  exit 1
fi

echo "============================================="
echo "Bắt đầu tiền xử lý cho dataset: ${DATASET_NAME}"
echo "Sử dụng file cấu hình: ${CONFIG_FILE}"
echo "============================================="

python src/data/preprocessing.py --config "${CONFIG_FILE}"

echo "============================================="
echo "Tiền xử lý cho dataset '${DATASET_NAME}' đã hoàn tất thành công!"
echo "============================================="
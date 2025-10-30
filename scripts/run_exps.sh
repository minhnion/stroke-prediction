set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của file cấu hình thí nghiệm."
  echo "Ví dụ: $0 tabtransformer_baseline"
  exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_FILE="configs/experiments/${EXPERIMENT_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Lỗi: Không tìm thấy file cấu hình tại đường dẫn: ${CONFIG_FILE}"
  exit 1
fi

echo "============================================="
echo "Bắt đầu chạy thí nghiệm: ${EXPERIMENT_NAME}"
echo "Sử dụng file cấu hình: ${CONFIG_FILE}"
echo "============================================="

python -m src.experiments.run_experiment --config "${CONFIG_FILE}"

echo "============================================="
echo "Thí nghiệm '${EXPERIMENT_NAME}' đã hoàn tất thành công!"
echo "============================================="
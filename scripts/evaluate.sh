set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của thí nghiệm đã hoàn thành để đánh giá."
  echo "Ví dụ: $0 xgboost_tuned"
  exit 1
fi

EXPERIMENT_NAME=$1
CONFIG_FILE="configs/experiments/${EXPERIMENT_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Lỗi: Không tìm thấy file cấu hình của thí nghiệm: ${CONFIG_FILE}"
  exit 1
fi

echo "============================================="
echo "Bắt đầu đánh giá CUỐI CÙNG trên TẬP TEST cho: ${EXPERIMENT_NAME}"
echo "============================================="

python -m src.experiments.evaluate --name "${EXPERIMENT_NAME}"

echo "============================================="
echo "Đánh giá cho '${EXPERIMENT_NAME}' đã hoàn tất."
echo "Kết quả được lưu tại: results/experiments/${EXPERIMENT_NAME}/test_evaluation/"
echo "============================================="
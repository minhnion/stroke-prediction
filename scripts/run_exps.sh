set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của file cấu hình thí nghiệm."
  echo "Ví dụ: $0 xgboost_baseline"
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


if grep -q "experiment_type: ml" "$CONFIG_FILE"; then
    echo "Phát hiện loại thí nghiệm: Machine Learning (ml)"
    PYTHON_SCRIPT="src.experiments.run_ml_experiment"

elif grep -q "experiment_type: dl" "$CONFIG_FILE"; then
    echo "Phát hiện loại thí nghiệm: Deep Learning (dl)"
    PYTHON_SCRIPT="src.experiments.run_experiment"
else
    echo "Lỗi: Không tìm thấy hoặc không nhận diện được trường 'experiment_type' trong file ${CONFIG_FILE}."
    echo "Vui lòng thêm 'experiment_type: ml' hoặc 'experiment_type: dl' vào file config."
    exit 1
fi

echo "Thực thi lệnh: python -m ${PYTHON_SCRIPT} --config ${CONFIG_FILE}"
python -m "${PYTHON_SCRIPT}" --config "${CONFIG_FILE}"


echo "============================================="
echo "Thí nghiệm '${EXPERIMENT_NAME}' đã hoàn tất thành công!"
echo "============================================="
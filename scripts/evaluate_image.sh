#!/bin/bash

# =====================================================================================
# Script để đánh giá MỘT LẦN DUY NHẤT một mô hình CHỈ ẢNH đã huấn luyện
# trên TẬP TEST.
#
# Cách sử dụng:
# ./scripts/evaluate_image.sh --model <path> --data <path> --trainer <path>
#
# Ví dụ:
# ./scripts/evaluate_image.sh \
#   --model configs/models/vit_classifier.yaml \
#   --data configs/data/image_only_stroke.yaml \
#   --trainer configs/trainers/adamw_bce_sqrt.yaml
# =====================================================================================

set -e

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --data)
      DATA_CONFIG="$2"
      shift 2
      ;;
    --trainer)
      TRAINER_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$MODEL_CONFIG" ] || [ -z "$DATA_CONFIG" ] || [ -z "$TRAINER_CONFIG" ]; then
    echo "Lỗi: Vui lòng cung cấp đầy đủ các tham số --model, --data, và --trainer."
    exit 1
fi

MODEL_NAME_BASE=$(basename -s .yaml "$MODEL_CONFIG")
DATA_NAME_BASE=$(basename -s .yaml "$DATA_CONFIG")
TRAINER_NAME_BASE=$(basename -s .yaml "$TRAINER_CONFIG")
EXPERIMENT_NAME="${MODEL_NAME_BASE}_${DATA_NAME_BASE}_${TRAINER_NAME_BASE}"

echo "============================================="
echo "Bắt đầu đánh giá CUỐI CÙNG trên TẬP TEST cho: ${EXPERIMENT_NAME}"
echo "============================================="

python -m src.experiments.evaluate_image \
    --model "${MODEL_CONFIG}" \
    --data "${DATA_CONFIG}" \
    --trainer "${TRAINER_CONFIG}"

echo "============================================="
echo "Đánh giá cho '${EXPERIMENT_NAME}' đã hoàn tất."
echo "Kết quả được lưu tại: results/experiments/${EXPERIMENT_NAME}/test_evaluation/"
echo "============================================="
#!/bin/bash

# =====================================================================================
# Script để đánh giá MỘT LẦN DUY NHẤT một mô hình DEEP LEARNING (mới)
# đã huấn luyện trên TẬP TEST.
#
# Cách sử dụng:
# ./scripts/evaluate_new.sh --model <path> --data <path> --trainer <path>
#
# Ví dụ:
# ./scripts/evaluate_new.sh \
#   --model configs/models/fusion_vit_tabtransformer.yaml \
#   --data configs/data/multimodal_stroke_v1.yaml \
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

EXPERIMENT_DIR="results/experiments/${EXPERIMENT_NAME}"
CHECKPOINT_PATH="${EXPERIMENT_DIR}/checkpoints/best_model.pth"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Lỗi: Không tìm thấy checkpoint cho thí nghiệm '${EXPERIMENT_NAME}'."
  echo "Vui lòng chạy thí nghiệm huấn luyện trước: ./scripts/run_new_exp.sh ..."
  exit 1
fi

echo "============================================="
echo "Bắt đầu đánh giá CUỐI CÙNG trên TẬP TEST cho: ${EXPERIMENT_NAME}"
echo "============================================="

python -m src.experiments.evaluate_multimodal \
    --model "${MODEL_CONFIG}" \
    --data "${DATA_CONFIG}" \
    --trainer "${TRAINER_CONFIG}"

echo "============================================="
echo "Đánh giá cho '${EXPERIMENT_NAME}' đã hoàn tất."
echo "Kết quả được lưu tại: ${EXPERIMENT_DIR}/test_evaluation/"
echo "============================================="
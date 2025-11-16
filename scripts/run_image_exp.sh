#!/bin/bash

# =====================================================================================
# Script để chạy các thí nghiệm phân loại CHỈ ẢNH.
#
# Cách sử dụng:
# ./scripts/run_image_exp.sh --model <path> --data <path> --trainer <path>
#
# Ví dụ:
# ./scripts/run_image_exp.sh \
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
      shift 
      shift 
      ;;
    --data)
      DATA_CONFIG="$2"
      shift 
      shift 
      ;;
    --trainer)
      TRAINER_CONFIG="$2"
      shift 
      shift 
      ;;
    *)   
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$MODEL_CONFIG" ] || [ -z "$DATA_CONFIG" ] || [ -z "$TRAINER_CONFIG" ]; then
    echo "Lỗi: Vui lòng cung cấp đầy đủ các tham số --model, --data, và --trainer."
    echo "Ví dụ: $0 --model <path> --data <path> --trainer <path>"
    exit 1
fi

echo "============================================="
echo "Bắt đầu chạy thí nghiệm Phân loại Ảnh..."
echo "Model Config:   ${MODEL_CONFIG}"
echo "Data Config:    ${DATA_CONFIG}"
echo "Trainer Config: ${TRAINER_CONFIG}"
echo "============================================="

python -m src.experiments.run_image_exp \
    --model "${MODEL_CONFIG}" \
    --data "${DATA_CONFIG}" \
    --trainer "${TRAINER_CONFIG}"

echo "============================================="
echo "Thí nghiệm đã hoàn tất!"
echo "============================================="
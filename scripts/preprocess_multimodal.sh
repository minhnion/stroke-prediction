#!/bin/bash

# =====================================================================================
# Script để chạy tiền xử lý cho các bộ dữ liệu mới (multi-modal, etc.).
#
# Cách sử dụng:
# ./scripts/preprocess_new.sh <tên_config_data>
#
# Ví dụ:
# ./scripts/preprocess_new.sh multimodal_stroke_v1
# =====================================================================================

set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của file cấu hình data."
  echo "Ví dụ: $0 multimodal_stroke_v1"
  exit 1
fi

DATA_NAME=$1
CONFIG_FILE="configs/data/${DATA_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Lỗi: Không tìm thấy file cấu hình data tại: ${CONFIG_FILE}"
  exit 1
fi

echo "============================================="
echo "Bắt đầu tiền xử lý cho dataset: ${DATA_NAME}"
echo "============================================="

python -m src.data.preprocess_multimodal --config "${CONFIG_FILE}"

echo "============================================="
echo "Tiền xử lý hoàn tất!"
echo "============================================="
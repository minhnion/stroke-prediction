#!/bin/bash

# =====================================================================================
# Script để chạy tiền xử lý cho các bộ dữ liệu CHỈ ẢNH.
# Nó sẽ chia thư mục ảnh thô và sao chép vào thư mục processed.
#
# Cách sử dụng:
# ./scripts/preprocess_image.sh <tên_config_data>
#
# Ví dụ:
# ./scripts/preprocess_image.sh image_only_stroke
# =====================================================================================

set -e

if [ -z "$1" ]; then
  echo "Lỗi: Vui lòng cung cấp tên của file cấu hình data."
  echo "Ví dụ: $0 image_only_stroke"
  exit 1
fi

DATA_NAME=$1
CONFIG_FILE="configs/data/${DATA_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Lỗi: Không tìm thấy file cấu hình data tại: ${CONFIG_FILE}"
  exit 1
fi

echo "============================================="
echo "Bắt đầu tiền xử lý ảnh cho dataset: ${DATA_NAME}"
echo "============================================="

python -m src.data.preprocess_image --config "${CONFIG_FILE}"

echo "============================================="
echo "Tiền xử lý ảnh hoàn tất!"
echo "============================================="
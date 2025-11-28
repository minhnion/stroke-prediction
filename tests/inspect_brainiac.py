# tests/inspect_brainiac.py

import torch
import os
import sys

def inspect_checkpoint(ckpt_path):
    print(f"--- INSPECTING: {ckpt_path} ---")
    
    if not os.path.exists(ckpt_path):
        print(f"ERROR: File not found at {ckpt_path}")
        return

    try:
        # Tải checkpoint lên CPU để tránh lỗi nếu không có GPU
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return

    # 1. Kiểm tra cấu trúc Dictionary của file .ckpt
    print(f"\n[1] CHECKPOINT KEYS: {list(checkpoint.keys())}")
    
    # Xác định xem trọng số thực sự nằm ở đâu
    # Thông thường PyTorch Lightning lưu trong 'state_dict', nhưng SSL có thể lưu trong 'model' hoặc 'teacher'
    state_dict = None
    if 'state_dict' in checkpoint:
        print("-> Detected PyTorch Lightning structure ('state_dict' key found).")
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        print("-> Detected Standard/SSL structure ('model' key found).")
        state_dict = checkpoint['model']
    elif 'teacher' in checkpoint:
        print("-> Detected SSL Student/Teacher structure ('teacher' key found). Using Teacher weights.")
        state_dict = checkpoint['teacher']
    else:
        print("-> Assuming raw state_dict structure.")
        state_dict = checkpoint

    # 2. Phân tích các lớp (Layer Analysis)
    print(f"\n[2] LAYER ANALYSIS (Total keys: {len(state_dict)})")
    print("Printing shape of the first 5 layers and identifying Input Conv Layer...")
    
    is_3d = False
    is_2d = False
    input_layer_found = False

    for i, (key, value) in enumerate(state_dict.items()):
        # In ra 5 layer đầu tiên để xem tên
        if i < 5:
            print(f"  - {key}: {value.shape}")

        # 3. Thuật toán phát hiện 2D hay 3D dựa trên shape của Convolution/Embedding
        # Tìm các từ khóa thường thấy ở lớp đầu vào
        if any(x in key for x in ['conv1.weight', 'patch_embed.proj.weight', 'stem.0.weight']):
            print(f"\n[3] DETECTING DIMENSIONALITY BASED ON: '{key}'")
            shape = value.shape
            print(f"  -> Weight Shape: {shape}")
            
            # Logic phân biệt:
            # Conv2D weight shape: (Out, In, H, W) -> 4 chiều
            # Conv3D weight shape: (Out, In, D, H, W) -> 5 chiều
            if value.ndim == 5:
                is_3d = True
                print(f"  -> Dimension count is 5. This suggests a **3D CONVOLUTION** (D, H, W).")
            elif value.ndim == 4:
                is_2d = True
                print(f"  -> Dimension count is 4. This suggests a **2D CONVOLUTION** (H, W).")
            
            input_layer_found = True
            # Break sau khi tìm thấy lớp đầu vào quan trọng nhất để kết luận
            break
    
    if not input_layer_found:
        print("\n[3] WARNING: Could not explicitly identify standard input layer names (conv1, patch_embed).")
        print("Scanning all weights for 5D tensors...")
        for key, value in state_dict.items():
            if value.ndim == 5:
                print(f"  -> Found 5D tensor at '{key}': {value.shape}. Likely 3D Model.")
                is_3d = True
                break
    
    # 4. Kết luận
    print("\n[4] FINAL CONCLUSION:")
    if is_3d:
        print("✅ MODEL IS PRETRAINED ON **3D DATA** (Volumes like MRI/CT).")
    elif is_2d:
        print("✅ MODEL IS PRETRAINED ON **2D DATA** (Slices/Images).")
    else:
        print("❓ Cannot determine dimensionality definitively.")

if __name__ == "__main__":
    # Đường dẫn tương đối từ thư mục gốc dự án
    ckpt_path = "model_weights/BrainIAC.ckpt"
    inspect_checkpoint(ckpt_path)
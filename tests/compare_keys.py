# tests/compare_keys.py

import torch
import timm
import sys
import os

def compare_structures(ckpt_path):
    print(f"--- COMPARING KEYS: BrainIAC vs TIMM ViT ---")
    
    # 1. Load BrainIAC Keys
    if not os.path.exists(ckpt_path):
        print(f"Error: File {ckpt_path} not found.")
        return

    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # BrainIAC là PyTorch Lightning nên weight nằm trong 'state_dict'
        brainiac_dict = checkpoint.get('state_dict', checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Lọc lấy các keys của Block 0 để dễ nhìn (đại diện cho tất cả các block)
    brainiac_keys = [k.replace('backbone.', '') for k in brainiac_dict.keys() if 'blocks.0.' in k or 'blocks.0' in k]
    brainiac_keys.sort()

    # 2. Load TIMM ViT Keys
    model_timm = timm.create_model('vit_base_patch16_224', pretrained=False)
    timm_keys = [k for k in model_timm.state_dict().keys() if 'blocks.0.' in k]
    timm_keys.sort()

    # 3. In ra so sánh
    print(f"\n{'BRAINIAC (Source)':<50} | {'TIMM (Target)':<50}")
    print("-" * 105)
    
    # In BrainIAC keys
    # Chúng ta sẽ cố gắng tìm cặp tương ứng bằng mắt
    max_len = max(len(brainiac_keys), len(timm_keys))
    
    for i in range(max_len):
        b_key = brainiac_keys[i] if i < len(brainiac_keys) else ""
        t_key = timm_keys[i] if i < len(timm_keys) else ""
        print(f"{b_key:<50} | {t_key:<50}")

    # 4. Kiểm tra các layer ngoài Block (Embeddings, Norm)
    print("\n--- OTHER LAYERS (Embeddings / Norm) ---")
    other_b = [k.replace('backbone.', '') for k in brainiac_dict.keys() if 'blocks' not in k and 'head' not in k]
    other_t = [k for k in model_timm.state_dict().keys() if 'blocks' not in k and 'head' not in k]
    
    for k in other_b:
        print(f"BrainIAC: {k}")
    print("...")
    for k in other_t:
        print(f"TIMM    : {k}")

if __name__ == "__main__":
    # Đường dẫn đến file checkpoint của bạn
    ckpt_path = "model_weights/BrainIAC.ckpt" 
    compare_structures(ckpt_path)
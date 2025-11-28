import torch
import os

WEIGHTS_DIR = "model_weights"
WEIGHT_FILES = {
    # "ResNet50": "ResNet50.pt",
    "DenseNet121": "DenseNet121.pt",
    "InceptionV3": "InceptionV3.pt"
}

def inspect_file(model_name, filename):
    path = os.path.join(WEIGHTS_DIR, filename)
    print(f"\n{'-'*20} INSPECTING: {model_name} {'-'*20}")
    
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # Unwrap dictionary
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # In ra 10 keys đầu tiên để bắt bài quy luật đặt tên
        print(f"Total keys: {len(state_dict)}")
        print("First 15 keys sample:")
        keys = list(state_dict.keys())
        for i, key in enumerate(keys):
            if i >= 15: break
            print(f"  {key}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    for name, file in WEIGHT_FILES.items():
        inspect_file(name, file)
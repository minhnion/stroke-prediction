import torch
import torch.nn as nn
import torchvision.models as models
import os

WEIGHTS_DIR = "model_weights"
WEIGHT_FILES = {
    "ResNet50": "ResNet50.pt",
    "DenseNet121": "DenseNet121.pt", 
    "InceptionV3": "InceptionV3.pt" 
}

def get_mapping_rules(model_name):

    if model_name == "ResNet50":
        return {
            'backbone.0.': 'conv1.',
            'backbone.1.': 'bn1.',
            'backbone.4.': 'layer1.',
            'backbone.5.': 'layer2.',
            'backbone.6.': 'layer3.',
            'backbone.7.': 'layer4.'
        }
    
    elif model_name == "DenseNet121":
        return {
            'backbone.0.': 'features.', 
        }
    
    elif model_name == "InceptionV3":
        inception_blocks = [
            'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 
            'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 
            'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 
            'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
            'Mixed_7a', 'Mixed_7b', 'Mixed_7c'
        ]
        
        mapping = {}

        for i, block_name in enumerate(inception_blocks):
            mapping[f'backbone.{i}.'] = f'{block_name}.'
            
        return mapping
        
    return {}

def load_and_inspect_model(model_name, weight_file):
    print(f"\n{'='*20} TESTING: {model_name} {'='*20}")
    weight_path = os.path.join(WEIGHTS_DIR, weight_file)
    
    if not os.path.exists(weight_path):
        print(f"❌ Skip: {weight_file} not found.")
        return

    # 1. Init Architecture
    print("1. Initializing Architecture...")
    try:
        if model_name == "ResNet50":
            model = models.resnet50(weights=None)
            target_dim = 2048
            input_size = 224
        elif model_name == "DenseNet121":
            model = models.densenet121(weights=None)
            target_dim = 1024
            input_size = 224
        elif model_name == "InceptionV3":
            model = models.inception_v3(weights=None, aux_logits=True)
            target_dim = 2048
            input_size = 299
    except Exception as e:
        print(f"❌ Error init model: {e}")
        return
    
    # 2. Load & Map Weights
    print(f"2. Loading weights from {weight_file}...")
    try:
        checkpoint = torch.load(weight_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: state_dict = checkpoint['model']
            else: state_dict = checkpoint
        else:
            state_dict = checkpoint

        print("   -> Applying Mapping...")
        mapping_rules = get_mapping_rules(model_name)
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('module.'): k = k[7:]
            
            new_key = k
            for old, new in mapping_rules.items():
                if k.startswith(old):
                    new_key = k.replace(old, new)
                    break
            
            new_state_dict[new_key] = v

        # 3. Load vào model
        model_state = model.state_dict()
        matched = {k: v for k, v in new_state_dict.items() if k in model_state and v.size() == model_state[k].size()}
        
        print(f"   -> Successfully matched {len(matched)} / {len(model_state)} layers.")
        
        if len(matched) < len(model_state) * 0.5:
            print("❌ WARNING: Low match rate. Check mapping rules!")
            # In ra vài key không khớp để debug
            unmatched_ckpt = list(set(new_state_dict.keys()) - set(matched.keys()))[:3]
            unmatched_model = list(set(model_state.keys()) - set(matched.keys()))[:3]
            print(f"      Unmatched Checkpoint keys: {unmatched_ckpt}")
            print(f"      Unmatched Model keys: {unmatched_model}")
        else:
            model.load_state_dict(matched, strict=False)
            print("✅ SUCCESS: Weights loaded.")

    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 4. Test Forward
    print("3. Testing Forward Pass...")
    if hasattr(model, 'fc'): model.fc = nn.Identity()
    elif hasattr(model, 'classifier'): model.classifier = nn.Identity()
    if model_name == "InceptionV3": model.aux_logits = False

    dummy_input = torch.randn(2, 3, input_size, input_size)
    try:
        out = model(dummy_input)
        # InceptionV3 output có thể là tuple nếu aux_logits chưa tắt hẳn, kiểm tra kỹ
        if isinstance(out,  torch.Tensor) == False: out = out[0]

        if out.shape[1] == target_dim:
            print(f"✅ Final Check Passed: Output shape is {out.shape}")
        else:
            print(f"❌ Dimension Mismatch: Got {out.shape}, expected (2, {target_dim})")
    except Exception as e:
        print(f"❌ Forward Error: {e}")

if __name__ == "__main__":
    for name, file in WEIGHT_FILES.items():
        load_and_inspect_model(name, file)
import open_clip

model, _, _ = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

visual = model.visual

print("=== Attributes của model.visual ===")
print([attr for attr in dir(visual) if not attr.startswith('_')])

print("\n=== Kiểm tra các thuộc tính dim phổ biến ===")
for attr in ['embed_dim', 'width', 'hidden_size', 'output_dim', 'head_hidden_size']:
    if hasattr(visual, attr):
        print(f"  {attr}: {getattr(visual, attr)}")

print("\n=== Kiểm tra trunk ===")
if hasattr(visual, 'trunk'):
    print("  Có trunk")
    trunk = visual.trunk
    for attr in ['embed_dim', 'width', 'hidden_size', 'num_features']:
        if hasattr(trunk, attr):
            print(f"    trunk.{attr}: {getattr(trunk, attr)}")
else:
    print("  Không có trunk")

print("\n=== Test forward ===")
import torch
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = visual(dummy)
print(f"  Output shape: {out.shape}")

import torch
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    trunk_out = model.visual.trunk(dummy)
print(f"Trunk output shape: {trunk_out.shape}")
import torch
import torch.nn as nn
import open_clip

class BiomedCLIPImageEncoder(nn.Module):
    def __init__(self, model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        super().__init__()
        model, _, self.preprocess = open_clip.create_model_and_transforms(model_name)
        
        # Vision encoder (ViT trunk), bỏ projection head của CLIP
        self.trunk = model.visual.trunk
        
        # Embed_dim từ ViT config
        self.embed_dim = self.trunk.embed_dim  
        
    # def forward(self, x):
    #     return self.trunk(x)
    
    # def forward_features(self, x):
    #     return self.trunk(x)

    def forward(self, x): # Đổi tên từ forward_features thành forward
        # Hugging Face CLIPVisionModel nhận đầu vào là 'pixel_values'
        outputs = self.model(pixel_values=x)
        
        # outputs.pooler_output chứa vector đại diện cho ảnh (CLS token đã qua pooler)
        # Kích thước: (Batch, Hidden_Size) -> Đúng ý chúng ta muốn
        return outputs.pooler_output
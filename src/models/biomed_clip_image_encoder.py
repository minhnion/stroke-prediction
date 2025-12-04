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
        

    def forward(self, x): 
        
        features = self.trunk(x)
        return features
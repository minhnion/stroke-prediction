import torch
import torch.nn as nn

class RadImageNetEncoder(nn.Module):
    def __init__(self, model, model_type):
        super().__init__()
        self.model = model
        
        # Xác định kích thước embedding dựa trên loại model
        if 'resnet50' in model_type:
            self.embed_dim = 2048
        elif 'densenet121' in model_type:
            self.embed_dim = 1024
        elif 'inception' in model_type:
            self.embed_dim = 2048
        else:
            # Fallback
            self.embed_dim = 1024
            
        # Thêm lớp Pooling để chuyển (Batch, C, H, W) -> (Batch, C, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Thêm lớp Flatten để chuyển (Batch, C, 1, 1) -> (Batch, C)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # 1. Đi qua backbone có sẵn
        x = self.model(x)
        
        # 2. Xử lý trường hợp Inception trả về tuple (aux logits)
        if isinstance(x, tuple):
            x = x[0]
            
        # 3. Pooling và Flatten
        x = self.pool(x)
        x = self.flatten(x)
        
        return x

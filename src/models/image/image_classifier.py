import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_embedding_dim, mlp_params):
        super().__init__()
        self.image_encoder = image_encoder

        output_dim = mlp_params.get('output_dim', 1)
        dropout = mlp_params.get('dropout', 0.0)

        self.head = nn.Sequential(
            nn.LayerNorm(image_embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(image_embedding_dim, output_dim)
        )
    
    def forward(self, batch):
        image = batch['image']
        
        features = self.image_encoder(image) 
        
        output = self.head(features)
        
        return output
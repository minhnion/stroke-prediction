import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, image_encoder, image_embedding_dim, mlp_params):
        super().__init__()
        self.image_encoder = image_encoder
        
        output_dim = mlp_params.get('output_dim', 1)

        hidden_dims = mlp_params.get('hidden_dims', []) 
        dropout = mlp_params.get('dropout', 0.0)

        layers = []
        input_dim = image_embedding_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            
            layers.append(nn.BatchNorm1d(h_dim)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            
            input_dim = h_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, batch):
        image = batch['image']
        features = self.image_encoder(image) 
        output = self.head(features)
        return output
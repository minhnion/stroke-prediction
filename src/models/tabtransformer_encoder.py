import torch
import torch.nn as nn

class TabTransformerEncoder(nn.Module):
    def __init__(self, *, categories, num_continuous, dim, depth, heads):
        super().__init__()
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, dim) for num_categories in categories
        ])
        
        # Continuous layer norm
        self.cont_norm = nn.LayerNorm(num_continuous) if num_continuous > 0 else nn.Identity()

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_categ, x_cont):
        # Embed categorical features
        cat_embs = [emb(x_categ[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]
        x = torch.cat(cat_embs, dim=1)
        
        # Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(x)
        cls_features = transformer_output[:, 0] # Get CLS token representation

        # Process continuous features
        cont_features = self.cont_norm(x_cont)
        
        # Concatenate CLS features and continuous features
        return torch.cat((cls_features, cont_features), dim=1)
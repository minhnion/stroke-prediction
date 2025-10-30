import torch.nn as nn
from tab_transformer_pytorch import TabTransformer

class TabTransformerModel(nn.Module):
    def __init__(
        self,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_act=nn.ReLU(),
        continuous_mean_std=None
    ):
        super().__init__()
        
        self.tab_transformer = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_out=dim_out,
            mlp_hidden_mults=mlp_hidden_mults,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            mlp_act=mlp_act,
            continuous_mean_std=continuous_mean_std
        )
        
    def forward(self, x_categ, x_cont):
        return self.tab_transformer(x_categ, x_cont)
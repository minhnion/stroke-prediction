import torch
import torch.nn as nn
import timm
from src.models.tabtransformer_encoder import TabTransformerEncoder

def create_image_encoder(name, params):
    if name == "timm_vit":
        model = timm.create_model(
            params['model_name'],
            pretrained=params['pretrained'],
            num_classes=0  
        )
        
        embedding_dim = model.embed_dim
        
        # Nếu có checkpoint tùy chỉnh, load nó
        if params.get('pretrained_checkpoint_path'):
            print(f"Loading custom image checkpoint from: {params['pretrained_checkpoint_path']}")
            model.load_state_dict(torch.load(params['pretrained_checkpoint_path']))
            
        return model, embedding_dim
    
    # elif name == "timm_convnext":
    #     ...
    
    else:
        raise ValueError(f"Unknown image encoder name: {name}")


def create_tabular_encoder(name, params, data_config):
    if name == "tab_transformer":

        num_cols = data_config['numerical_features']
        
        if 'categories' not in params:
             raise ValueError("'categories' (cardinalities) must be provided in params for TabTransformer")

        model = TabTransformerEncoder(
            categories=params['categories'],
            num_continuous=len(num_cols),
            dim=params['dim'],
            depth=params['depth'],
            heads=params['heads'],
        )
        
        embedding_dim = params['dim'] + len(num_cols)
        
        return model, embedding_dim
        
    # elif name == "ft_transformer":
    #     ...
        
    else:
        raise ValueError(f"Unknown tabular encoder name: {name}")
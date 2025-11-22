import torch
import torch.nn as nn
import timm
import logging
from src.models.tabtransformer_encoder import TabTransformerEncoder
from src.models.biomed_clip_image_encoder import BiomedCLIPImageEncoder

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
    
    elif name == "radimagenet_hub":
        repo = 'Warvito/radimagenet-models'
        model_name = params.get('model_name', 'radimagenet_resnet50')
        logging.info(f"Downloading/Loading {model_name} from torch.hub ({repo})...")
        model = torch.hub.load(repo, model_name, verbose=True, trust_repo=True)

        if "resnet50" in model_name:
            embedding_dim = 2048
        elif "resnet18" in model_name:
            embedding_dim = 512
        elif "dense" in model_name:
            embedding_dim = 1024
        elif "inception" in model_name:
            embedding_dim = 2048
        else:
            embedding_dim = 2048  # default fallback

        model = nn.Sequential(
            model,                        # Backbone
            nn.AdaptiveAvgPool2d((1, 1)), # Current: [Batch, 2048, 7, 7].  -> [Batch, 2048, 1, 1]
            nn.Flatten()                  # Flatten vector -> [Batch, 2048]
        )
        
        logging.info(f"RadImageNet model loaded as Backbone + Pooling. Output dim: {embedding_dim}")        

        return model, embedding_dim
    
    elif name == "biomedclip_hf":
        hf_model_name = params.get('model_name', "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model = BiomedCLIPImageEncoder(model_name=hf_model_name)
        embedding_dim = model.embed_dim
        logging.info(f"Successfully loaded BiomedCLIP. Embedding dim: {embedding_dim}")
        return model, embedding_dim
    
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
import torch
import torch.nn as nn
import timm
import logging
from src.models.tabtransformer_encoder import TabTransformerEncoder
from src.models.biomed_clip_image_encoder import BiomedCLIPImageEncoder
from src.trainers.fine_tuning import apply_finetuning_strategy
from src.utils import load_weights
from src.models.radimagenet_encoder import RadImageNetEncoder

def create_image_encoder(name, params):
    
    model = None
    embedding_dim = 0

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

    if name == "timm_cnn":
        model_name = params['model_name']

        is_pretrained = params.get('pretrained', False) 
        checkpoint_path = params.get('pretrained_checkpoint_path')

        logging.info(f"Creating CNN model: {model_name} (ImageNet Pretrained: {is_pretrained})")
        
        model = timm.create_model(model_name, pretrained=is_pretrained, num_classes=0)
        embedding_dim = model.num_features
        

        if checkpoint_path:
            logging.info(f"Overriding weights with custom checkpoint: {checkpoint_path}")
            model = load_weights(model, checkpoint_path, model_name)
        return model, embedding_dim

    elif name == "timm_cnn":
        logging.info(f"Creating Generic TIMM CNN: {params['model_name']} (Pretrained ImageNet: {params['pretrained']})")
        model = timm.create_model(
            params['model_name'],
            pretrained=params['pretrained'],
            num_classes=0 
        )
        embedding_dim = model.num_features

    elif name == "radimagenet_hub":
        repo = 'Warvito/radimagenet-models'
        model_name = params.get('model_name', 'radimagenet_resnet50')
        
        logging.info(f"Loading {model_name} directly from torch.hub ({repo})...")
        
        try:
            # Tải backbone từ Hub
            raw_model = torch.hub.load(repo, model_name, verbose=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load from torch.hub: {e}")

        # Bọc backbone vào trong Wrapper của chúng ta để xử lý output
        model = RadImageNetEncoder(raw_model, model_name)
        
        logging.info(f"Successfully loaded {model_name}. Added AvgPool + Flatten. Embedding dim: {model.embed_dim}")
        
        return model, model.embed_dim
    
    elif name == "biomedclip_hf":
        hf_model_name = params.get('model_name', "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        model = BiomedCLIPImageEncoder(model_name=hf_model_name)
        embedding_dim = model.embed_dim
        logging.info(f"Successfully loaded BiomedCLIP. Embedding dim: {embedding_dim}")
    
    else:
        raise ValueError(f"Unknown image encoder name: {name}")
    
    if 'fine_tuning' in params:
        model = apply_finetuning_strategy(model, params['fine_tuning'])
    
    return model, embedding_dim


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
        
    else:
        raise ValueError(f"Unknown tabular encoder name: {name}")
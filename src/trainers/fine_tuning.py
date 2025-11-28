import torch
import torch.nn as nn
import logging
from peft import get_peft_model, LoraConfig, TaskType

def apply_finetuning_strategy(model, config):
    method = config.get('method', 'full')
    
    logging.info(f"Applying fine-tuning strategy: {method}")

    # --- 1. FULL FINE-TUNING (Mặc định) ---
    if method == 'full':
        for param in model.parameters():
            param.requires_grad = True
        return model

    # --- 2. FREEZING (Đóng băng) ---
    elif method == 'freeze':
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze last N blocks/layers
        unfreeze_last_blocks = config.get('params', {}).get('unfreeze_last_blocks', 0)
        
        if unfreeze_last_blocks > 0:
            logging.info(f"Unfreezing the last {unfreeze_last_blocks} blocks/layers...")

            total_layers = len(list(model.named_children()))
            layers_to_train = list(model.children())[-unfreeze_last_blocks:]
            
            for layer in layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = True
        
        return model

    # --- 3. LoRA (Low-Rank Adaptation) ---
    elif method == 'lora':
        lora_params = config.get('params', {})
        
        peft_config = LoraConfig(
            r=lora_params.get('r', 16),           # Rank
            lora_alpha=lora_params.get('alpha', 16),
            target_modules=lora_params.get('target_modules', ["qkv"]), # Layer names to apply LoRA
            lora_dropout=lora_params.get('dropout', 0.1),
            bias="none",

            modules_to_save=lora_params.get('modules_to_save', []),
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() 
        return model

    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")
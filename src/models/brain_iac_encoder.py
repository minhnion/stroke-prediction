import logging
import torch
def load_brainiac_3d_to_2d(model_2d, checkpoint_path):
    """
    Hàm chuyển trọng số BrainIAC 3D -> TIMM 2D.
    Cập nhật mapping chính xác dựa trên kết quả compare_keys.py.
    """
    logging.info(f"Converting BrainIAC 3D weights from {checkpoint_path} to 2D...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    new_state_dict = {}
    model_2d_dict = model_2d.state_dict()
    
    for key, value in state_dict.items():
        # 1. Xóa prefix backbone
        new_key = key.replace('backbone.', '')
        
        # 2. MAPPING CHÍNH XÁC (Dựa trên kết quả compare)
        if 'patch_embedding.position_embeddings' in new_key: 
            new_key = new_key.replace('patch_embedding.position_embeddings', 'pos_embed')
        elif 'patch_embedding.patch_embeddings' in new_key: 
            new_key = new_key.replace('patch_embedding.patch_embeddings', 'patch_embed.proj')
        elif 'mlp.linear1' in new_key: 
            new_key = new_key.replace('mlp.linear1', 'mlp.fc1')
        elif 'mlp.linear2' in new_key: 
            new_key = new_key.replace('mlp.linear2', 'mlp.fc2')
        
        # --- ĐÂY LÀ CHỖ SỬA QUAN TRỌNG NHẤT ---
        elif 'attn.out_proj' in new_key: 
            new_key = new_key.replace('attn.out_proj', 'attn.proj')
        # --------------------------------------

        # Kiểm tra tồn tại
        if new_key not in model_2d_dict:
            continue

        target_shape = model_2d_dict[new_key].shape

        # 3. Xử lý Shape (Conv3D -> 2D, Channel 1->3, Interpolate Pos)
        
        # Patch Embedding
        if 'patch_embed.proj.weight' in new_key:
            if value.ndim == 5: # [768, 1, 16, 16, 16]
                logging.info(f"Projecting 3D Conv {new_key} {value.shape} to 2D...")
                value = value.sum(dim=2) # -> [768, 1, 16, 16]
                
                # Expand 1 channel -> 3 channels (để khớp với model timm init)
                if value.shape[1] == 1 and target_shape[1] == 3:
                    logging.info(f"Expanding channels for {new_key} from 1 to 3...")
                    value = value.repeat(1, 3, 1, 1) / 3.0

        # Positional Embedding
        if 'pos_embed' in new_key:
            if value.shape != target_shape:
                logging.warning(f"Resizing pos_embed: {value.shape} -> {target_shape}")
                cls_pos = value[:, 0:1, :]
                patch_pos = value[:, 1:, :]
                # BrainIAC pos_embed shape: [1, 216, 768] -> Patch token là 215?
                # Cần kiểm tra kỹ, nhưng logic interpolate sẽ tự xử lý theo target_seq_len
                target_seq_len = target_shape[1] - 1 
                
                patch_pos = patch_pos.permute(0, 2, 1)
                patch_pos = torch.nn.functional.interpolate(patch_pos, size=target_seq_len, mode='linear', align_corners=False)
                patch_pos = patch_pos.permute(0, 2, 1)
                value = torch.cat((cls_pos, patch_pos), dim=1)

        # 4. Gán vào state dict mới
        if value.shape == target_shape:
            new_state_dict[new_key] = value
        else:
            logging.warning(f"Skipping {new_key}: Shape mismatch {value.shape} vs target {target_shape}")
    
    # Load vào model
    missing, unexpected = model_2d.load_state_dict(new_state_dict, strict=False)
    
    # Lọc bỏ lỗi qkv.bias (vì BrainIAC không có, đây là bình thường)
    real_missing = [k for k in missing if 'attn.qkv.bias' not in k and 'head' not in k]
    
    if len(real_missing) > 0:
        logging.warning(f"Missing keys: {len(real_missing)}. Examples: {real_missing[:5]}")
    else:
        logging.info("All weights loaded perfectly (ignoring qkv.bias and head).")
        
    logging.info(f"BrainIAC weights loaded. Transferred {len(new_state_dict)}/{len(model_2d_dict)} layers.")
    return model_2d
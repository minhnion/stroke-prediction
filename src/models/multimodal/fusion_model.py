import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, image_encoder, tabular_encoder, image_embedding_dim, tabular_embedding_dim, fusion_params, mlp_params):
        """
        Mô hình hợp nhất "Plug-and-Play".
        
        Args:
            image_encoder: Đối tượng encoder ảnh đã được khởi tạo.
            tabular_encoder: Đối tượng encoder tabular đã được khởi tạo.
            image_embedding_dim (int): Kích thước vector đầu ra của encoder ảnh.
            tabular_embedding_dim (int): Kích thước vector đầu ra của encoder tabular.
            fusion_params (dict): Cấu hình cho khối hợp nhất.
            mlp_params (dict): Cấu hình cho MLP head cuối cùng.
        """

        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        
        fusion_type = fusion_params.get('type', 'concat')
        
        if 'concat' in fusion_type:
            # Chỉ concat -> không cần chiếu (projection head)
            if fusion_type == 'concat':
                self.projection_image = nn.Identity()
                self.projection_tabular = nn.Identity()
                fusion_input_dim = image_embedding_dim + tabular_embedding_dim

            # concat -> Transformer -> chiếu về cùng dimension
            elif fusion_type == 'concat_transformer':
                projection_dim = fusion_params.get('projection_dim', 512)
                
                # Linear (MLP 1 lớp) để chiếu
                self.projection_image = nn.Linear(image_embedding_dim, projection_dim)
                self.projection_tabular = nn.Linear(tabular_embedding_dim, projection_dim)
                
                # Transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=projection_dim,
                    nhead=fusion_params.get('nhead', 4),
                    dim_feedforward=projection_dim * 4,
                    dropout=fusion_params.get('dropout', 0.1),
                    activation='gelu',
                    batch_first=True
                )

                self.fusion_block = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=fusion_params.get('depth', 2)
                )
                
                # CLS token 
                self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, projection_dim))
                
                # Input  MLP head cuối = output của Transformer
                fusion_input_dim = projection_dim

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # MLP Head
        hidden_dims = mlp_params.get('hidden_dims', [128, 64])
        output_dim = mlp_params.get('output_dim', 1)
        dropout = mlp_params.get('dropout', 0.2)
        
        all_layers = []
        layer_sizes = [fusion_input_dim] + hidden_dims
        
        for i in range(len(layer_sizes) - 1):
            all_layers.extend([
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        all_layers.append(nn.Linear(layer_sizes[-1], output_dim))
        
        self.mlp_head = nn.Sequential(*all_layers)


    def forward(self, batch):
        image = batch['image']
        x_categ = batch['x_categ']
        x_cont = batch['x_cont']
        
        # 1. Mã hóa
        image_features = self.image_encoder.forward_features(image)[:, 0]
        tabular_features = self.tabular_encoder(x_categ, x_cont)
        
        # 2. Chiếu (Project) các features về cùng một không gian
        projected_image = self.projection_image(image_features)
        projected_tabular = self.projection_tabular(tabular_features)

        # 3. Hợp nhất (Fusion)
        # concat_transformer -> 2 vector projected đã có cùng dimension
        if hasattr(self, 'fusion_block') and isinstance(self.fusion_block, nn.TransformerEncoder):
            # Shape: (batch_size, 2, projection_dim)
            fused_sequence = torch.stack([projected_image, projected_tabular], dim=1)
            
            # [CLS] token
            batch_size = fused_sequence.shape[0]
            cls_tokens = self.fusion_cls_token.expand(batch_size, -1, -1)
            fused_sequence = torch.cat((cls_tokens, fused_sequence), dim=1)
            
            fused_output = self.fusion_block(fused_sequence)
            
            # Lấy output của [CLS] token làm đại diện cho toàn bộ
            final_features = fused_output[:, 0]

        else: 
            final_features = torch.cat([projected_image, projected_tabular], dim=1)
            
        # 4. Đưa ra dự đoán
        output = self.mlp_head(final_features)
        
        return output
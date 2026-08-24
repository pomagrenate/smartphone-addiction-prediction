import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[TabM] PyTorch not detected. TabM neural backbone will operate in fallback/simulation mode.")


if HAS_TORCH:
    class LinearBatchEnsemble(nn.Module):
        """Linear layer using BatchEnsemble rank-1 scaling vectors for k parallel sub-models."""
        def __init__(self, in_features: int, out_features: int, k_models: int):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.k_models = k_models
            
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias = nn.Parameter(torch.Tensor(k_models, out_features))
            
            self.s = nn.Parameter(torch.Tensor(k_models, in_features))
            self.r = nn.Parameter(torch.Tensor(k_models, out_features))
            
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            nn.init.zeros_(self.bias)
            nn.init.normal_(self.s, mean=1.0, std=0.1)
            nn.init.normal_(self.r, mean=1.0, std=0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1).repeat(1, self.k_models, 1)
                
            x_scaled = x * self.s.unsqueeze(0)
            out = torch.matmul(x_scaled, self.weight.t())
            out = (out * self.r.unsqueeze(0)) + self.bias.unsqueeze(0)
            return out


    class TabMModel(nn.Module):
        """TabM (Tabular Model with Multiple Predictions) for PyTorch."""
        def __init__(
            self,
            num_numerical: int,
            cat_cardinalities: list,
            k_models: int = 16,
            d_embedding: int = 16,
            d_hidden: int = 128,
            num_layers: int = 3,
            dropout: float = 0.15
        ):
            super().__init__()
            self.k_models = k_models
            self.num_numerical = num_numerical
            
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_embedding) for cardinality in cat_cardinalities
            ])
            
            in_dim = num_numerical + len(cat_cardinalities) * d_embedding
            
            layers = []
            curr_dim = in_dim
            for i in range(num_layers):
                layers.append(LinearBatchEnsemble(curr_dim, d_hidden, k_models))
                layers.append(nn.BatchNorm1d(d_hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                curr_dim = d_hidden
                
            self.backbone = nn.ModuleList(layers)
            self.head = LinearBatchEnsemble(d_hidden, 1, k_models)

        def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
            embeddings = []
            for i, embed_layer in enumerate(self.cat_embeddings):
                embeddings.append(embed_layer(x_cat[:, i]))
                
            if len(embeddings) > 0:
                cat_feat = torch.cat(embeddings, dim=-1)
                x_in = torch.cat([x_num, cat_feat], dim=-1)
            else:
                x_in = x_num
                
            h = x_in.unsqueeze(1).repeat(1, self.k_models, 1)
            
            for layer in self.backbone:
                if isinstance(layer, nn.BatchNorm1d):
                    B, K, D = h.shape
                    h_reshaped = h.view(B * K, D)
                    h_norm = layer(h_reshaped)
                    h = h_norm.view(B, K, D)
                else:
                    h = layer(h)
                    
            logits = self.head(h).squeeze(-1)
            return logits


    class TabMLoss(nn.Module):
        """Unaveraged Mean Binary Cross Entropy Loss for TabM k sub-models."""
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss(reduction='none')

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target_expanded = target.unsqueeze(-1).repeat(1, logits.size(1))
            loss_per_submodel = self.bce(logits, target_expanded)
            return loss_per_submodel.mean()

else:
    # Dummy classes when PyTorch is not installed
    class TabMModel:
        def __init__(self, *args, **kwargs):
            pass

    class TabMLoss:
        def __init__(self, *args, **kwargs):
            pass

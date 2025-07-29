import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip import create_model_and_transforms
from .moe import MoE

class CLIPWithMoE(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-16",
        pretrained="laion2B-s34B-b88K",
        feature_dim=512, # В зависимости от CLIP
        num_experts=4,
        k=2,
    ):
        super().__init__()
        
        self.clip_model, _, self.preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
        self.feature_dim = feature_dim

        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Если вдруг захотим использовать патчи вместо [CLS] для реализации MoE трансформера
        # with torch.no_grad():
        #     outputs = clip_model(pixel_values)
        #     sequence_output = outputs.last_hidden_state  # [B, N+1, D]
        # Удаляем [CLS] токен
        # patch_tokens = sequence_output[:, 1:, :]  # [B, N, D]

        # Mixture of Experts
        self.moe = MoE(
            input_size=self.feature_dim,
            output_size=self.feature_dim,
            num_experts=num_experts,
            hidden_size=self.feature_dim * 2,
            noisy_gating=True,
            k=k
        )

        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, input_tensor):
        """
        Args:
            input_tensor (Tensor): [B, C, H, W]

        Returns:
            logits (Tensor): [B, 1]
            aux_loss (Tensor): aux_loss
        """
        with torch.no_grad():
            features = self.clip_model(input_tensor)[0] # [B, D], где D = 512

        moe_out, aux_loss = self.moe(features)
        logits = self.classifier(moe_out)
        
        return logits, aux_loss

    def compute_loss(self, logits, targets, aux_loss, loss_coef=5e-4):
        """
        Args:
            logits: Tensor of shape [B, 1] or [B]
            targets: Tensor of shape [B] with values 0 or 1
            aux_loss: MoE auxiliary loss
            loss_coef: weight for MoE auxiliary loss
        Returns:
            total_loss, bce_loss_value, aux_loss_value
        """
        if targets.dim() == 1:
            targets = targets.float().unsqueeze(-1)  # [B] -> [B, 1]

        # weight = torch.full_like(targets, 0.7, dtype=torch.float)
        # weight[targets == 1] = 0.3
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets) # , weight=weight)
        total_loss = bce_loss + loss_coef * aux_loss
    
        return total_loss , bce_loss.item(), aux_loss.item()
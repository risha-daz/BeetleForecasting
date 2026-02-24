import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head: backbone_dim → hidden → embedding_dim."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class BeetleContrastiveModel(nn.Module):
    """
    Shared DINOv2-small backbone + per-stream projection heads.

    Forward pass returns:
        z_beetle, z_color, z_scale  — L2-normalised embeddings (B, embed_dim)
        z_fused                     — concatenation of the three (B, 3*embed_dim)

    DINOv2 (facebook/dinov2-small) chosen because:
    - Pretrained with self-supervised patch-level objectives → rich local &
      global features, ideal for specimen images.
    - ViT patch tokens naturally handle any input resolution ≥ patch_size (14).
    - Small variant (~22 M params) is fast to fine-tune.
    """

    DINOV2_VARIANTS = {
        "small":  ("facebookresearch/dinov2", "dinov2_vits14", 384),
        "base":   ("facebookresearch/dinov2", "dinov2_vitb14", 768),
        "large":  ("facebookresearch/dinov2", "dinov2_vitl14", 1024),
    }

    def __init__(
        self,
        variant: str = "small",
        embed_dim: int = 128,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        repo, model_name, backbone_dim = self.DINOV2_VARIANTS[variant]

        # Load pretrained DINOv2 from torch.hub
        self.backbone = torch.hub.load(repo, model_name, pretrained=True)
        self.backbone_dim = backbone_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # One projection head per image stream (they do NOT share weights so
        # the model can specialise per modality)
        self.proj_beetle = ProjectionHead(backbone_dim, 512, embed_dim)
        self.proj_color  = ProjectionHead(backbone_dim, 512, embed_dim)
        self.proj_scale  = ProjectionHead(backbone_dim, 512, embed_dim)

        self.embed_dim = embed_dim

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Run backbone; return CLS token (B, backbone_dim)."""
        return self.backbone(imgs)  # DINOv2 returns CLS by default

    def forward(self, beetle_imgs, color_imgs, scale_imgs):
        f_beetle = self.encode(beetle_imgs)
        f_color  = self.encode(color_imgs)
        f_scale  = self.encode(scale_imgs)

        z_beetle = self.proj_beetle(f_beetle)
        z_color  = self.proj_color(f_color)
        z_scale  = self.proj_scale(f_scale)

        z_fused = torch.cat([z_beetle, z_color, z_scale], dim=-1)  # (B, 3*embed_dim)

        return z_beetle, z_color, z_scale, z_fused

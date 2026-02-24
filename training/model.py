"""
Beetle SPEI Predictor
======================
Visual backbone (frozen DINOv2) + fixed categorical embeddings → SPEI regression.

Inputs
------
  beetle_img, color_img, scale_img : images       → DINOv2 → z_fused  (B, 384)
  scientificName                   : str (144)    → Embedding          (B,  32)
  domainID                         : int (10)     → Embedding          (B,  16)

domainID is treated as a non-ordinal category. The 10 known values
(1, 3, 4, 7, 9, 11, 32, 46, 99, 202) are mapped to indices 0–9.
Unknown values fall back to index 0.

Outputs
-------
  mu    : (B, 3)  predicted means   [SPEI_30d, SPEI_1y, SPEI_2y]
  sigma : (B, 3)  predicted std devs (uncertainty, always > 0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contrastive_trainer import BeetleContrastiveModel, get_sentinel_beetles_loader_with_collate


# ---------------------------------------------------------------------------
# Fixed vocabularies
# ---------------------------------------------------------------------------

# Fill this list with the real 144 species names from your dataset.
# Order doesn't matter — each gets its own embedding row.
from utils.col_vals import *

# ---------------------------------------------------------------------------
# Fixed-vocab categorical encoder
# ---------------------------------------------------------------------------

class CategoricalEncoder(nn.Module):
    """
    Maps a fixed set of known values (str or int) to learned embeddings.
    Unknown values at inference time fall back to a dedicated <UNK> row.

    vocab_values : list of all known values (order defines the index)
    embed_dim    : output embedding size
    """

    def __init__(self, vocab_values: list, embed_dim: int):
        super().__init__()
        # index 0 = <UNK>, known values start at 1
        self.vocab: dict = {v: i + 1 for i, v in enumerate(vocab_values)}
        n_embeddings = len(vocab_values) + 1   # +1 for UNK
        self.embedding = nn.Embedding(n_embeddings, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, values: list) -> torch.Tensor:
        indices = torch.tensor(
            [self.vocab.get(v, 0) for v in values],   # 0 = UNK
            dtype=torch.long,
            device=self.embedding.weight.device,
        )
        return self.embedding(indices)   # (B, embed_dim)


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

class SPEIHead(nn.Module):
    """MLP: concat(visual, species, domain) → (mu, sigma) for 3 SPEI targets."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, n_targets: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(
                in_features = hidden_dim //2,
                out_features=3,
                )
            )

    def forward(self, x: torch.Tensor):
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class BeetleSPEIPredictor(nn.Module):
    """
    Args:
        checkpoint_path   : BeetleContrastiveModel checkpoint path, or None
        variant           : DINOv2 size — 'small' | 'base' | 'large'
        embed_dim         : visual projection dim (must match checkpoint)
        species_embed_dim : embedding size for scientificName
        domain_embed_dim  : embedding size for domainID
        freeze_backbone   : freeze DINOv2 weights (recommended)
        scientific_names  : list of 144 known species strings
        domain_ids        : list of 10 known domain int values
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        variant: str = "small",
        embed_dim: int = 128,
        species_embed_dim: int = 32,
        domain_embed_dim: int = 16,
        freeze_backbone: bool = True,
        scientific_names: list[str] = SCIENTIFIC_NAMES,
        domain_ids: list[int] = DOMAIN_IDS,
    ):
        super().__init__()

        # --- Visual backbone -------------------------------------------
        self.backbone = BeetleContrastiveModel(
            variant=variant,
            embed_dim=embed_dim,
            freeze_backbone=freeze_backbone,
        )
        if checkpoint_path is not None:
            ckpt  = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("model_state", ckpt)
            self.backbone.load_state_dict(state)
            print(f"Loaded backbone from {checkpoint_path}")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

        # --- Categorical encoders (fixed vocab) -----------------------
        self.species_encoder = CategoricalEncoder(scientific_names, species_embed_dim)
        self.domain_encoder  = CategoricalEncoder(domain_ids, domain_embed_dim)

        # --- Regression head ------------------------------------------
        visual_dim = embed_dim * 3
        total_dim  = visual_dim + species_embed_dim + domain_embed_dim
        self.head  = SPEIHead(in_dim=total_dim, hidden_dim=256)

    def forward(
        self,
        beetle_imgs: torch.Tensor,
        color_imgs:  torch.Tensor,
        scale_imgs:  torch.Tensor,
        sci_names:   list[str],
        domain_ids:  list[int],
    ):
        # 1. Visual features (no grad if backbone frozen)
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                _, _, _, z_fused = self.backbone(beetle_imgs, color_imgs, scale_imgs)
        else:
            _, _, _, z_fused = self.backbone(beetle_imgs, color_imgs, scale_imgs)

        # 2. Categorical embeddings
        e_species = self.species_encoder(sci_names).to(z_fused.device)   # (B, 32)
        e_domain  = self.domain_encoder(domain_ids).to(z_fused.device)   # (B, 16)

        # 3. Concatenate and regress
        combined  = torch.cat([z_fused, e_species, e_domain], dim=-1)
        return self.head(combined)
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
from dino import BeetleContrastiveModel, get_sentinel_beetles_loader_with_collate


# ---------------------------------------------------------------------------
# Fixed vocabularies
# ---------------------------------------------------------------------------

# Fill this list with the real 144 species names from your dataset.
# Order doesn't matter — each gets its own embedding row.
from scientificNames import SCIENTIFIC_NAMES
DOMAIN_IDS: list[int] = [1, 3, 4, 7, 9, 11, 32, 46, 99, 202]


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

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TARGET_NAMES = ["SPEI_1y", "SPEI_2y", "SPEI_30d"]


def run_epoch(model, loader, device, optimizer=None, max_batches: int = 10):
    """
    Runs up to `max_batches` batches from `loader`.
    If optimizer is provided → training mode (prints every 2 steps).
    If optimizer is None     → validation mode (no grad, no update, silent).

    Returns: avg_mse, per_target_mse
    """
    is_train = optimizer is not None
    model.head.train(is_train)
    model.species_encoder.train(is_train)
    model.domain_encoder.train(is_train)
    if not any(p.requires_grad for p in model.backbone.parameters()):
        model.backbone.eval()   # always keep frozen backbone in eval

    total_mse, steps = 0.0, 0
    total_per_target = torch.zeros(3)

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids, targets in loader:
            if steps >= max_batches:
                break

            beetle_imgs = beetle_imgs.to(device)
            color_imgs  = color_imgs.to(device)
            scale_imgs  = scale_imgs.to(device)
            targets     = targets.to(device)

            outputs = model(beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids)

            loss = F.mse_loss(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()

            total_mse        += loss.item()
            total_per_target += ((outputs.detach().cpu() - targets.cpu()) ** 2).mean(dim=0)

            if is_train and steps % 2 == 0:
                per_step_mse = ((outputs.detach().cpu() - targets.cpu()) ** 2).mean(dim=0)
                mse_str = "  ".join(f"{n}: {v:.4f}" for n, v in zip(TARGET_NAMES, per_step_mse.tolist()))
                print(
                    f"  step {steps:2d} | MSE {loss.item():.4f}\n"
                    f"         per-target MSE: {mse_str}"
                )

            steps += 1

    avg_mse        = total_mse / steps
    per_target_mse = (total_per_target / steps).tolist()

    return avg_mse, per_target_mse


def train(
    checkpoint_path: str | None = None,
    n_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    variant: str = "small",
    embed_dim: int = 128,
    species_embed_dim: int = 32,
    domain_embed_dim: int = 16,
    freeze_backbone: bool = True,
    scientific_names: list[str] = SCIENTIFIC_NAMES,
    domain_ids: list[int] = DOMAIN_IDS,
    device_str: str | None = None,
    max_batches: int = 10,
):
    device = torch.device(
        device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    model = BeetleSPEIPredictor(
        checkpoint_path=checkpoint_path,
        variant=variant,
        embed_dim=embed_dim,
        species_embed_dim=species_embed_dim,
        domain_embed_dim=domain_embed_dim,
        freeze_backbone=freeze_backbone,
        scientific_names=scientific_names,
        domain_ids=domain_ids,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    train_loader = get_sentinel_beetles_loader_with_collate(
        split="train", batch_size=batch_size, num_workers=0
    )
    val_loader = get_sentinel_beetles_loader_with_collate(
        split="validation", batch_size=batch_size, num_workers=0
    )

    best_val_nll = float("inf")

    for epoch in range(1, n_epochs + 1):

        # --- Training -------------------------------------------------
        train_mse, train_per = run_epoch(model, train_loader, device, optimizer, max_batches=max_batches)

        # --- Validation -----------------------------------------------
        val_mse, val_per = run_epoch(model, val_loader, device, optimizer=None, max_batches=max_batches)

        scheduler.step(val_mse)

        # --- Logging --------------------------------------------------
        train_per_str = "  ".join(f"{n}: {v:.4f}" for n, v in zip(TARGET_NAMES, train_per))
        val_per_str   = "  ".join(f"{n}: {v:.4f}" for n, v in zip(TARGET_NAMES, val_per))

        print(
            f"\nEpoch {epoch:3d}"
            f"\n  train | MSE {train_mse:.4f}  per-target: {train_per_str}"
            f"\n  val   | MSE {val_mse:.4f}  per-target: {val_per_str}"
        )

        # --- Checkpoint -----------------------------------------------
        is_best = val_mse < best_val_nll
        if is_best:
            best_val_nll = val_mse
            print(f"  ✓ New best val NLL: {best_val_nll:.4f} — saving best checkpoint")

        torch.save(
            {
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_mse":       train_mse,
                "val_mse":         val_mse,
            },
            f"beetle_spei_epoch{epoch:02d}.pt",
        )
        if is_best:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                "beetle_spei_best.pt",
            )

    print(f"\nTraining complete. Best val NLL: {best_val_nll:.4f}")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids, device):
    """
    Returns {"mu": (B,3), "sigma": (B,3)}.
    Columns: [SPEI_30d, SPEI_1y, SPEI_2y].
    """
    model.eval()
    mu, sigma = model(
        beetle_imgs.to(device),
        color_imgs.to(device),
        scale_imgs.to(device),
        sci_names,
        domain_ids,
    )
    return {"mu": mu.cpu(), "sigma": sigma.cpu()}


def load_from_checkpoint(path: str, device_str: str | None = None, **model_kwargs):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt   = torch.load(path, map_location="cpu")
    model  = BeetleSPEIPredictor(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded from {path} (epoch {ckpt.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Replace SCIENTIFIC_NAMES with your actual 144 species list before running
    trained_model = train(
        checkpoint_path="beetle_contrastive_epoch9.pt",
        n_epochs=5,
        batch_size=16,
        lr=1e-3,
        freeze_backbone=True,
        scientific_names=SCIENTIFIC_NAMES,   # ← plug in the real list
        domain_ids=DOMAIN_IDS,
        max_batches=10,
    )
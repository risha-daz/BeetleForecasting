"""
Beetle Contrastive Learning
============================
Uses DINOv2 (ViT-S/14) as the vision backbone — it handles variable-resolution
images well and produces strong visual features out of the box.

Two main components
-------------------
1. VariableSizeTransform  – collate-time logic that pads images in a batch to
   the same spatial size so they can be stacked into a single tensor.
2. BeetleContrastiveModel – encodes the three views (beetle, color-chip, scale-
   bar) with a shared DINOv2 backbone + lightweight projection head, then
   combines them into a single per-observation embedding for NT-Xent loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


# ---------------------------------------------------------------------------
# 1.  Variable-size transform + custom collate
# ---------------------------------------------------------------------------

class VariableSizeTransform:
    """
    Resize the shortest side to `min_size` (preserving aspect ratio) then
    pad to a fixed canvas of `canvas_size × canvas_size`.

    This keeps every image's aspect ratio intact and avoids squishing, while
    still producing tensors of identical shape that can be stacked.
    """

    def __init__(self, min_size: int = 224, canvas_size: int = 224):
        self.min_size = min_size
        self.canvas_size = canvas_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize shortest side → min_size
        w, h = img.size
        scale = self.min_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Clamp so neither side exceeds canvas
        if new_w > self.canvas_size or new_h > self.canvas_size:
            scale = self.canvas_size / max(new_w, new_h)
            new_w, new_h = int(new_w * scale), int(new_h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        tensor = self.to_tensor(img)  # (3, H, W)

        # Pad to canvas_size × canvas_size (bottom-right padding)
        pad_h = self.canvas_size - tensor.shape[1]
        pad_w = self.canvas_size - tensor.shape[2]
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)

        return self.normalize(tensor)


def beetle_collate_fn(batch):
    """
    Custom collate that handles:
    - Variable-size PIL images for beetle / colorpicker / scalebar
    - Stacks them into (B, 3, H, W) tensors
    - Passes through scalars and strings unchanged

    `batch` is a list of items yielded by the dataset __iter__:
        ([image, colorpicker_img, scalebar_img, scientificName, domainID], targets)
    """
    transform = VariableSizeTransform(min_size=224, canvas_size=224)

    beetle_imgs, color_imgs, scale_imgs = [], [], []
    sci_names, domain_ids, targets = [], [], []

    for (image, colorpicker_img, scalebar_img, sci_name, domain_id), target in batch:
        # image / colorpicker_img may already be tensors (from the dataloader's
        # transform2) or PIL Images — handle both.
        def to_tensor(x):
            if isinstance(x, Image.Image):
                return transform(x)
            # Already a tensor of shape (3, H, W) — normalise + pad
            if isinstance(x, torch.Tensor):
                # Re-apply padding / normalisation consistently
                pil = TF.to_pil_image(x)
                return transform(pil)
            return x

        beetle_imgs.append(to_tensor(image))
        color_imgs.append(to_tensor(colorpicker_img))
        scale_imgs.append(to_tensor(scalebar_img))
        sci_names.append(sci_name)
        domain_ids.append(domain_id)
        targets.append(target)

    return (
        torch.stack(beetle_imgs),   # (B, 3, 224, 224)
        torch.stack(color_imgs),    # (B, 3, 224, 224)
        torch.stack(scale_imgs),    # (B, 3, 224, 224)
        sci_names,                  # list[str]
        domain_ids,                 # list[str]
        torch.stack(targets),       # (B, 3)
    )


# ---------------------------------------------------------------------------
# 2.  Contrastive model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3.  NT-Xent (InfoNCE) contrastive loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    NT-Xent loss between two sets of L2-normalised embeddings.
    Positive pairs: (z_a[i], z_b[i])  for all i in batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive indices: for row i, positive is at i+B (and vice versa)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device),
        ])

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# 4.  Example training loop
# ---------------------------------------------------------------------------

def get_sentinel_beetles_loader_with_collate(split="train", batch_size=32, num_workers=0):
    """
    Drop-in replacement for the original loader — adds the custom collate_fn.
    Import and use the original `get_sentinel_beetles_loader` logic but swap
    in `beetle_collate_fn`.
    """
    from datasets import load_dataset
    from torch.utils.data import IterableDataset, DataLoader
    from torchvision import transforms

    hf_dataset = load_dataset(
        "imageomics/sentinel-beetles",
        split=split,
        streaming=True,
    )

    class _StreamingDataset(IterableDataset):
        def __iter__(self):
            for example in hf_dataset:
                image          = example["file_path"]
                colorpicker_img = example["colorpicker_full_path"]
                scalebar_img   = example["scalebar_full_path"]
                scientificName = example["scientificName"]
                domainID       = example["domainID"]
                SPEI_1y  = torch.tensor(example["SPEI_1y"],  dtype=torch.float32)
                SPEI_2y  = torch.tensor(example["SPEI_2y"],  dtype=torch.float32)
                SPEI_30d = torch.tensor(example["SPEI_30d"], dtype=torch.float32)
                yield (
                    [image, colorpicker_img, scalebar_img, scientificName, domainID],
                    torch.stack([SPEI_1y, SPEI_2y, SPEI_30d]),
                )

    dataset = _StreamingDataset()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=beetle_collate_fn,
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for step, (beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids, targets) in enumerate(loader):
        beetle_imgs = beetle_imgs.to(device)
        color_imgs  = color_imgs.to(device)
        scale_imgs  = scale_imgs.to(device)

        z_beetle, z_color, z_scale, _ = model(beetle_imgs, color_imgs, scale_imgs)

        # Three pairwise contrastive losses
        loss = (
            criterion(z_beetle, z_color)
            + criterion(z_beetle, z_scale)
            + criterion(z_color,  z_scale)
        ) / 3.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 2 == 0:
            print(f"  step {step:4d} | loss {loss.item():.4f}")
        if step == 20 :
            print(f"  step {step:4d} | loss {loss.item():.4f}")
            break

    return total_loss / max(step + 1, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BeetleContrastiveModel(variant="small", embed_dim=128, freeze_backbone=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    criterion = NTXentLoss(temperature=0.07)

    loader = get_sentinel_beetles_loader_with_collate(
        split="train",
        batch_size=16,   # keep small — DINOv2 is memory-hungry
        num_workers=0,
    )

    for epoch in range(10):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} | avg loss {avg_loss:.4f}")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            f"beetle_contrastive_epoch{epoch + 1}.pt",
        )
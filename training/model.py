"""
Sample predictive model.
The ingestion program will call `predict` to get a prediction for each test image and then save the predictions for scoring. The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
"""

import torch
import torch._dynamo as dynamo
import os
import torch.nn as nn
from col_vals import *
from contrastive_model import BeetleContrastiveModel
from encoding import CategoricalEncoder, SPEIHead
from dataloader import get_sentinel_beetles_loader_with_collate

class BeetleSPEIPredictor(nn.Module):
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

        self.species_encoder = CategoricalEncoder(scientific_names, species_embed_dim)
        self.domain_encoder  = CategoricalEncoder(domain_ids, domain_embed_dim)

        visual_dim = embed_dim * 3
        total_dim  = visual_dim + species_embed_dim + domain_embed_dim
        self.head  = SPEIHead(in_dim=total_dim, hidden_dim=256)

    def forward(self,
                beetle_imgs: torch.Tensor,
                color_imgs:  torch.Tensor,
                scale_imgs:  torch.Tensor,
                sci_names:   list[str],
                domain_ids:  list[int]): 
        
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                _, _, _, z_fused = self.backbone(beetle_imgs, color_imgs, scale_imgs)
        else:
            _, _, _, z_fused = self.backbone(beetle_imgs, color_imgs, scale_imgs)

        # 2. Categorical embeddings
        # dynamo.disable prevents recompilation for each unique species/domain string,
        # which was causing torch.compile to hit config.recompile_limit (8).
        e_species = dynamo.disable(self.species_encoder)(sci_names).to(z_fused.device)
        e_domain  = dynamo.disable(self.domain_encoder)(domain_ids).to(z_fused.device)

        # 3. Concatenate and regress
        combined  = torch.cat([z_fused, e_species, e_domain], dim=-1)
        return self.head(combined)

class Model:
    def __init__(self):
        # model will be called from the load() method
        self.model = None
        self.transforms = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        # Enable TF32 for better matmul performance on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')

        contrastive_model_path = os.path.join(os.path.dirname(__file__), "contrastive.pt")
        model_path = os.path.join(os.path.dirname(__file__), "model.pt")
        self.model = BeetleSPEIPredictor(
            checkpoint_path=contrastive_model_path
        )
        ckpt = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])

        # Move model to GPU if available
        self.model = self.model.to(self.device)

        # Set eval once here — no need to call again in predict()
        self.model.eval()

        # Compile model for faster inference (PyTorch 2.0+)
        self.model = torch.compile(self.model, dynamic=True)

    def predict(self, datapoints, batch_size=16):
        outputs = []

        for i in range(0, len(datapoints), batch_size):
            batch = datapoints[i:i+batch_size]

            beetle_imgs, color_imgs, scale_imgs, sci_names, domain_ids = \
                get_sentinel_beetles_loader_with_collate(batch)

            # Move tensors to device (GPU if available)
            beetle_imgs = beetle_imgs.to(self.device)
            color_imgs  = color_imgs.to(self.device)
            scale_imgs  = scale_imgs.to(self.device)

            with torch.inference_mode():
                out = self.model(
                    beetle_imgs,
                    color_imgs,
                    scale_imgs,
                    sci_names,
                    domain_ids
                )

            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)

        mu = outputs.mean(dim=0)
        # Guard against std() with a single sample (undefined with N=1)
        sigma = outputs.std(dim=0) if outputs.shape[0] > 1 else torch.zeros_like(mu)

        return {
            "SPEI_1y":  {"mu": mu[0].item(), "sigma": sigma[0].item()},
            "SPEI_2y":  {"mu": mu[1].item(), "sigma": sigma[1].item()},
            "SPEI_30d": {"mu": mu[2].item(), "sigma": sigma[2].item()},
        }
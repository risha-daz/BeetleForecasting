import torch
import torch.nn as nn

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
        return self.embedding(indices)
    
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


import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DINO_MODELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

class DINOv2Extractor(nn.Module):

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert backbone in DINO_MODELS, f"Unknown backbone: {backbone}"
        self.backbone_name = backbone
        self.embed_dim = DINO_MODELS[backbone]

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading {backbone} from torch.hub...")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone,
            pretrained=True,
        )
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()
        self.backbone.to(self.device)
        print(f"  Backbone loaded. Embedding dim: {self.embed_dim}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(self.device)
        features = self.backbone(x)
        return features 

    @torch.no_grad()
    def extract_batch(self, loader) -> np.ndarray:

        all_embeddings = []
        for batch in tqdm(loader, desc="Extracting embeddings"):
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch

            imgs = self._normalize_if_needed(imgs)
            embeddings = self.forward(imgs)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    @staticmethod
    def _normalize_if_needed(imgs: torch.Tensor) -> torch.Tensor:
        if imgs.max() <= 1.0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            imgs = (imgs - mean) / std
        return imgs


if __name__ == "__main__":
    extractor = DINOv2Extractor(backbone="dinov2_vitb14")
    dummy = torch.rand(2, 3, 256, 256)
    emb = extractor(dummy)
    print(f"Input shape     : {dummy.shape}")
    print(f"Embedding shape : {emb.shape}")
    print("DINOv2Extractor OK ✓")

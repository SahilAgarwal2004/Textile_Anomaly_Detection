"""
inference_reconstruction.py
────────────────────────────
Runs Pipeline A (autoencoder) on test images and returns anomaly scores.
"""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_A.unet_autoencoder import UNetAutoencoder
from preprocessing.dataset_loader import get_test_loader
from preprocessing.visualization import save_residual_heatmap


class ReconstructionInference:

    def __init__(self, cfg: dict, category: str):
        self.cfg = cfg
        self.category = category
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            and cfg["device"].get("use_cuda", True)
            else "cpu"
        )
        pipe_cfg = cfg["pipeline_a"]
        self.threshold = pipe_cfg["residual_threshold"]

        # Load model
        ckpt_dir = Path(pipe_cfg["save_dir"]) / category
        ckpt_path = ckpt_dir / "best_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. Train the model first."
            )

        self.model = UNetAutoencoder(latent_dim=pipe_cfg["latent_dim"])
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def run(
        self,
        save_heatmaps: bool = False,
        output_dir: str = "evaluation/results/pipeline_a/",
    ) -> Dict[str, Any]:
        """
        Returns a dict with:
          scores  : list of anomaly scores
          labels  : list of ground-truth labels
          names   : list of filenames
        """
        ds_cfg = cfg = self.cfg["dataset"]
        loader = get_test_loader(
            root=ds_cfg["root"],
            category=self.category,
            batch_size=1,
            num_workers=ds_cfg["num_workers"],
            image_size=ds_cfg["image_size"],
            for_reconstruction=True,
        )

        scores, labels, names, areas = [], [], [], []

        for imgs, lbls, fnames in loader:
            imgs = imgs.to(self.device)

            recon, residual, score, area = self.model.compute_residual(imgs)

            scores.append(score)
            labels.append(int(lbls.item()))
            names.append(fnames[0])
            areas.append(area)

            if save_heatmaps:
                save_residual_heatmap(
                    original=imgs[0].cpu(),
                    reconstructed=recon[0].cpu(),
                    residual=residual[0].cpu(),
                    save_path=output_dir,
                    filename=fnames[0],
                    score=score,
                    label=int(lbls.item()),
                )

        return {
            "scores":  scores,
            "labels":  labels,
            "names":   names,
            "areas":   areas,
            "pipeline": "reconstruction",
        }

    def infer_single(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Infer on a single image tensor [3, H, W].
        Returns structured dict for the fusion layer.
        """
        with torch.no_grad():
            x = tensor.unsqueeze(0).to(self.device)
            recon, residual, score, area = self.model.compute_residual(x)

        pixel_count = int((residual[0].mean(dim=0) > self.threshold).sum().item())

        return {
            "reconstruction_score": score,
            "residual_area_percent": area * 100,
            "anomaly_patch_count": pixel_count,
            "residual_map": residual[0].cpu(),
            "reconstructed": recon[0].cpu(),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--heatmaps", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    inf = ReconstructionInference(cfg, args.category)
    results = inf.run(save_heatmaps=args.heatmaps)

    print(f"\nPipeline A results for [{args.category}]")
    print(f"  Total images : {len(results['scores'])}")
    print(f"  Mean score   : {np.mean(results['scores']):.4f}")
    print(f"  Normal images: {results['labels'].count(0)}")
    print(f"  Defect images: {results['labels'].count(1)}")

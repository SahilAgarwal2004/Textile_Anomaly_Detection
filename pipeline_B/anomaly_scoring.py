

import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_B.embedding_extractor import DINOv2Extractor
from pipeline_B.feature_distribution import NormalDistributionModel
from preprocessing.dataset_loader import get_train_loader, get_test_loader


class ZeroShotAnomalyScorer:

    def __init__(self, cfg: dict, category: str):
        self.cfg      = cfg
        self.category = category

        pipe_cfg = cfg["pipeline_b"]
        dev_cfg  = cfg["device"]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            and dev_cfg.get("use_cuda", True)
            else "cpu"
        )

        self.save_dir = Path(pipe_cfg["save_dir"]) / category
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = DINOv2Extractor(
            backbone=pipe_cfg["backbone"],
            device=self.device,
        )
        self.dist_model = NormalDistributionModel(
            regularization=pipe_cfg["regularization"]
        )

    def fit(self):

        ds_cfg = self.cfg["dataset"]
        pipe_cfg = self.cfg["pipeline_b"]

        loader = get_train_loader(
            root=ds_cfg["root"],
            category=self.category,
            batch_size=pipe_cfg["batch_size"],
            num_workers=ds_cfg["num_workers"],
            image_size=ds_cfg["image_size"],
            for_reconstruction=False,  
        )

        print(f"\n[Pipeline B] Fitting distribution for [{self.category}]")
        print(f"  Training images: {len(loader.dataset)}")

        embeddings = self.extractor.extract_batch(loader)
        print(f"  Embeddings shape: {embeddings.shape}")

        self.dist_model.fit(embeddings)
        self.dist_model.save(str(self.save_dir / "distribution_model.pkl"))

    def load(self):
        """Load a previously fitted distribution model from disk."""
        model_path = self.save_dir / "distribution_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No distribution model at {model_path}. Run fit() first."
            )
        self.dist_model = NormalDistributionModel.load(str(model_path))

    def run(self) -> Dict[str, Any]:
        """
        Step 3: Score all test images. Returns dict with scores and labels.
        """
        ds_cfg   = self.cfg["dataset"]
        pipe_cfg = self.cfg["pipeline_b"]

        loader = get_test_loader(
            root=ds_cfg["root"],
            category=self.category,
            batch_size=pipe_cfg["batch_size"],
            num_workers=ds_cfg["num_workers"],
            image_size=ds_cfg["image_size"],
            for_reconstruction=False,
        )

        all_scores, all_labels, all_names = [], [], []

        print(f"\n[Pipeline B] Scoring test images for [{self.category}]")
        for imgs, labels, names in loader:
            imgs = DINOv2Extractor._normalize_if_needed(imgs)
            with torch.no_grad():
                embeddings = self.extractor(imgs).cpu().numpy()

            scores = self.dist_model.score_batch(embeddings)
            all_scores.extend(scores.tolist())
            all_labels.extend(labels.tolist())
            all_names.extend(list(names))

        return {
            "scores":   all_scores,
            "labels":   all_labels,
            "names":    all_names,
            "pipeline": "zero_shot",
        }

    def infer_single(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Score a single image tensor [3, H, W].
        Returns structured dict for fusion layer.
        """
        tensor = DINOv2Extractor._normalize_if_needed(tensor.unsqueeze(0))
        with torch.no_grad():
            embedding = self.extractor(tensor).cpu().numpy()

        score = self.dist_model.score(embedding[0])
        return {
            "semantic_score": score,
            "embedding": embedding[0],
        }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--category", required=True)
    parser.add_argument("--fit",      action="store_true", help="Fit distribution")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scorer = ZeroShotAnomalyScorer(cfg, args.category)

    if args.fit:
        scorer.fit()
    else:
        scorer.load()

    results = scorer.run()
    print(f"\nPipeline B results for [{args.category}]")
    print(f"  Total images : {len(results['scores'])}")
    print(f"  Mean score   : {np.mean(results['scores']):.4f}")
    print(f"  Normal count : {results['labels'].count(0)}")
    print(f"  Defect count : {results['labels'].count(1)}")

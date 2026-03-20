"""
train_reconstruction.py
────────────────────────
Trains the U-Net autoencoder on normal (good) images only.

RTX 5050 compatibility:
  - Uses float32 by default.
  - Mixed precision disabled unless config explicitly enables it.
  - Falls back to CPU automatically if CUDA ops fail.
"""

import os
import sys
import yaml
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline_A.unet_autoencoder import UNetAutoencoder
from preprocessing.dataset_loader import get_train_loader

console = Console()


# ─────────────────────────────────────────────────────────────
#  Device setup (RTX 5050 safe)
# ─────────────────────────────────────────────────────────────

def get_device(cfg: dict) -> torch.device:
    dev_cfg = cfg.get("device", {})
    if dev_cfg.get("force_cpu", False):
        console.print("[yellow]Force CPU mode enabled.[/yellow]")
        return torch.device("cpu")

    if dev_cfg.get("use_cuda", True) and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[green]GPU detected: {gpu_name}[/green]")
        return device

    console.print("[yellow]CUDA not available — using CPU.[/yellow]")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
#  Loss function
# ─────────────────────────────────────────────────────────────

class ReconstructionLoss(nn.Module):
    """
    Combined MSE + SSIM-inspired loss for better perceptual quality.
    Using L1 + MSE gives a good balance between sharpness and stability.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.l1(pred, target)


# ─────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────

class AutoencoderTrainer:

    def __init__(self, cfg: dict, category: str):
        self.cfg = cfg
        self.category = category
        self.device = get_device(cfg)

        pipe_cfg = cfg["pipeline_a"]
        self.epochs     = pipe_cfg["epochs"]
        self.lr         = pipe_cfg["learning_rate"]
        self.wd         = pipe_cfg["weight_decay"]
        self.save_dir   = Path(pipe_cfg["save_dir"]) / category
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = UNetAutoencoder(latent_dim=pipe_cfg["latent_dim"]).to(self.device)
        self.criterion = ReconstructionLoss(alpha=0.5)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)

        self.use_amp = (
            cfg["device"].get("mixed_precision", False)
            and self.device.type == "cuda"
        )
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        ds_cfg = cfg["dataset"]
        self.loader = get_train_loader(
            root=ds_cfg["root"],
            category=category,
            batch_size=pipe_cfg["batch_size"],
            num_workers=ds_cfg["num_workers"],
            image_size=ds_cfg["image_size"],
            for_reconstruction=True,
        )

        console.print(f"[cyan]Category        : {category}[/cyan]")
        console.print(f"[cyan]Training images : {len(self.loader.dataset)}[/cyan]")
        console.print(f"[cyan]Epochs          : {self.epochs}[/cyan]")
        console.print(f"[cyan]Mixed precision : {self.use_amp}[/cyan]")

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.loader:
            imgs = batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    recon = self.model(imgs)
                    loss  = self.criterion(recon, imgs)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                recon = self.model(imgs)
                loss  = self.criterion(recon, imgs)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.loader)

    def train(self):
        console.print(f"\n[bold]Starting training for [{self.category}][/bold]")
        best_loss = float("inf")
        history = []

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            loss = self.train_epoch()
            self.scheduler.step()
            elapsed = time.time() - t0
            history.append(loss)

            console.print(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"loss={loss:.5f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"({elapsed:.1f}s)"
            )

            # Save best model
            if loss < best_loss:
                best_loss = loss
                ckpt_path = self.save_dir / "best_model.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "loss": best_loss,
                    "category": self.category,
                }, ckpt_path)

        # Save final model
        torch.save(self.model.state_dict(), self.save_dir / "final_model.pth")
        console.print(f"\n[green]Training complete. Best loss: {best_loss:.5f}[/green]")
        console.print(f"[green]Checkpoints saved to: {self.save_dir}[/green]")
        return history


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", default=None, help="Single category to train")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    categories = (
        [args.category] if args.category
        else cfg["dataset"]["categories"]
    )

    for cat in categories:
        console.rule(f"[bold blue]{cat}[/bold blue]")
        trainer = AutoencoderTrainer(cfg, cat)
        trainer.train()

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial dims don't match exactly
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[-1] - x.shape[-1],
                          0, skip.shape[-2] - x.shape[-2]])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        #Encoder
        self.enc1 = ConvBlock(3, 64)        
        self.enc2 = Down(64, 128)          
        self.enc3 = Down(128, 256)           
        self.enc4 = Down(256, 512)         

        #Bottleneck 
        self.bottleneck = Down(512, latent_dim) 

        # ── Decoder 
        self.dec4 = Up(latent_dim, 512, 512)
        self.dec3 = Up(512, 256, 256)
        self.dec2 = Up(256, 128, 128)
        self.dec1 = Up(128, 64, 64)

        #Output head 
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decode with skip connections
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = torch.sigmoid(self.out_conv(d1))
        return out

    def compute_residual(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        
        with torch.no_grad():
            recon = self.forward(x)
        residual = torch.abs(x - recon)
        anomaly_score = float(residual.mean().item())
        anomaly_area = float((residual.mean(dim=1) > 0.05).float().mean().item())
        return recon, residual, anomaly_score, anomaly_area

if __name__ == "__main__":
    model = UNetAutoencoder(latent_dim=256)
    x = torch.randn(2, 3, 256, 256)
    recon = model(x)
    print(f"Input shape      : {x.shape}")
    print(f"Reconstruction   : {recon.shape}")

    recon, res, score, area = model.compute_residual(x)
    print(f"Residual map     : {res.shape}")
    print(f"Anomaly score    : {score:.4f}")
    print(f"Anomaly area     : {area*100:.2f}%")
    print(f"Params           : {sum(p.numel() for p in model.parameters()):,}")
    print("UNetAutoencoder OK ✓")

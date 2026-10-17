import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Double Convolution Block
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Shared Encoder
# ============================================================

class SharedEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):

        # Encoder level 1
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        # Encoder level 2
        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        # Encoder level 3
        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Encoder level 4
        e4 = self.enc4(p3)

        return e1, e2, e3, e4


# ============================================================
# Task Decoder
# ============================================================

class TaskDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        self.output = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(
        self,
        e1,
        e2,
        e3,
        e4
    ):

        # Decoder level 3
        x = self.up3(e4)

        x = torch.cat(
            [x, e3],
            dim=1
        )

        x = self.dec3(x)

        # Decoder level 2
        x = self.up2(x)

        x = torch.cat(
            [x, e2],
            dim=1
        )

        x = self.dec2(x)

        # Decoder level 1
        x = self.up1(x)

        x = torch.cat(
            [x, e1],
            dim=1
        )

        x = self.dec1(x)

        # Binary segmentation output
        x = self.output(x)

        return x


# ============================================================
# Multi-Task Network
# ============================================================

class MultiTaskUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # One shared encoder
        self.encoder = SharedEncoder()

        # Separate task-specific decoders
        self.lane_decoder = TaskDecoder()
        self.road_decoder = TaskDecoder()

    def forward(self, x):

        # Shared representation
        e1, e2, e3, e4 = self.encoder(x)

        # Task-specific predictions
        lane_output = self.lane_decoder(
            e1, e2, e3, e4
        )

        road_output = self.road_decoder(
            e1, e2, e3, e4
        )

        return {
            "lane": lane_output,
            "road": road_output
        }


# ============================================================
# Model Test
# ============================================================

if __name__ == "__main__":

    from config import DEVICE

    print("=" * 60)
    print("Multi-Task U-Net Test")
    print("=" * 60)

    model = MultiTaskUNet().to(DEVICE)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # Test input
    x = torch.randn(
        1,
        3,
        384,
        1248,
        device=DEVICE
    )

    print("Input shape:", x.shape)

    # Forward pass
    with torch.no_grad():

        outputs = model(x)

    print(
        "Lane output shape:",
        outputs["lane"].shape
    )

    print(
        "Road output shape:",
        outputs["road"].shape
    )

    # Parameter count
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    print("=" * 60)
    print("Model test completed successfully.")
    print("=" * 60)
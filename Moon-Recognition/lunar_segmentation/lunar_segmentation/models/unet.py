import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]  # Computes the difference in height between the two feature maps
        diff_x = x2.size()[3] - x1.size()[3]  # Computes the difference in width between the two feature maps

        # Apply symmetric padding to x1 to match the size of x2
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        # Concatenate the upsampled feature map (x1) with the corresponding feature map from the downsampling path (x2)
        x = torch.cat([x2, x1], dim=1)  # Tot channels = channels of x2 + channels of x1 = in_ch
        return self.conv(x)  # output channels = out_ch


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SmallUNet(nn.Module):
    """U-Net with configurable depth, width, and regularization.

    Args:
        in_channels:        Number of input channels (default 3).
        num_classes:        Number of output segmentation classes (default 1).
        base_width:         Filters in the first encoder block. Each level
                            doubles this up to the bottleneck (default 32).
        depth:              Number of Down/Up pairs, excluding the initial
                            DoubleConv. 3 = original, 4 = one extra level
                            (default 4).
        bottleneck_dropout: Dropout probability applied at the bottleneck
                            only. Good for small datasets (default 0.3).
        decoder_dropout:    Dropout probability applied inside each decoder
                            block. Set to 0.0 to disable (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_width: int = 32,
        depth: int = 4,
        bottleneck_dropout: float = 0.3,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        # --- Encoder ---
        self.inc = DoubleConv(in_channels, base_width)
        self.downs = nn.ModuleList()
        in_ch = base_width
        for i in range(depth):
            out_ch = in_ch * 2
            # Extra dropout at the deepest encoder step (just before bottleneck)
            drop = bottleneck_dropout if i == depth - 1 else 0.0
            self.downs.append(Down(in_ch, out_ch, dropout=drop))
            in_ch = out_ch

        # --- Decoder ---
        # in_ch is now base_width * 2**depth (bottleneck channels)
        self.ups = nn.ModuleList()
        for i in range(depth):
            out_ch = in_ch // 2
            drop = decoder_dropout if i < depth - 1 else 0.0
            self.ups.append(Up(in_ch, out_ch, dropout=drop))
            in_ch = out_ch

        self.outc = OutConv(base_width, num_classes)

    def forward(self, x):
        # Encoder: save each skip-connection feature map
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        # Bottleneck is the last element; pop it as the starting point
        x = skips.pop()

        # Decoder: pair each Up block with its skip connection
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)

        return self.outc(x)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configs = [
        dict(depth=3, base_width=32),   # original size
        dict(depth=4, base_width=32),   # one extra level (recommended)
        dict(depth=4, base_width=64),   # wider + deeper
    ]
    x = torch.randn(2, 3, 256, 256)
    for cfg in configs:
        model = SmallUNet(in_channels=3, num_classes=4, **cfg)
        out = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"depth={cfg['depth']} base_width={cfg['base_width']:>3} | "
            f"output: {tuple(out.shape)} | params: {params:.2f}M"
        )

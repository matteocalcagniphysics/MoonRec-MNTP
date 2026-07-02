import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]    # Computes the difference in height between the two feature maps
        diff_x = x2.size()[3] - x1.size()[3]    # Computes the difference in width between the two feature maps
        # Apply symmetric padding to x1 to match the size of x2
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)  
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_width: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_width)      # 3x256x256 -> 32x256x256
        self.down1 = Down(base_width, base_width * 2)       # 32x256x256 -> 64x128x128
        self.down2 = Down(base_width * 2, base_width * 4)   # 64x128x128 -> 128x64x64
        self.down3 = Down(base_width * 4, base_width * 8)   # 128x64x64 -> 256x32x32
        self.up1 = Up(base_width * 8, base_width * 4)
        self.up2 = Up(base_width * 4, base_width * 2)
        self.up3 = Up(base_width * 2, base_width)
        self.outc = OutConv(base_width, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

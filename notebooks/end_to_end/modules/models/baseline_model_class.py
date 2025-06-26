
import torch
import torch.nn as nn
import torchvision.models as models

class HeatmapHead(nn.Module):
    def __init__(self, num_keypoints, in_channels):
        super().__init__()
        # three convolutional layers: 8×8 → 16×16 → 32×32 → 64×64
        #resnet initially makes the image smaller by 32 so in our case the picture becomes 8x8
        self.deconv = nn.Sequential(
            # 1st deconv: 512→256 channels, doubles spatial size (8→16)
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 2nd deconv: 256→256 channels, doubles spatial size (16→32)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 3rd deconv: 256→256 channels, doubles spatial size (32→64)
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Final 1×1 conv: 256→num_keypoints channels, keeps spatial size 64×64
        self.final = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = self.deconv(x)  #[B,256,64,64]
        x = self.final(x)   #[B,num_keypoints,64,64]
        return x

class KeypointHeatmapNet(nn.Module):
    def __init__(self, num_keypoints=50):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = HeatmapHead(num_keypoints, in_channels=512)

    def forward(self, x):
        # x: [B, 3, 256, 256], B is batch size
        feat = self.backbone(x)     # feat: [B, 2048, 8, 8]
        heatmaps = self.head(feat)  # heatmaps: [B, 50, 64, 64]
        return heatmaps

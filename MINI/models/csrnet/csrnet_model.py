import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        vgg = models.vgg16(pretrained=load_weights)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 64,  3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 1)
        )

        # EXACT initialization used by the CSRNet authors
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x
# === model.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDQN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),  # Input: (8, 7, 5)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 5 + 5, 256),  # 64 conv features + 5 aux features
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, viewcone, aux):
        x = self.conv(viewcone)              # shape: (B, 64, 7, 5)
        x = x.view(x.size(0), -1)            # flatten to (B, 64*7*5)
        x = torch.cat([x, aux], dim=1)       # concat aux: (B, features + 5)
        return self.fc(x)

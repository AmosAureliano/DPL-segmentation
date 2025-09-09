import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torchmetrics
import matplotlib.pyplot as plt

import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder_layers = list(self.encoder.children())
        self.enc1 = nn.Sequential(*self.encoder_layers[:3])
        self.enc2 = nn.Sequential(*self.encoder_layers[3:5])
        self.enc3 = self.encoder_layers[5]
        self.enc4 = self.encoder_layers[6]
        self.enc5 = self.encoder_layers[7]

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.up1(e5)
        d2 = self.up2(d1)
        d3 = self.up3(d2)
        d4 = self.up4(d3)
        return self.final(d4)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.PILToTensor()
    ])

    dataset = datasets.OxfordIIITPet(
        root="./data",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    subset = Subset(dataset, range(1000))
    train_loader = DataLoader(subset, batch_size=8, shuffle=True)

    model = UNet(n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.squeeze(1).long().to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "models/unet_pet.pth")
    print("Modelo salvo em models/unet_pet.pth")

if __name__ == "__main__":
    main()

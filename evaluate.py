import torch
import torch.nn as nn
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from train import UNet

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

    subset = Subset(dataset, range(200))
    loader = DataLoader(subset, batch_size=4, shuffle=False)

    model = UNet(n_classes=2).to(device)
    model.load_state_dict(torch.load("models/unet_pet.pth", map_location=device))
    model.eval()

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=2).to(device)
    acc = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.squeeze(1).long().to(device)
            out = model(imgs)
            preds = out.argmax(1)
            iou.update(preds, masks)
            acc.update(preds, masks)

    print(f"IoU: {iou.compute().item():.4f}")
    print(f"Accuracy: {acc.compute().item():.4f}")

    # Exemplo de visualização
    img, mask = dataset[0]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(1).cpu().squeeze()

    plt.subplot(1,3,1)
    plt.imshow(img.permute(1,2,0))
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(mask.squeeze(), cmap="gray")
    plt.title("Ground Truth")

    plt.subplot(1,3,3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predição")
    plt.show()

if __name__ == "__main__":
    main()

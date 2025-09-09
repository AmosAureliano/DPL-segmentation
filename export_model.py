import torch
from train import UNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(n_classes=2).to(device)
    model.load_state_dict(torch.load("models/unet_pet.pth", map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 128).to(device)

    traced = torch.jit.trace(model, dummy_input)
    traced.save("models/unet_pet.pt")
    print("Modelo exportado para TorchScript em models/unet_pet.pt")

if __name__ == "__main__":
    main()

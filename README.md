# Segmentação Semântica

Este projeto implementa uma rede **U-Net** para segmentação semântica no dataset **Oxford-IIIT Pet**.  
O modelo é treinado em PyTorch, avaliado com IoU e Acurácia, e exportado para uso em um **aplicativo Android**.

## 🚀 Como rodar

### Treinamento
```bash
pip install -r requirements.txt
python train.py
```

### Avaliação
```bash
python evaluate.py
```

### Exportação para TorchScript
```bash
python export_model.py
```

### Aplicativo Android
O app está na pasta `app/` e usa **PyTorch Mobile**.  
Coloque o modelo exportado (`unet_pet.pt`) em `app/assets/`.


import torch.nn as nn
import timm

class ViTDiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

from Backbones.ResNet import resnet10, resnet18
from Backbones.fedavgnet import FedAvgNetCIFAR

Backbone_NAMES = {
    "resnet10": resnet10,
    "resnet18": resnet18,
    "fedavg_cifar": FedAvgNetCIFAR,
}


def get_private_backbones(cfg):
    if not isinstance(cfg.DATASET.backbone, str):
        raise TypeError("DATASET.backbone must be a string in the simplified public repository.")

    if cfg.DATASET.backbone not in Backbone_NAMES:
        raise ValueError(f"Unsupported backbone: {cfg.DATASET.backbone}")

    priv_models = []
    for _ in range(cfg.DATASET.parti_num):
        if cfg.DATASET.backbone == "fedavg_cifar":
            priv_model = Backbone_NAMES[cfg.DATASET.backbone](cfg.DATASET.n_classes)
        else:
            priv_model = Backbone_NAMES[cfg.DATASET.backbone](cfg)
        priv_models.append(priv_model)
    return priv_models

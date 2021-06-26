import torchvision


def get_backbone(name, pretrained=False):
    backbones = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in backbones.keys():
        raise KeyError(f"{name} is not a valid backbone architecture")
    return backbones[name]

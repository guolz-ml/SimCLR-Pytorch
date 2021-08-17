import torchvision

def get_backbone(name, pretrained=False):
    backbones = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "wide_resnet50_2": torchvision.models.wide_resnet50_2(pretrained=pretrained)
    }
    if name not in backbones.keys():
        raise KeyError(f"{name} is not a valid ResNet version")

    model = backbones[name]
    if name == 'resnet18' or 'resnet50':
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    return model

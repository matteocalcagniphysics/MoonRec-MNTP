import torch.nn as nn
from backbone import ResNetFeatureMapsExtractor
from utils import _get_shapes
import torchvision as tv

resnet = tv.models.resnet.__dict__['resnet18'](weights="DEFAULT")
backbone = ResNetFeatureMapsExtractor(model=resnet)

shapes_list = _get_shapes(model=backbone, channels=3, size=(256, 256))

for shape in shapes_list:
    print(shape)

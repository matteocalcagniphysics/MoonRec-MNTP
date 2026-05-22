# PANOPTIC MODULE STRUCTURE

The module should be read starting from ```factory.py```, in which there is the function **make_fpn_resnet**. This function takes in input the following arguments:

* **name (str, optional)**: Name of the resnet backbone. Only those available in torchvision are supported. Defaults to 'resnet18'.
* **fpn_type (str, optional)**: Type of FPN. 'fpn' | 'panoptic' | 'panet'. Defaults to 'fpn'.
* o**ut_size (Tuple[int, int], optional)**: Size of segmentation output. Defaults to (224, 224).
* **fpn_channels (int, optional)**: Number of hidden channels to use in the FPN. Defaults to 256.
* **num_classes (int, optional)**: Number of classes for which to make predictions. Determines the channel width of the output. Defaults to 1000.
* **pretrained (bool, optional)**: Whether to use pretrained backbone. Defaults to True.
* **in_channels (int, optional)**: Channel width of the input. If less than 3, conv1 is replaced with a smaller one. Defaults to 3.
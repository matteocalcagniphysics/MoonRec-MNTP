from typing import Tuple, Sequence, Optional, Iterable
import torch
from torch import nn
from typing import Any
from torchvision.models.detection.image_list import ImageList   
from collections import OrderedDict
from .PAN2_layers import Interpolate, Sum
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import FeaturePyramidNetwork


class Parallel(nn.ModuleList):
    ''' Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''

    def forward(self, xs: Any) -> tuple:
        # if multiple inputs, pass the 1st input through the 1st module,
        # the 2nd input through the 2nd module, and so on.
        if isinstance(xs, (list, tuple)):
            return tuple(m(x) for m, x in zip(self, xs))
        # if single input, pass it through all modules
        return tuple(m(xs) for m in self)


class SemanticBranch(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Panoptic Feature Pyramid Networks" by Kirilov et al.,
    https://arxiv.com/abs/1901.02446.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    a single final output, with out_channels channels.

    Architecture diagram:

    nth feat. map ────[nth in_conv]─────────>───[nth upsampler]──────────┐
                                                                         │
                                                                         │
                                                                         V
    (n-1)th feat. map ──[(n-1)th in_conv]───>───[(n-1)th upsampler]────>(+)
                                                                         │
                                                                         │
                                                                         V
          .                     .                     .
          .                     .                     .
          .                     .                     .
                                                                         │
                                                                         │
                                                                         V
    1st feat. map ────[1st in_conv]─────────>───[1st upsampler]─────────(+)
                                                                         │
                                                                         │
                                                                         V
                                                                        out
    """

    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2,
                 out_size: Optional[int] = None,
                 num_upsamples_per_layer: Optional[Sequence[int]] = None,
                 upsamplng_factor: int = 2,
                 num_groups_for_norm: int = 32):
        """Constructor.

        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
            out_size (Optional[int], optional): Size of output. If None, 
                the size of the first feature map will be used.
                Defaults to None.
            num_upsamples_per_layer (Optional[Sequence[int]], optional): Number
                of upsampling iterations for each feature map. Will depend on
                the size of the feature map. Each upsampling iteration
                comprises a conv-group_norm-relu block followed by a scaling
                using torch.nn.functional.interpolate.
                If None, each feature map is assumed to be half the size of the
                preceeding one, meaning that it requires one more upsampling
                iteration than the last one.
                Defaults to None.
            upsamplng_factor (int, optional): How much to scale per upsampling
                iteration. Defaults to 2.
            num_groups_for_norm (int, optional): Number of groups for group
                norm layers. Defaults to 32.
        """
        # in_feats_shapes is a tuple with as many entries as the number of 
        # layers of my network. 
        if num_upsamples_per_layer is None:
            num_upsamples_per_layer = list(range(len(in_feats_shapes)))
        
        # It automatically calculates the output Height and width
        # For me is useless since I give it the correct output size (the same as the
        # input)
        if out_size is None:
            out_size = in_feats_shapes[0][-2:]
        
        # This transforms every layer output of the backbone. The number of channels
        # gets standardized to 256
        in_convs = Parallel([
            nn.Conv2d(s[1], hidden_channels, kernel_size=1)
            for s in in_feats_shapes
        ])

        # Defines the upsamplers instance for the Pyramid coming from the backbone
        upsamplers = self._make_upsamplers(
            in_channels=hidden_channels,
            size=out_size,
            num_upsamples_per_layer=num_upsamples_per_layer,
            num_groups=num_groups_for_norm)
        
        # Final output convolution, so that the number of channels corresponds to 
        # the number of classes (2 since this is a semantic branch)
        out_conv = nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1)

        # yapf: disable
        layers = [
            in_convs,        # Standardize the channel number (256)
            upsamplers,      # Upsample process (Final channels = 128)
            Sum(),           # Sums all the feature maps obtained element-wise
            out_conv         # Final convolution (out channels = num classes = 2)
        ]
        # yapf: enable
        super().__init__(*layers)

    @classmethod
    def _make_upsamplers(cls,
                         in_channels: int,
                         size: int,
                         num_upsamples_per_layer: Iterable[int],
                         num_groups: int = 32) -> Parallel:
        # For every layer it defines the correct upsampler and appends everything
        # to the list layers
        layers = []
        for num_upsamples in num_upsamples_per_layer:
            upsampler = cls._upsample_feat(
                in_channels=in_channels,
                num_upsamples=num_upsamples,
                size=size,
                num_groups=num_groups)
            layers.append(upsampler)
        
        # This instance applies the list of upsamplers to the tuple of outputs coming
        # from the backbone, It will give me the final "pyramid" of the semantic branch 
        upsamplers = Parallel(layers)
        return upsamplers

    @classmethod
    def _upsample_feat(cls,
                       in_channels: int,
                       num_upsamples: int,
                       size: int,
                       scale_factor: float = 2.,
                       num_groups: int = 32) -> nn.Sequential:
        # This means that I am upsampling the most generic feature map 
        # Return a single upsampling block (or step)
        if num_upsamples == 0:
            return cls._make_upsampling_block(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                scale=1,        # No scaling needed
                num_groups=num_groups)
        # Let's say that num_upsamples = 3. Then the first two blocks do both
        # the convolution step and the scaling up. In the last one I reached 
        # the correct size, so I do a last conv step halving the number of channels
        blocks = []
        for _ in range(num_upsamples - 1):
            blocks.append(
                cls._make_upsampling_block(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    scale=scale_factor,
                    num_groups=num_groups))
        blocks.append(
            cls._make_upsampling_block(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                size=size,
                num_groups=num_groups))
        return nn.Sequential(*blocks)

    @classmethod
    def _make_upsampling_block(cls,
                               in_channels: int,
                               out_channels: int = None,
                               scale: float = 2,
                               size: int = None,
                               num_groups: int = 32) -> nn.Sequential:
        if out_channels is None:
            out_channels = in_channels

        # conv block that preserves size. It takes the feature map and output a 
        # new one with the same H x W but different number of channels
        conv_block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_channels=out_channels, num_groups=num_groups),
            nn.ReLU(inplace=True)
        ]
        if scale == 1: # This means that I want to keep the size unchanged
            return nn.Sequential(*conv_block)

        if size is None: 
            # If I do not give a fine size (H x W) I scale the one that I have by a 
            # scale factor
            upsample_layer = Interpolate(scale_factor=scale)
        else:
            upsample_layer = Interpolate(size=size)
        
        # Complete upsampling step for a generic layer
        conv_block.append(upsample_layer) 

        return nn.Sequential(*conv_block)


class CustomMaskRCNNHeads(nn.Module):

    """
    FPN Output: Dictionary (Features)

    RPN Input: Dictionary (Features) + List of Tensors (Images)

    RPN Output: List of Tensors (Box coordinates)

    Pooler Input: Dictionary (Features) + List of Tensors (Box coordinates)

    Pooler Output: One massive 4D Tensor
    
    """

    def __init__(self, in_channels=256, num_classes=8):
        super().__init__()
        
        # 1. Region Proposal Network (RPN)
        # Reverted to 5 levels because your backbone outputs 5 feature maps!
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )

        # FIX 1: Use Torchvision's RPNHead instead of nn.Sequential
        # It automatically handles the list of feature maps and calculates the regression layers
        rpn_head = RPNHead(
            in_channels, 
            anchor_generator.num_anchors_per_location()[0]
        )

        self.rpn = RegionProposalNetwork(
            anchor_generator, rpn_head,
            fg_iou_thresh=0.7, bg_iou_thresh=0.3, 
            batch_size_per_image=256, positive_fraction=0.5, 
            pre_nms_top_n={'training': 2000, 'testing': 1000}, 
            nms_thresh=0.7, 
            post_nms_top_n={'training': 2000, 'testing': 1000} 
        )
        
        # 2. Multi-Scale RoI Poolers
        # FIX 2: Added '4' to the featmap_names so they don't ignore the 5th map
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3', '4'], 
            output_size=7,
            sampling_ratio=2
        )
        
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3', '4'],
            output_size=14,
            sampling_ratio=2
        )
        
        # 3. Box Head Components
        # TwoMLPHead flattens the 7x7 features and passes them through two Linear layers
        # defines to which class this box belongs and how to adjust the box coordinates
        box_head = TwoMLPHead(in_channels * 7 * 7, representation_size=1024)
        box_predictor = FastRCNNPredictor(in_channels=1024, num_classes=num_classes)
        
        # 4. Mask Head Components
        # A sequence of 4 convolutional layers to extract spatial patterns
        # This understand what pixels belong to the object in the box and which don't, 
        # thus creating a binary mask for each class
        mask_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
        )
        # Upsamples 14x14 features to 28x28 and predicts masks per class
        mask_predictor = MaskRCNNPredictor(in_channels=in_channels, dim_reduced=256, num_classes=num_classes)
        
        # 5. Combine into RoIHeads
        # This standard torchvision class manages the coordination between box and mask predictions
        self.roi_heads = RoIHeads(
            box_roi_pool=self.box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,       # Filters out absolutely terrible predictions early to save compute
            nms_thresh=0.5,          # IoU threshold for Non-Maximum Suppression (removing duplicate overlapping boxes)
            detections_per_img=100,  # Maximum number of final objects it is allowed to output per image
            mask_roi_pool=self.mask_roi_pool,
            mask_head=mask_head,
            mask_predictor=mask_predictor
        )

    def forward(self, fp_features, images, targets=None):
        """
        Args:
            fp_features (dict): Dictionary mapping FPN level names (e.g., '0','1','2','3') to features
            images (ImageList): Torchvision ImageList wrapper containing image shapes
            targets (list[dict]): Ground truth annotations during training
        """
        # Convert dict keys if necessary to match expectations
        # Pass features through RPN to get candidate bounding boxes (proposals)
        proposals, rpn_losses = self.rpn(images, fp_features, targets)
        
        # Pass features and proposals through the RoI heads to generate detections/losses
        detections, roi_losses = self.roi_heads(fp_features, proposals, images.image_sizes, targets)
         
        return detections, rpn_losses, roi_losses

class PanopticFPN(nn.Module):
    # Added in_channels_list to map the raw ResNet18 outputs
    def __init__(self, backbone, semantic_branch, instance_branch, in_channels_list=[64, 64, 128, 256, 512], out_channels=256):
        super().__init__()
        self.backbone = backbone
        self.semantic_branch = semantic_branch 
        self.instance_branch = instance_branch  
        
        # FIX: The missing bridge! This will standardize the ResNet features into 256 channels
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

    def forward(self, images_list, targets=None):
        # 1. Format images
        image_sizes = [img.shape[-2:] for img in images_list]
        batched_image_tensor = torch.stack(images_list)
        image_list_obj = ImageList(batched_image_tensor, image_sizes)
        
        # 2. Extract RAW features from ResNet (List of 64, 64, 128, 256, 512 channels)
        fp_features_list = self.backbone(batched_image_tensor)
        
        # 3. The Semantic Branch accepts the RAW list directly
        semantic_output = self.semantic_branch(fp_features_list)
        
        # 4. Convert the RAW list to an OrderedDict for the FPN
        fp_features_dict = OrderedDict([
            (str(i), feat) for i, feat in enumerate(fp_features_list)
        ])
        
        # 5. FIX: Pass the raw dictionary through the new FPN bridge! 
        # This returns a NEW dictionary where every tensor perfectly has 256 channels.
        fpn_features_dict = self.fpn(fp_features_dict)
        
        # 6. Pass the STANDARDIZED FPN dictionary to your Instance Head!
        detections, rpn_losses, roi_losses = self.instance_branch(fpn_features_dict, image_list_obj, targets)
        
        return semantic_output, detections, rpn_losses, roi_losses
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import numpy as np
from torch import nn, Tensor
import math
import torch
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, Union, Sequence

from .transformer import LinearAttnFFN
from .base_module import BaseModule
from ..misc.profiler import module_profile
from ..layers import ConvLayer, get_normalization_layer
from ..layers.frozen_bn import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class SeparableSelfAttentionHeadModule(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        feat_sz=16,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 1,
        patch_w: Optional[int] = 1,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:

        super(SeparableSelfAttentionHeadModule, self).__init__()

        self.pre_ssat_cls = ConvLayer(
            opts=opts["cls"],
            in_channels=opts["cls"]["in_channels"],
            out_channels=opts["cls"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.pre_ssat_reg = ConvLayer(
            opts=opts["reg"],
            in_channels=opts["reg"]["in_channels"],
            out_channels=opts["reg"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.global_rep_cls, attn_unit_dim = self._build_attn_layer(
            opts=opts["cls"],
            d_model=opts["cls"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["cls"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.global_rep_reg, attn_unit_dim = self._build_attn_layer(
            opts=opts["reg"],
            d_model=opts["reg"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["reg"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        # center predict
        freeze_bn = False
        cls_in_channel = self.pre_ssat_cls.out_channels
        self.conv1_ctr = conv(cls_in_channel, cls_in_channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(cls_in_channel, cls_in_channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(cls_in_channel // 2, cls_in_channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = nn.Conv2d(cls_in_channel // 4, 1, kernel_size=1)

        # offset regress with conv weights
        cls_in_reg = self.pre_ssat_reg.out_channels
        self.conv1_offset = conv(cls_in_reg, cls_in_reg, freeze_bn=freeze_bn)
        self.conv2_offset = conv(cls_in_reg, cls_in_reg // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(cls_in_reg // 2, cls_in_reg // 4, freeze_bn=freeze_bn)
        self.conv4_offset = nn.Conv2d(cls_in_reg // 4, 2, kernel_size=1)

        # size regress with conv weights
        self.conv1_size = conv(cls_in_reg, cls_in_reg, freeze_bn=freeze_bn)
        self.conv2_size = conv(cls_in_reg, cls_in_reg // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(cls_in_reg // 2, cls_in_reg // 4, freeze_bn=freeze_bn)
        self.conv4_size = nn.Conv2d(cls_in_reg // 4, 2, kernel_size=1)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.feat_sz = feat_sz

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = opts["cls"]["in_channels"]
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

        self.enable_coreml_compatible_fn = getattr(
            opts, "common.enable_coreml_compatible_module", False
        )
        # self.enable_coreml_compatible_fn = True

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
        self,
        opts,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
        )
        assert (
            self.patch_h == self.patch_w
        ), "For Coreml, we need patch_h and patch_w are the same"
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        x_cls = self.pre_ssat_cls(x)
        x_reg = self.pre_ssat_reg(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            x_cls_patches, x_cls_output_size = self.unfolding_coreml(x_cls)
            x_reg_patches, x_reg_output_size = self.unfolding_coreml(x_reg)
        else:
            x_cls_patches, x_cls_output_size = self.unfolding_pytorch(x_cls)
            x_reg_patches, x_reg_output_size = self.unfolding_pytorch(x_reg)

        # learn global representations on all patches
        x_cls_patches = self.global_rep_cls(x_cls_patches)
        x_reg_patches = self.global_rep_reg(x_reg_patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            x_cls = self.folding_coreml(patches=x_cls_patches, output_size=x_cls_output_size)
            x_reg = self.folding_coreml(patches=x_reg_patches, output_size=x_reg_output_size)
        else:
            x_cls = self.folding_pytorch(patches=x_cls_patches, output_size=x_cls_output_size)
            x_reg = self.folding_pytorch(patches=x_reg_patches, output_size=x_reg_output_size)

        """ Forward pass through the lightweight convolutional block. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x_cls, x_reg)

        # assert gt_score_map is None
        bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x_cls, x_reg):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x_cls)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        score_map_ctr = self.conv4_ctr(x_ctr3)

        x_offset1 = self.conv1_offset(x_reg)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        score_map_offset = self.conv4_offset(x_offset3)

        x_size1 = self.conv1_size(x_reg)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        score_map_size = self.conv4_size(x_size3)

        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        return res, params, macs

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str


class SeparableSelfAttentionLiteHeadModule(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        feat_sz=16,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 1,
        patch_w: Optional[int] = 1,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:

        super(SeparableSelfAttentionLiteHeadModule, self).__init__()

        self.pre_ssat_cls = ConvLayer(
            opts=opts["cls"],
            in_channels=opts["cls"]["in_channels"],
            out_channels=opts["cls"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.pre_ssat_reg = ConvLayer(
            opts=opts["reg"],
            in_channels=opts["reg"]["in_channels"],
            out_channels=opts["reg"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.global_rep_cls, attn_unit_dim = self._build_attn_layer(
            opts=opts["cls"],
            d_model=opts["cls"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["cls"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.global_rep_reg, attn_unit_dim = self._build_attn_layer(
            opts=opts["reg"],
            d_model=opts["reg"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["reg"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        # center predict
        freeze_bn = False
        cls_in_channel = self.pre_ssat_cls.out_channels
        self.conv1x1_ctr = nn.Conv2d(cls_in_channel, 1, kernel_size=1)

        # offset regress with conv weights
        cls_in_reg = self.pre_ssat_reg.out_channels
        self.conv1x1_offset = nn.Conv2d(cls_in_reg, 2, kernel_size=1)

        # size regress with conv weights
        self.conv1x1_size = nn.Conv2d(cls_in_reg, 2, kernel_size=1)

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.feat_sz = feat_sz

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = opts["cls"]["in_channels"]
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

        self.enable_coreml_compatible_fn = getattr(
            opts, "common.enable_coreml_compatible_module", False
        )
        # self.enable_coreml_compatible_fn = True

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
        self,
        opts,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def unfolding_coreml(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(
            batch_size, in_dim * self.patch_h * self.patch_w, n_patches_h, n_patches_w
        )
        assert (
            self.patch_h == self.patch_w
        ), "For Coreml, we need patch_h and patch_w are the same"
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        x_cls = self.pre_ssat_cls(x)
        x_reg = self.pre_ssat_reg(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            x_cls_patches, x_cls_output_size = self.unfolding_coreml(x_cls)
            x_reg_patches, x_reg_output_size = self.unfolding_coreml(x_reg)
        else:
            x_cls_patches, x_cls_output_size = self.unfolding_pytorch(x_cls)
            x_reg_patches, x_reg_output_size = self.unfolding_pytorch(x_reg)

        # learn global representations on all patches
        x_cls_patches = self.global_rep_cls(x_cls_patches)
        x_reg_patches = self.global_rep_reg(x_reg_patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            x_cls = self.folding_coreml(patches=x_cls_patches, output_size=x_cls_output_size)
            x_reg = self.folding_coreml(patches=x_reg_patches, output_size=x_reg_output_size)
        else:
            x_cls = self.folding_pytorch(patches=x_cls_patches, output_size=x_cls_output_size)
            x_reg = self.folding_pytorch(patches=x_reg_patches, output_size=x_reg_output_size)

        """ Forward pass through the lightweight convolutional block. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x_cls, x_reg)

        # assert gt_score_map is None
        bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x_cls, x_reg):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        score_map_ctr = self.conv1x1_ctr(x_cls) # ctr branch
        score_map_offset = self.conv1x1_offset(x_reg) # offset branch
        score_map_size = self.conv1x1_size(x_reg) # size branch

        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        return res, params, macs

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str


class SeparableSelfAttentionCornerHeadModule(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        attn_unit_dim: int,
        feat_sz=16,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 1,
        patch_w: Optional[int] = 1,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:

        super(SeparableSelfAttentionCornerHeadModule, self).__init__()

        self.pre_ssat_tl = ConvLayer(
            opts=opts["tl"],
            in_channels=opts["tl"]["in_channels"],
            out_channels=opts["tl"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.pre_ssat_br = ConvLayer(
            opts=opts["br"],
            in_channels=opts["br"]["in_channels"],
            out_channels=opts["br"]["in_channels"],
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        self.global_rep_tl, attn_unit_dim = self._build_attn_layer(
            opts=opts["cls"],
            d_model=opts["cls"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["cls"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.global_rep_br, attn_unit_dim = self._build_attn_layer(
            opts=opts["reg"],
            d_model=opts["reg"]["attn_unit_dim"],
            ffn_mult=ffn_multiplier,
            n_layers=opts["reg"]["attn_blocks"],
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        # top-left corner predictor
        freeze_bn = False
        tl_in_channel = self.pre_ssat_tl.out_channels
        self.conv1_tl = conv(tl_in_channel, tl_in_channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(tl_in_channel, tl_in_channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(tl_in_channel // 2, tl_in_channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = nn.Conv2d(tl_in_channel // 4, 1, kernel_size=1)

        # bottom-right corner predictor
        br_in_reg = self.pre_ssat_br.out_channels
        self.conv1_br = conv(br_in_reg, br_in_reg, freeze_bn=freeze_bn)
        self.conv2_br = conv(br_in_reg, br_in_reg // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(br_in_reg // 2, br_in_reg // 4, freeze_bn=freeze_bn)
        self.conv4_br = nn.Conv2d(br_in_reg // 4, 2, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.feat_sz = feat_sz

        self.enable_coreml_compatible_fn = getattr(
            opts, "common.enable_coreml_compatible_module", False
        )

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name="unfolding_weights",
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(
        self,
        opts,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model
            )
        )

        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        x_tl = self.pre_ssat_tl(x)
        x_br = self.pre_ssat_br(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            x_tl_patches, x_tl_output_size = self.unfolding_coreml(x_tl)
            x_br_patches, x_br_output_size = self.unfolding_coreml(x_br)
        else:
            x_tl_patches, x_tl_output_size = self.unfolding_pytorch(x_tl)
            x_br_patches, x_br_output_size = self.unfolding_pytorch(x_br)

        # learn global representations on all patches
        x_tl_patches = self.global_rep_tl(x_tl_patches)
        x_br_patches = self.global_rep_br(x_br_patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            x_tl = self.folding_coreml(patches=x_tl_patches, output_size=x_tl_output_size)
            x_br = self.folding_coreml(patches=x_br_patches, output_size=x_br_output_size)
        else:
            x_tl = self.folding_pytorch(patches=x_tl_patches, output_size=x_tl_output_size)
            x_br = self.folding_pytorch(patches=x_br_patches, output_size=x_br_output_size)

        """ Forward pass through the lightweight convolutional block. """
        score_map_tl, score_map_br = self.get_score_map(x_tl, x_br)

        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def forward(
        self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_score_map(self, x_tl, x_br):
        # top-left branch
        x_tl1 = self.conv1_tl(x_tl)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x_br)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        return res, params, macs

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)

        repr_str += "\n\t Local representations"
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.local_rep)

        repr_str += "\n\t Global representations with patch size of {}x{}".format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += "\n\t\t {}".format(m)
        else:
            repr_str += "\n\t\t {}".format(self.conv_proj)

        repr_str += "\n)"
        return repr_str
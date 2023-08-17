"""
Basic MobileViT-Track model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.mobilevit_track.layers.conv_layer import Conv2d
from .layers.neck import build_neck, build_feature_fusor
from .layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.utils.box_ops import box_xyxy_to_cxcywh
from easydict import EasyDict as edict


class MobileViTv2_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, neck, feature_fusor, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        if neck is not None:
            self.neck = neck
            self.feature_fusor = feature_fusor
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor, search: torch.Tensor):
        x, z = self.backbone(x=search, z=template)

        # Forward neck
        x, z = self.neck(x, z)

        # Forward feature fusor
        feat_fused = self.feature_fusor(z, x)

        # Forward head
        out = self.forward_head(feat_fused, None)

        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        # bs, Nq, C, HW = opt.size()
        # opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif "CENTER" in self.head_type:
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mobilevitv2_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if "mobilevitv2" in cfg.MODEL.BACKBONE.TYPE:
        width_multiplier = float(cfg.MODEL.BACKBONE.TYPE.split('-')[-1])
        backbone = create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN)
        if cfg.MODEL.BACKBONE.MIXED_ATTN is True:
            backbone.mixed_attn = True
        else:
            backbone.mixed_attn = False
        hidden_dim = backbone.model_conf_dict['layer4']['out']
        patch_start_index = 1
    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # build neck module to fuse template and search region features
    if cfg.MODEL.NECK:
        neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)
    else:
        neck = nn.Identity()

    if cfg.MODEL.NECK.TYPE == "BN_PWXCORR":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features = backbone.model_conf_dict['layer4']['out'],
                                               xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)
    elif cfg.MODEL.NECK.TYPE == "BN_SSAT" or cfg.MODEL.NECK.TYPE == "BN_HSSAT":
        feature_fusor = build_feature_fusor(cfg=cfg, in_features = backbone.model_conf_dict['layer4']['out'],
                                               xcorr_out_features=None)
    else:
        raise NotImplementedError

    # build head module
    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = MobileViTv2_Track(
        backbone=backbone,
        neck=neck,
        feature_fusor=feature_fusor,
        box_head=box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'mobilevit_track' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

        assert missing_keys == [] and unexpected_keys == [], "The backbone layers do not exactly match with the " \
                                                             "checkpoint state dictionaries. Please have a look at " \
                                                             "what those missing keys are!"

        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


def create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn):
    """
    function to create an instance of MobileViT backbone
    Args:
        pretrained:  str
        path to the pretrained image classification model to initialize the weights.
        If empty, the weights are randomly initialized
    Returns:
        model: nn.Module
        An object of Pytorch's nn.Module with MobileViT backbone (i.e., layer-1 to layer-4)
    """
    opts = {}
    opts['mode'] = width_multiplier
    opts['head_dim'] = None
    opts['number_heads'] = 4
    opts['conv_layer_normalization_name'] = 'batch_norm'
    opts['conv_layer_activation_name'] = 'relu'
    opts['mixed_attn'] = has_mixed_attn
    model = MobileViTv2_backbone(opts)

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        assert missing_keys == [], "The backbone layers do not exactly match with the checkpoint state dictionaries. " \
                                   "Please have a look at what those missing keys are!"

        print('Load pretrained model from: ' + pretrained)

    return model
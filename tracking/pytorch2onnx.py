"""
Basic MobileViTv2-Track model.
"""
import importlib
import math
import os
from typing import List
import numpy as np
import torch
import onnxruntime
import onnx
from torch import nn

from lib.models.mobilevit_track.layers.neck import build_neck, build_feature_fusor
from lib.models.mobilevit_track.layers.head import build_box_head

from lib.models.mobilevit_track.mobilevit_v2 import MobileViTv2_backbone
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation import Tracker

class MobileViTv2_Track(nn.Module):
    """ This is the base class for MobileViTv2-Track """

    def __init__(self, backbone, neck, feature_fusor, box_head, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.feature_fusor = feature_fusor
        self.box_head = box_head

        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)


    def forward(self, template: torch.Tensor, search: torch.Tensor):

        x, z = self.backbone(x=search, z=template) # Forward backbone
        x, z = self.neck(x, z) # Forward neck
        feat_fused = self.feature_fusor(z, x) # Forward feature fusor
        out = self.forward_head(feat_fused, None) # Forward head

        return out

    def forward_head(self, backbone_feature, gt_score_map=None):
        """
        backbone_feature: output embeddings of the backbone for search region
        """
        opt_feat = backbone_feature.contiguous()
        bs, _, _, _ = opt_feat.size()

        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, 1, 4)
        out = {'pred_boxes': outputs_coord_new,
               'score_map': score_map_ctr,
               'size_map': size_map,
               'offset_map': offset_map}

        return out


def build_mobilevitv2_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    # build mobilevitv2 backbone
    width_multiplier = float(cfg.MODEL.BACKBONE.TYPE.split('-')[-1])
    backbone = create_mobilevitv2_backbone(pretrained, width_multiplier, has_mixed_attn=cfg.MODEL.BACKBONE.MIXED_ATTN)
    hidden_dim = backbone.model_conf_dict['layer4']['out']

    # build neck module to fuse template and search region features
    neck = build_neck(cfg=cfg, hidden_dim=hidden_dim)

    feature_fusor = build_feature_fusor(cfg=cfg, in_features = backbone.model_conf_dict['layer4']['out'],
                                        xcorr_out_features=cfg.MODEL.NECK.NUM_CHANNS_POST_XCORR)

    # build head module
    box_head = build_box_head(cfg, cfg.MODEL.HEAD.NUM_CHANNELS)

    model = MobileViTv2_Track(
        backbone=backbone,
        neck=neck,
        feature_fusor=feature_fusor,
        box_head=box_head,
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


def get_parameters(name, parameter_name):
    """Get parameters."""
    param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
    params = param_module.parameters(parameter_name)
    return params

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def convert_tracking_model(net, pytorch_model_path, search_size = 256, template_size = 128):
    x = torch.randn(1, 3, search_size, search_size).cuda()
    z = torch.randn(1, 3, template_size, template_size).cuda()
    ort_inputs = {'x': to_numpy(x).astype(np.float32),
                  'z': to_numpy(z).astype(np.float32)}

    ########### complete model pytorch->onnx #############
    onnx_model_path = pytorch_model_path.split('.pth.tar')[0] + '.onnx'
    print("Converting tracking model now!")
    torch.onnx.export(net, (z, x), onnx_model_path, export_params=True,
                      opset_version=11, do_constant_folding=True, input_names=['z','x'], output_names=['cls','reg'])

    #######load the converted model and inference#######
    with torch.no_grad():
        oup = net(z, x)

    # onnx_model = onnx.load("model.onnx")
    # onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(oup['pred_boxes']), ort_outs[0], atol=1e-03)
    print("onnx model conversion is complete!")

def convert_template_model(net,template_size):
    zf = torch.randn(1, 3, template_size, template_size).cuda()
    onnx_inputs = {'zf': to_numpy(zf).astype(np.float32)}
    backbone = net.backbone
    print("Converting template model now!")
    torch.onnx.export(backbone, (zf), './models/back_cov.onnx_test', export_params=True,
          opset_version=11, do_constant_folding=True, input_names=['zf'],
          output_names=['out','pos'])
    with torch.no_grad():
        out,pos = backbone(zf)
        onnx_backbone = onnx.load("models/back_cov.onnx_test")
        onnx.checker.check_model(onnx_backbone)
        ort_session = onnxruntime.InferenceSession("./models/back_cov.onnx_test")
        ort_outs = ort_session.run(None, onnx_inputs)
        np.testing.assert_allclose(to_numpy(pos[0]), ort_outs[1], rtol=1e-03, atol=1e-05)
        print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(out[0])-ort_outs[0]))))
        print("The deviation between the second output: {}".format(np.max(np.abs(to_numpy(pos[0])-ort_outs[1]))))
    print(onnxruntime.get_device())
    print("Template onnx model has done!")


if __name__ == "__main__":

    # convert model to onnx
    tracker_name = "mobilevitv2_track"
    tracker_param = "mobilevitv2_256_128x1_ep300"
    tracker = Tracker(tracker_name, tracker_param, "video")

    settings = env_settings()
    params = get_parameters(tracker_name, tracker_param)

    network = build_mobilevitv2_track(params.cfg, training=False)
    # ckpt = torch.load(params.checkpoint, map_location='cpu')["net"]
    network.load_state_dict(torch.load(params.checkpoint, map_location='cpu')['net'], strict=True)

    use_gpu = True
    if use_gpu:
        network.cuda()
        network.eval()

    ######convert and check tracking pytorch model to onnx#####
    convert_tracking_model(network, params.checkpoint)
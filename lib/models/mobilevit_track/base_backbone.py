#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union, Any
from .misc import backbone_logger
from .misc.profiler import module_profile
from .misc.init_utils import initialize_weights


class BaseEncoder(nn.Module):
    """
    Base class for different classification models
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None

        self.dilation = 1

    def check_model(self):
        raise NotImplementedError

    def reset_parameters(self, opts):
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())

    def _forward_conv_layer(self, layer: nn.Module, x: Tensor) -> Tensor:
        return layer(x)

    def _forward_MobileViT_layer(self, layer: nn.Module, x: Tensor, z: Tensor):

        num_blocks = len(layer)

        # the first block is always MobilenetV2 with down-sampling
        MobilenetV2_block = layer[0]
        x = MobilenetV2_block(x)
        z = MobilenetV2_block(z)

        # compute output for remaining Transformer blocks (i.e., MobileViT/MobileViT-v2)
        for i in range(1, num_blocks):
            block = layer[i]
            x, z = block(x, z)

        return x, z

    def finetune_track(self, cfg, patch_start_index):
        print("Not Yet Implemented!")
        return None

    def forward_features(self, x, z):

        # conv_1 (i.e., the first conv3x3 layer) output for
        x = self._forward_conv_layer(self.conv_1, x)
        # z = self._forward_conv_layer(self.conv_1, z)

        # layer_1 (i.e., MobileNetV2 block) output
        x = self._forward_conv_layer(self.layer_1, x)
        # z = self._forward_conv_layer(self.layer_1, z)

        # layer_2 (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output
        x = self._forward_conv_layer(self.layer_2, x)
        # z = self._forward_conv_layer(self.layer_2, z)

        # layer_3 (i.e., MobileNetV2 with down-sampling + 2 x Separable Mixed Attention block) output
        x, z = self._forward_MobileViT_layer(self.layer_3, x, z)

        # layer_4 (i.e., MobileNetV2 with down-sampling + 4 x Separable Mixed Attention block) output
        x, z = self._forward_MobileViT_layer(self.layer_4, x, z)

        return x, z

    def forward(self, x: Tensor, z: Tensor):

        """
        Joint feature extraction and relation modeling for the MobileViT backbone.
        Args:
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]
            z (torch.Tensor): template feature, [B, C, H_z, W_z]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_x, C]
            attn : None
        """
        x, z = self.forward_features(x, z,)

        return x, z

    @staticmethod
    def _profile_layers(
        layers, input, overall_params, overall_macs, *args, **kwargs
        ) -> Tuple[Tensor, float, float]:
        if not isinstance(layers, list):
            layers = [layers]

        for layer in layers:
            if layer is None:
                continue
            input, layer_param, layer_macs = module_profile(module=layer, x=input)

            overall_params += layer_param
            overall_macs += layer_macs

            if isinstance(layer, nn.Sequential):
                module_name = "\n+".join([l.__class__.__name__ for l in layer])
            else:
                module_name = layer.__class__.__name__
            print(
                "{:<15} \t {:<5}: {:>8.3f} M \t {:<5}: {:>8.3f} M".format(
                    module_name,
                    "Params",
                    round(layer_param / 1e6, 3),
                    "MACs",
                    round(layer_macs / 1e6, 3),
                )
            )
            backbone_logger.singe_dash_line()
        return input, overall_params, overall_macs

    def profile_model(
        self, input: Tensor, is_classification: Optional[bool] = True, *args, **kwargs
                    ) -> Tuple[Union[Tensor, Dict[str, Tensor]], float, float]:
        """
        Helper function to profile a model.

        .. note::
            Model profiling is for reference only and may contain errors as it solely relies on user implementation to
            compute theoretical FLOPs
        """
        overall_params, overall_macs = 0.0, 0.0

        input_fvcore = input.clone()

        if is_classification:
            backbone_logger.log("Model statistics for an input of size {}".format(input.size()))
            backbone_logger.double_dash_line(dashes=65)
            print("{:>35} Summary".format(self.__class__.__name__))
            backbone_logger.double_dash_line(dashes=65)

        out_dict = {}
        input, overall_params, overall_macs = self._profile_layers(
            [self.conv_1, self.layer_1],
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l1"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_2,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l2"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_3,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l3"] = input

        input, overall_params, overall_macs = self._profile_layers(
            self.layer_4,
            input=input,
            overall_params=overall_params,
            overall_macs=overall_macs,
        )
        out_dict["out_l4"] = input

        backbone_logger.double_dash_line(dashes=65)
        print(
            "{:<20} = {:>8.3f} M".format("Overall parameters", overall_params / 1e6)
        )
        overall_params_py = sum([p.numel() for p in self.parameters()])
        print(
            "{:<20} = {:>8.3f} M".format(
                "Overall parameters (sanity check)", overall_params_py / 1e6
            )
        )

        # Counting Addition and Multiplication as 1 operation
        print(
            "{:<20} = {:>8.3f} M".format(
                "Overall MACs (theoretical)", overall_macs / 1e6
            )
        )

        # compute flops using FVCore
        try:
            # compute flops using FVCore also
            from fvcore.nn import FlopCountAnalysis

            flop_analyzer = FlopCountAnalysis(self.eval(), input_fvcore)
            flop_analyzer.unsupported_ops_warnings(False)
            flop_analyzer.uncalled_modules_warnings(False)
            flops_fvcore = flop_analyzer.total()

            print(
                "{:<20} = {:>8.3f} M".format(
                    "Overall MACs (FVCore)**", flops_fvcore / 1e6
                )
            )
            print(
                "\n** Theoretical and FVCore MACs may vary as theoretical MACs do not account "
                "for certain operations which may or may not be accounted in FVCore"
            )
        except Exception:
            pass

        print("Note: Theoretical MACs depends on user-implementation. Be cautious")
        backbone_logger.double_dash_line(dashes=65)

        return out_dict, overall_params, overall_macs
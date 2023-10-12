import math

from lib.models.mobilevit_track.mobilevitv2_track import build_mobilevitv2_track
from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils_mobilevit import Preprocessor, PreprocessorX_onnx
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import onnx, onnxruntime

class MobileViTv2Track(BaseTracker):

    def __init__(self, params, dataset_name):
        super(MobileViTv2Track, self).__init__(params)

        self.network = build_mobilevitv2_track(params.cfg, training=False)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        ort_device = onnxruntime.get_device()
        onnx_checkpoint = self.params.checkpoint.split('.pth')[0] + '.onnx'
        self.ort_session = onnxruntime.InferenceSession(onnx_checkpoint, providers=['CPUExecutionProvider'])

        # model_fp16 = float16.convert_float_to_float16(onnx.load(onnx_checkpoint))
        # onnx.save(model_fp16, "model_fp16.onnx")

        self.cfg = params.cfg
        if self.cfg.TEST.DEVICE == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        # self.network = network.to(self.device)
        # self.network.eval()

        self.preprocessor_onnx = PreprocessorX_onnx()
        # self.preprocessor = Preprocessor()

        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constraint
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        """
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # pre-compute the template features corresponding to first three of the tracker model (for speed-up)
        with torch.no_grad():
            self.z_dict = template
        """

        template_ort = self.preprocessor_onnx.process(z_patch_arr, z_amask_arr)
        self.z_dict_ort = template_ort

        self.box_mask_z = None

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        # ################################ pytorch-based inference ################################
        """
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        x_dict = search
        with torch.no_grad():
            # merge the template and the search
            out_dict = self.network.forward(
                template=self.z_dict.tensors.to(self.device), search=x_dict.tensors.to(self.device))
        """

        # ################################ complete model pytorch->onnx ##################################
        """
        ort_inputs = {'x': to_numpy(x_dict.tensors).astype(np.float32),
                      'z': to_numpy(self.z_dict.tensors).astype(np.float32)}

        print("Converting tracking model now!")
        torch.onnx.export(self.network, (self.z_dict.tensors.to(self.device), x_dict.tensors.to(self.device)),
                          'model.onnx', export_params=True, opset_version=11, do_constant_folding=True,
                          input_names=['z', 'x'], output_names=['cls', 'reg'])
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model, True)
        ort_session = onnxruntime.InferenceSession("model.onnx")
        ort_outs = ort_session.run(None, ort_inputs)
        np.testing.assert_allclose(to_numpy(out_dict['score_map']), ort_outs[1], atol=1e-03)
        print(
            "The deviation between the score map: {}".format(np.max(np.abs(to_numpy(out_dict['score_map']) - ort_outs[1]))))
        """

        ##################################################################################################
        """
        pred_score_map = out_dict['score_map']

        # add hann windows
        response = self.output_window * pred_score_map

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        best_bbox = self.map_box_back(pred_box, resize_factor)
        """

        # ################################## onnx-based inference ##################################
        search_ort = self.preprocessor_onnx.process(x_patch_arr, x_amask_arr)
        x_dict_ort = search_ort

        ort_inputs = {'x': x_dict_ort[0].astype(np.float32),
                      'z': self.z_dict_ort[0].astype(np.float32)}
        out_ort = self.ort_session.run(None, ort_inputs)

        pred_score_map_ort = out_ort[1]

        # add hann windows
        response_ort = self.output_window * torch.from_numpy(pred_score_map_ort).to(self.device)

        # response = pred_score_map
        pred_boxes_ort = self.network.box_head.cal_bbox(response_ort, torch.from_numpy(out_ort[2]).to(self.device),
                                                        torch.from_numpy(out_ort[3]).to(self.device))
        pred_boxes_ort = pred_boxes_ort.view(-1, 4)

        # Baseline: Take the mean of all pred boxes as the final result
        pred_box_ort = (pred_boxes_ort.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        best_bbox_ort = self.map_box_back(pred_box_ort, resize_factor)

        # get the final box result
        self.state = clip_box(best_bbox_ort, H, W, margin=10)

        """
        # visualize the search region with tracker bbox
        cv2.rectangle(image, (int(best_bbox[0]), int(best_bbox[1])),
                      (int(best_bbox[0] + best_bbox[2]), int(best_bbox[1] + best_bbox[3])), (0, 255, 0), 3)
        search_area_with_bbox, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)
        template_img = cv2.resize(self.z_patch_arr, (64, 64))
        search_area_with_bbox[0:64, 0:64,:] = template_img
        search_area_with_bbox = cv2.putText(search_area_with_bbox, '#{}'.format(self.frame_id), (192, 32),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow("img", search_area_with_bbox)
        # cv2.waitKey(0)
        
        # visualize the response map
        cls_map = (torch.squeeze(torch.squeeze(response, 0), 0) * 255).detach().byte().cpu().numpy()
        colored_cls_map = cv2.resize(cv2.applyColorMap(cls_map, cv2.COLORMAP_JET), (256, 256))
        """

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "search_area": x_patch_arr
                    # "target_response": colored_cls_map,
                    # "vis_bbox": search_area_with_bbox
                    }
        else:
            return {"target_bbox": self.state,
                    "search_area": x_patch_arr
                    # "target_response": colored_cls_map,
                    # "vis_bbox": search_area_with_bbox
                    }

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return MobileViTv2Track


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
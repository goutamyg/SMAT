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

from lib.test.tracker.data_utils_mobilevit import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class MobileViTv2Track(BaseTracker):

    def __init__(self, params, dataset_name):
        super(MobileViTv2Track, self).__init__(params)
        network = build_mobilevitv2_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        if self.cfg.TEST.DEVICE == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = Preprocessor()
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

        # save the model state dictionary only (to verify the actual model size on disk)
        self.save_state_dict = False
        if self.save_state_dict:
            model_name = self.params.checkpoint
            torch.save(network.state_dict(), model_name.split('.pth.tar')[0] + '_state_dict.pt')


    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)

        # pre-compute the template features corresponding to first three of the tracker model (for speed-up)
        with torch.no_grad():
            # conv_1 (i.e., the first conv3x3 layer) output for
            z = self.network.backbone.conv_1.forward(template.tensors.to(self.device))

            # layer_1 (i.e., MobileNetV2 block) output
            z = self.network.backbone.layer_1.forward(z)

            # layer_2 (i.e., MobileNetV2 with down-sampling + 2 x MobileNetV2) output
            z = self.network.backbone.layer_2.forward(z)

            self.z_dict1 = z

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
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            out_dict = self.network.forward(
                template=self.z_dict1.to(self.device), search=x_dict.tensors.to(self.device))

        pred_score_map = out_dict['score_map']

        # add hann windows
        response = self.output_window * pred_score_map
        # response = pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        best_bbox = self.map_box_back(pred_box, resize_factor)

        # get the final box result
        self.state = clip_box(best_bbox, H, W, margin=10)

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

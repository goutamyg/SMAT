import math

from lib.models.mobilevit_track.mobilevit_track import build_mobilevit_track
from lib.test.tracker.basetracker import BaseTracker
import torch

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


class MobileViTTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MobileViTTrack, self).__init__(params)
        network = build_mobilevit_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        if self.cfg.TEST.DEVICE == 'cpu':
            self.device = 'cpu'
        else:
            self.device = 'cuda'
        self.network = network.to(self.device) # network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constraint
        # self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

        # post-processing related parameters
        self.post_processing = False
        self.penalty_k = 0.007
        self.window_influence = 0.225
        self.lr = 0.616

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

        # save the model state dictionary only (to verify the actual model size)
        self.save_state_dict = True
        if self.save_state_dict:
            model_name = self.params.checkpoint
            torch.save(network.state_dict(), model_name.split('.pth.tar')[0] + '_state_dict.pt')

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        """
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        """

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
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors.to(self.device), search=x_dict.tensors.to(self.device))

        """ The original implementation by OSTrack
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        """

        if 'score_map' in out_dict:
            pred_score_map = out_dict['score_map']

            if not self.post_processing:
                # without "heavy" post-processing
                # add hann windows
                response = self.output_window * pred_score_map
                pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
                pred_boxes = pred_boxes.view(-1, 4)
                # Baseline: Take the mean of all pred boxes as the final result
                pred_box = (pred_boxes.mean(
                    dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                best_bbox = self.map_box_back(pred_box, resize_factor)

            else:
                #  "heavy" post-processing
                dense_bounding_box_preds = []
                size_map = out_dict['size_map']
                offset_map = out_dict['offset_map']
                s_c = np.zeros((pred_score_map.size(2), pred_score_map.size(3)), dtype=np.float)
                r_c = np.zeros((pred_score_map.size(2), pred_score_map.size(3)), dtype=np.float)

                # compute bounding box predictions for all filter response locations
                for idx in range(pred_score_map.size(2)*pred_score_map.size(3)):
                    idx = torch.from_numpy(np.array([idx]))
                    idx_y = idx // self.feat_sz
                    idx_x = idx % self.feat_sz

                    idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1).cuda()
                    size = size_map.flatten(2).gather(dim=2, index=idx)
                    offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

                    bbox = torch.cat([(idx_x.to(torch.float).cuda() + offset[:, :1]) / self.feat_sz,
                                      (idx_y.to(torch.float).cuda() + offset[:, 1:]) / self.feat_sz,
                                      size.squeeze(-1)], dim=1)
                    bbox = bbox.view(-1, 4)
                    pred_box = (bbox.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                    mapped_box = self.map_box_back(pred_box, resize_factor)

                    s_c[idx_x, idx_y] = self.change( self.sz(mapped_box[2], mapped_box[3]) / self.sz_wh(self.state[2:]) ) # size
                    r_c[idx_x, idx_y] = self.change( (self.state[2]/self.state[3]) / (mapped_box[2]/mapped_box[3]) ) # aspect-ratio
                    dense_bounding_box_preds.append(mapped_box)

                # compute the size and aspect-ration penalty
                penalty = np.exp(-(r_c * s_c - 1) * self.penalty_k)
                cls_score = pred_score_map.squeeze().cpu().data.numpy()
                pscore = penalty * cls_score

                # window penalty
                pscore = (1 - self.window_influence) * pscore + \
                         self.output_window.squeeze().cpu().data.numpy() * self.window_influence

                # get the bounding box corresponding to max value in the penalized cls response
                r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)
                idx = np.argmax(pscore.flatten(), axis=0)
                best_bbox = dense_bounding_box_preds[idx]

                # size learning rate
                lr = float(penalty[r_max, c_max] * cls_score[r_max, c_max] * self.lr)
                best_bbox[2] = (1 - lr) * self.state[2] + lr * best_bbox[2]
                best_bbox[3] = (1 - lr) * self.state[3] + lr * best_bbox[3]
        
        else:
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist() #(cx, cy, w, h) [0,1]
            best_bbox = self.map_box_back(pred_box, resize_factor)

        # get the final box result
        self.state = clip_box(best_bbox, H, W, margin=10)


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
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

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
    return MobileViTTrack
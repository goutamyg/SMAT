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
from pathlib import Path

from lib.test.tracker.data_utils_mobilevit import Preprocessor, PreprocessorX_onnx
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


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
        # print(self.device)

        # if self.device == 'cuda':
        #     print('__CUDNN VERSION:', torch.backends.cudnn.version())
        #     print('__Number CUDA Devices:', torch.cuda.device_count())
        #     print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        #     print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        
        self.network = network.to(self.device)
        self.network.eval()
        self.preprocessor = PreprocessorX_onnx()
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

        weights_path = Path(self.params.checkpoint.split('.tar')[0])

        # Paths where ONNX model and TensorRT Engine will be stored.
        self.onnx_file_path = Path(weights_path.with_suffix('.onnx'))
        self.engine_file_path = Path(weights_path.with_suffix('.trt'))

        # Specify batch size and model precision
        self.batch_size = 1
        self.precision = np.float32

        # Create logger. You can adjust the log level if needed i.e, WARN, INFO, etc.
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

        # Perform model conversion from ONNX to equivalent TensorRT Optimized Engine
        self.engine = self.get_engine(self.onnx_file_path, self.engine_file_path)

        # Create TensorRT Context
        self.context = self.engine.create_execution_context()

        # Allocate memory
        self.allocate_memory(self.batch_size)

        
    def get_engine(self, onnx_file_path, engine_file_path=""):
        """
          Attempts to load a serialized engine if available, otherwise builds a new TensorRT optimized engine and saves it.
        """    

        def build_engine():
            """
                Takes an ONNX file and creates a TensorRT engine to run inference with
            """
            with trt.Builder(self.TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                    ) as network, builder.create_builder_config() as config, trt.OnnxParser(network, self.TRT_LOGGER
                        ) as parser, trt.Runtime(self.TRT_LOGGER) as runtime:

                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8*1024*1024*1024) #8GB
                builder.max_batch_size = 1

                # Parse model file
                print("Loading ONNX file from path {}...".format(onnx_file_path))
                with open(onnx_file_path, "rb") as model:
                    print("Beginning ONNX file parsing")
                    if not parser.parse(model.read()):
                        print("ERROR: Failed to parse the ONNX file.")
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        return None
                            
                # print(network.get_input(0).shape)
                # print(network.get_input(1).shape)
                print("Completed parsing of ONNX file")
                print("Building an engine from file {}; this may take a while...".format(onnx_file_path))

                plan = builder.build_serialized_network(network, config)
                engine = runtime.deserialize_cuda_engine(plan)

                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
                return engine

        if os.path.exists(engine_file_path):
            # Load plugins to avoid potential error
            trt.init_libnvinfer_plugins(None,'')
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine()

    def allocate_memory(self, batch_size):
        """
        Method that allocates memory for input and output bindings as well GPU buffers 
        """
        print("Allocating memory")

        # Need to set both input and output precisions to appropriate precision
        self.z_input = np.empty([batch_size, 3, 128, 128], dtype=self.precision)
        self.x_input = np.empty([batch_size, 3, 256, 256], dtype=self.precision)

        # Allocate GPU memory for input tensors
        self.d_z_input = cuda.mem_alloc(1 * self.z_input.nbytes)
        self.d_x_input = cuda.mem_alloc(1 * self.x_input.nbytes)

        # Output buffers 
        self.cls = np.empty([batch_size, 1, 4], dtype=self.precision)
        self.reg = np.empty([batch_size, 1, 16, 16], dtype=self.precision)
        self.size_map = np.empty([batch_size, 2, 16, 16], dtype=self.precision)
        self.offset_map = np.empty([batch_size, 2, 16, 16], dtype=self.precision) 

        # Allocate GPU buffers for output tensors
        self.d_cls = cuda.mem_alloc(1 * self.cls.nbytes)
        self.d_reg = cuda.mem_alloc(1 * self.reg.nbytes)
        self.d_size_map = cuda.mem_alloc(1 * self.size_map.nbytes)
        self.d_offset_map = cuda.mem_alloc(1 * self.offset_map.nbytes)

        self.bindings = [int(self.d_z_input),
                        int(self.d_x_input),
                        int(self.d_cls),
                        int(self.d_reg),
                        int(self.d_size_map),
                        int(self.d_offset_map)]
                    
        self.stream = cuda.Stream()
    
    def infer(self, batch):

        # Ensure input arrays are contiguous to enable copying to CUDA devices
        contiguous_z = np.ascontiguousarray(batch['z'])
        contiguous_x = np.ascontiguousarray(batch['x'])

        # transfer input data to device
        cuda.memcpy_htod_async(self.d_z_input, contiguous_z, self.stream)
        cuda.memcpy_htod_async(self.d_x_input, contiguous_x, self.stream)

        # execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)

        # transfer predictions back to cpu
        cuda.memcpy_dtoh_async(self.cls, self.d_cls, self.stream)
        cuda.memcpy_dtoh_async(self.reg, self.d_reg, self.stream)
        cuda.memcpy_dtoh_async(self.size_map, self.d_size_map, self.stream)
        cuda.memcpy_dtoh_async(self.offset_map, self.d_offset_map, self.stream)

        # syncronize threads
        self.stream.synchronize()
        
        return [self.cls, self.reg, self.size_map, self.offset_map]
        
    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        
        self.z_dict = template

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

            # Model inputs
            tensorrt_inputs = {'x': x_dict[0].astype(self.precision),
                            'z': self.z_dict[0].astype(self.precision)}
 
            # Run inference on input data
            out_dict = self.infer(tensorrt_inputs)
        
        # Post-process predictions to evaluate tracker performance
        pred_score_map = out_dict[1]

        # add hann windows
        response = self.output_window * torch.from_numpy(pred_score_map).to(self.device)
        # response = pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, torch.from_numpy(out_dict[2]).to(self.device),
                                                     torch.from_numpy(out_dict[3]).to(self.device))
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

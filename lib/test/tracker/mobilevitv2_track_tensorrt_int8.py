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

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import create_config, engine_from_network, network_from_onnx_path, save_engine, engine_from_bytes, Calibrator
from polygraphy.logger import G_LOGGER
import onnx
import pickle


class MobileViTv2Track(BaseTracker):

    def __init__(self, params, dataset_name):
        super(MobileViTv2Track, self).__init__(params)
        network = build_mobilevitv2_track(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        
        onnx_checkpoint = self.params.checkpoint.split('.pth')[0] + '.onnx'
        self.onnx_model = onnx.load(onnx_checkpoint)
        onnx.checker.check_model(self.onnx_model, True)
        # Print a human readable representation of the graph
        # print(onnx.helper.printable_graph(self.onnx_model.graph))

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

        # path where the pytocrh model state dictionary is stored
        weights_path = Path(self.params.checkpoint.split('.tar')[0])

        # Path where calibration data file is located
        self.calibration_data_path = self.params.checkpoint.split('.pth')[0]+'_calib_data.pkl'

        # Paths where ONNX model and TensorRT Engine will be stored.
        self.onnx_file_path = Path(weights_path.with_suffix('.onnx'))
        self.engine_file_path = Path(weights_path.with_suffix('.polygraphy_int8.engine'))

        # Specify model precision
        self.precision = np.float32

        # Create logger. You can adjust the log level if needed i.e, WARN, INFO, etc.
        self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
 
        # Polygraphy Logger Verbosity
        G_LOGGER.verbosity(G_LOGGER.INFO)

        # Initialize the calibrator i.e., IInts8EntropyCalibrator2
        self.calibrator = Calibrator(data_loader=self.data_loader(), cache="int8-calib.cache")

        # Perform model conversion from ONNX to equivalent TensorRT Optimized Engine using polgraphy
        self.engine = self.get_engine(str(self.onnx_file_path), str(self.engine_file_path), self.calibrator)

    def store_inputs(self, model_inputs):
        """
        Method that stores the template and search region for each frame of a sequence
        and writes it to calibration file (pickle format). 
        Run this once using the satndard FP32 inference to collect input samples for 
        each sequence.  
        @param model_inputs: The search_region 'x' and the template 'z'
        """

        # Initialize the list
        inputs = []

        # Specify the file path
        file_path = self.calibration_data_path

        new_input = model_inputs

        # Append new data to the list
        inputs.append(new_input)

        # Write the updated list to the file
        with open(file_path, 'ab') as file:
            pickle.dump(inputs, file)


    def data_loader(self):
        """
        Method that loads input samples from a pickle file and yields/returns the model
        input dictionary, one input at a time. 
        """
        # This variable can be adjusted as per need. Impacts the accuracy of the engine
        num_of_samples = 20000
        print(f"Calibrating model using {num_of_samples} samples")

        with open(self.calibration_data_path, 'rb') as file:
            try:
                for _ in range(num_of_samples):

                    data = pickle.load(file)
                    # If data is a list with a single tuple, extract the tuple
                    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], tuple):
                        yield {'z': data[0][0],
                                'x': data[0][1]}
                    elif isinstance(data, tuple):
                        yield {'z': data[0],
                                'x': data[1]}
            except EOFError:
                pass 


    def get_engine(self, onnx_file_path, engine_file_path, calibrator):
        """
          Attempts to load a serialized engine if available, otherwise builds a new 
          TensorRT optimized engine and saves it.
          @param onnx_file_path: The path where the onnx model is saved.
          @param engine_file_path: The path where the optimized INT8 tensorRT engine is saved
          @param calibrator: Supplies calibration data to TensorRT to calibrate the network for INT8 inference.
          returns the optimized (calibrated) tensorrt engine
        """    

        if os.path.exists(engine_file_path):
            # Load a pre-built serialized engine from a file, then deserialize it into a TensorRT engine
            print("Reading engine from file {}".format(engine_file_path))
            loaded_engine = engine_from_bytes(bytes_from_path(engine_file_path))
            return loaded_engine
        else:
            # Build a new engine if it doesn't exist
            print("Reading ONNX model from {}".format(onnx_file_path))
            builder, network, parser = network_from_onnx_path(onnx_file_path)

            # Create a TensorRT IBuilderConfig so that we can build the engine with INT8 enabled.
            config = create_config(builder, network, int8=True, calibrator=calibrator, quantization_flags=[trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION])
            
            with builder, network, parser, config:
                engine = engine_from_network((builder, network), config)

            # To reuse the engine elsewhere, we can serialize it and save it to a file.
            save_engine(engine, path=engine_file_path)
            print("Engine successfully created and saved.")
            return engine

          
    def predict(self, engine, inputs):
        """
        Method that allocates buffer/memory on the GPU for the inputs and outputs,
        asynchronously performs inference, and returns the calculation to the CPU for 
        post-preprocessing.
        @param engine: The tensorrt engine required for inference
        @param inputs: The input tuple fed to the model
        returns The output/predeictions; a tuple containing four numpy arrays
        """

        with engine.create_execution_context() as context:

            # Allocate host and device buffers
            bindings = []
            for i, binding in enumerate(engine):
                
                size = trt.volume(context.get_binding_shape(i)) # Retrieve shape
                dtype = trt.nptype(engine.get_binding_dtype(binding)) # Retrieve datatype
                name = engine.get_binding_name(i) # Retrieve name
                
                if name == 'z':
                    input_buffer_z = np.ascontiguousarray(inputs[0])
                    input_memory_z = cuda.mem_alloc(inputs[0].nbytes) # Allocate GPU memory
                    bindings.append(int(input_memory_z))
                elif name == 'x':
                    input_buffer_x = np.ascontiguousarray(inputs[1])
                    input_memory_x = cuda.mem_alloc(inputs[1].nbytes)
                    bindings.append(int(input_memory_x))
                elif name == 'cls':
                    output_buffer_cls = cuda.pagelocked_empty(size, dtype)
                    output_memory_cls = cuda.mem_alloc(output_buffer_cls.nbytes)
                    bindings.append(int(output_memory_cls))
                elif name == 'reg':
                    output_buffer_reg = cuda.pagelocked_empty(size, dtype)
                    output_memory_reg = cuda.mem_alloc(output_buffer_reg.nbytes)
                    bindings.append(int(output_memory_reg))
                elif name == 'size_map':
                    output_buffer_size_map = cuda.pagelocked_empty(size, dtype)
                    output_memory_size_map = cuda.mem_alloc(output_buffer_size_map.nbytes)
                    bindings.append(int(output_memory_size_map))
                else: # offset_map
                    output_buffer_offset_map = cuda.pagelocked_empty(size, dtype)
                    output_memory_offset_map = cuda.mem_alloc(output_buffer_offset_map.nbytes)
                    bindings.append(int(output_memory_offset_map))

            stream = cuda.Stream()
            
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory_z, input_buffer_z, stream)
            cuda.memcpy_htod_async(input_memory_x, input_buffer_x, stream)
            
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer_cls, output_memory_cls, stream)
            cuda.memcpy_dtoh_async(output_buffer_reg, output_memory_reg, stream)
            cuda.memcpy_dtoh_async(output_buffer_size_map, output_memory_size_map, stream)
            cuda.memcpy_dtoh_async(output_buffer_offset_map, output_memory_offset_map, stream)
            
            # Synchronize the stream
            stream.synchronize()

            # Clean/Release GPU memory after each inference
            input_memory_z.free()
            input_memory_x.free()
            output_memory_cls.free()
            output_memory_reg.free()
            output_memory_size_map.free()
            output_memory_offset_map.free()

            # err = cuda.mem_get_info()
            # print(f"Available GPU memory: {err[0] / (1024 ** 2)} MB")

            output = (output_buffer_cls, output_buffer_reg.reshape(1,1,16,16), output_buffer_size_map.reshape(1,2,16,16), output_buffer_offset_map.reshape(1,2,16,16))
        
        return output

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
                    
        x_dict = search

        # calib_inputs = [
        #     self.z_dict[0].astype(np.float32),
        #     x_dict[0].astype(np.float32)
        #     ]
        # # Store inputs
        # self.store_inputs(calib_inputs)
                
        # Model inputs
        tensorrt_inputs = (self.z_dict[0].astype(self.precision),
                            x_dict[0].astype(self.precision))

        # Run inference
        out_dict = self.predict(self.engine, tensorrt_inputs)
        
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
        

    def cleanup(self):
        self.d_x_input.free()
        self.d_z_input.free()
        self.d_cls.free()
        self.d_reg.free()
        self.d_size_map.free()
        self.d_offset_map.free()

        print("Memory buffers clean-up successful!")

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
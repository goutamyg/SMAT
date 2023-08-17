import numpy as np
import os
from .utils import compute_IoU as compute_IoU_rect
from .utils import xywh_to_x1y1x2y2
from shapely.geometry import Polygon, box


class DatasetBuilder():

    def __init__(self, name, base_path):
        self.dataset_name = name
        if base_path is None:
            base_path = 'got10k_val_anno/'
        self.base_path = base_path
        self.get_dataset_path()
        self.find_groundtruth_filename()
        self.create_seq_info()
        self.sr_threshold = 0.5

    def get_dataset_path(self):
        if self. dataset_name == 'nfs':
            self.dataset_path = os.path.join(self.base_path, 'annotations_only/NfS30')
            self.dataset_fullname = "NfS30"
        elif self.dataset_name == 'lasot':
            self.dataset_path = os.path.join(self.base_path, 'annotations_only/LaSOTTest')
            self.dataset_fullname = 'LaSOT-Test'
        elif self.dataset_name == "trackingnet":
            self.dataset_path = os.path.join(self.base_path, 'TrackingNet/anno')
            self.dataset_fullname = 'TrackingNet-Test'
        elif self.dataset_name == 'otb100':
            self.dataset_path = os.path.join(self.base_path, 'annotations_only/OTB100')
            self.dataset_fullname = 'OTB100'
        elif self.dataset_name == 'tc128':
            self.dataset_path = os.path.join(self.base_path, 'annotations_only/TC128')
            self.dataset_fullname = 'TC128'
        elif self.dataset_name == 'vot2018':
            self.dataset_path = os.path.join(self.base_path, 'annotations_only/VOT2018')
            self.dataset_fullname = 'VOT2018'
        elif self.dataset_name == 'got10k':
            self.dataset_path = os.path.join(self.base_path)
            self.dataset_fullname = 'GOT10k'
        else:
            raise ValueError("unknown dataset name!")

    def find_groundtruth_filename(self):
        if self.dataset_name in ['lasot', 'nfs', 'tc128', 'vot2018', 'got10k']:
            self.gt_filename = 'groundtruth.txt'
        elif self.dataset_name == 'otb100':
            self.gt_filename = 'groundtruth_rect.txt'
        else:
            raise NameError("Dataset name not found")

    def create_seq_info(self):
        seq_names = sorted(os.listdir(self.dataset_path))
        self.seqs_info = {}
        if self.dataset_name == 'vot2018':
            anno_format = 'polygonal'
        elif self.dataset_name in ['otb100', 'tc128', 'lasot', 'nfs', 'got10k']:
            anno_format = 'rectangular'
        else:
            raise NameError("Unknown dataset name {}".format(self.dataset_name))

        for id, name in enumerate(seq_names):
            seq_gt_path = os.path.join(self.dataset_path, name, self.gt_filename)
            seq_gt_delimiter = find_delimiter(seq_gt_path)
            self.seqs_info[name] = {'gt_path': seq_gt_path,
                                    'delimiter': seq_gt_delimiter,
                                    'anno_format': anno_format}


    def compute_test_results(self, results_path):
        self.results_path = results_path
        for seq in self.seqs_info:
            # print(seq)
            seq_gt_info = self.seqs_info[seq]
            gt_anno_data = np.loadtxt(seq_gt_info['gt_path'], delimiter=seq_gt_info['delimiter'])

            delimiter_seq = find_delimiter(os.path.join(results_path, self.dataset_name, seq+'.txt'))
            tracker_pred = np.loadtxt(os.path.join(results_path, self.dataset_name, seq+'.txt'), delimiter=delimiter_seq)

            aor_ = np.mean(self.compute_aor(gt_anno_data, tracker_pred))
            fr_ = 1 - np.mean(self.compute_sr(gt_anno_data, tracker_pred))

            self.seqs_info[seq]["aor"] = aor_
            self.seqs_info[seq]["fr"] = fr_

    def compute_aor(self, gt, pred):
        if gt.shape[0] > pred.shape[0]:
            print("Groundtruth has more annotations than predicted boxes. The situation is handled by repeating "
                  "last bounding box n-times till their lengths match")
            for i in range(gt.shape[0] - pred.shape[0]):
                pred = np.concatenate((pred, pred[-1, :].reshape(1, -1)), axis=0)
        assert gt.shape[0] == pred.shape[0]

        if gt.shape[1] == 4:
            per_frame_iou_ = [compute_IoU_rect(xywh_to_x1y1x2y2(gt[i, :]), xywh_to_x1y1x2y2(pred[i, :]))
                                                                        for i in range(0, gt.shape[0])]
            return per_frame_iou_
        elif gt.shape[1] == 8:
            per_frame_iou_ = poly_iou(gt, pred)
            return per_frame_iou_
        else:
            raise AssertionError("Can not handle annotation with {} elements".format(gt.shape[1]))


    def compute_sr(self, gt, pred):
        if gt.shape[0] > pred.shape[0]:
            print("Groundtruth has more annotations than predicted boxes. The situation is handled by repeating "
                  "last bounding box n-times till their lengths match")
            for i in range(gt.shape[0] - pred.shape[0]):
                pred = np.concatenate((pred, pred[-1, :].reshape(1, -1)), axis=0)
        assert gt.shape[0] == pred.shape[0]
        if gt.shape[1] == 4:
            per_frame_sr_ = [int(compute_IoU_rect(xywh_to_x1y1x2y2(gt[i, :]),
                                 xywh_to_x1y1x2y2(pred[i, :])) > self.sr_threshold) for i in range(0, gt.shape[0])]
            return per_frame_sr_
        elif gt.shape[1] == 8:
            per_frame_iou_ = poly_iou(gt, pred)
            per_frame_sr_ = [int(i > self.sr_threshold) for i in list(per_frame_iou_)]
            return per_frame_sr_
        else:
            raise AssertionError("Can not handle annotation with {} elements".format(gt.shape[1]))


    def summarize_tracker_results(self):
        aor = [self.seqs_info[seq]['aor'] for seq in self.seqs_info]
        fr = [self.seqs_info[seq]['fr'] for seq in self.seqs_info]

        # print("Average Overlap Ratio for {} dataset = {}".format(self.dataset_fullname, sum(aor)/len(aor)))
        # print("Failure Rate for {} dataset = {}".format(self.dataset_fullname, sum(fr)/len(fr)))
        
        return sum(aor)/len(aor), 1 - (sum(fr)/len(fr))

################################################## utility functions ##################################################

def find_delimiter(file_path):
    # check the delimiter
    try:
        _ = np.loadtxt(file_path, delimiter=',')
        delimiter = ','
        return delimiter
    except:
        pass
    try:
        _ = np.loadtxt(file_path, delimiter=' ')
        delimiter = ' '
        return delimiter
    except:
        pass
    try:
        _ = np.loadtxt(file_path, delimiter='\t')
        delimiter = '\t'
        return delimiter
    except NotImplementedError:
        raise RuntimeError("Unable to find the right delimiter for {}".format(file_path))


def compute_IoU_polygonal(gt, pred):

    polygon1 = Polygon(gt)
    polygon2 = Polygon(pred)
    intersect = polygon1.intersection(polygon2).area
    union = polygon1.union(polygon2).area
    iou = intersect / union
    return iou

def poly_iou(polys1, polys2, bound=None):
    r"""Intersection over union of polygons.
    Args:
        polys1 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
        polys2 (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """
    assert polys1.ndim in [1, 2]
    if polys1.ndim == 1:
        polys1 = np.array([polys1])
        polys2 = np.array([polys2])
    assert len(polys1) == len(polys2)

    polys1 = _to_polygon(polys1)
    polys2 = _to_polygon(polys2)
    if bound is not None:
        bound = box(0, 0, bound[0], bound[1])
        polys1 = [p.intersection(bound) for p in polys1]
        polys2 = [p.intersection(bound) for p in polys2]

    eps = np.finfo(float).eps
    ious = []
    for poly1, poly2 in zip(polys1, polys2):
        area_inter = poly1.intersection(poly2).area
        area_union = poly1.union(poly2).area
        ious.append(area_inter / (area_union + eps))
    ious = np.clip(ious, 0.0, 1.0)

    return ious


def _to_polygon(polys):

    r"""Convert 4 or 8 dimensional array to Polygons
    Args:
        polys (numpy.ndarray): An N x 4 numpy array, each line represent a rectangle
            (left, top, width, height); or an N x 8 numpy array, each line represent
            the coordinates (x1, y1, x2, y2, x3, y3, x4, y4) of 4 corners.
    """

    def to_polygon(x):
        assert len(x) in [4, 8]
        if len(x) == 4:
            return box(x[0], x[1], x[0] + x[2], x[1] + x[3])
        elif len(x) == 8:
            return Polygon([(x[2 * i], x[2 * i + 1]) for i in range(4)])

    if polys.ndim == 1:
        return to_polygon(polys)
    else:
        return [to_polygon(t) for t in polys]

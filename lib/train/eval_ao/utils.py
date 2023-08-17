import numpy as np

def get_groundtruth_path(dataset_name):

    if dataset_name == 'nfs':
        gt_dir = '/home/goutam/Datasets/annotations_only/NfS30'
    elif dataset_name == 'lasot':
        gt_dir = '/home/goutam/Datasets/annotations_only/LaSOTTest'
    elif dataset_name == "trackingnet":
        gt_dir = '/home/goutam/Datasets/TrackingNet/anno'
    elif dataset_name == 'otb100':
        gt_dir = '/home/goutam/Datasets/TrackingNet/anno'
    else:
        raise ValueError("unknown dataset name!")

    return gt_dir


def compute_IoU(rect1, rect2):
    # overlap

    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2 - x1) * (y2 - y1)

    target_a = (tx2 - tx1) * (ty2 - ty1)

    inter = ww * hh
    overlap = inter / (area + target_a - inter)

    return overlap


def xywh_to_x1y1x2y2(box):
    return np.array([box[0], box[1], box[0] + box[2], box[1]+box[3]])
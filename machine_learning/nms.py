import numpy as np


def nms(pred_bboxes, iou_thres=0.1):
    '''
    Param:
        pred_bboxes: np.array(N, 5)
            (xmin, ymin, xmax, ymax confi)
    Retrun:
        pred_bboxes: np.array(N', 5)
    '''
    valid_indexs = (pred_bboxes[:, 2] > pred_bboxes[:, 0]) * (pred_bboxes[:, 3] > pred_bboxes[:, 1])
    pred_bboxes = pred_bboxes[valid_indexs]
    # areas.shape = (N1, )
    areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
    # xmins.shape = (N1, N1)
    xmins = np.maximum(pred_bboxes[None, :, 0], pred_bboxes[:, None, 0])
    xmaxs = np.minimum(pred_bboxes[None, :, 2], pred_bboxes[:, None, 2])
    ymins = np.maximum(pred_bboxes[None, :, 1], pred_bboxes[:, None, 1])
    ymaxs = np.minimum(pred_bboxes[None, :, 3], pred_bboxes[:, None, 3])
    inters = np.clip(xmaxs - xmins, 0, np.inf) * np.clip(ymaxs - ymins, 0, np.inf)
    # ious.shape = (N1, N1)
    ious = inters / areas[:, None]
    # nms
    print(ious)
    res = []
    orders = np.argsort(pred_bboxes[:, 4])[::-1]
    for i in orders:
        if len(res) > 0 and np.max(ious[i][res]) >= iou_thres:
            continue
        else:
            res.append(i)
    return pred_bboxes[res]


def faster_nms(pred_bboxes, iou_thres=0.1):
    '''
    Param:
        pred_bboxes: np.array(N, 5)
            (xmin, ymin, xmax, ymax confi)
    Retrun:
        pred_bboxes: np.array(N', 5)
    '''
    valid_indexs = (pred_bboxes[:, 2] > pred_bboxes[:, 0]) * (pred_bboxes[:, 3] > pred_bboxes[:, 1])
    pred_bboxes = pred_bboxes[valid_indexs]
    orders = np.argsort(pred_bboxes[:, 4])[::-1]
    pred_bboxes = pred_bboxes[orders]
    # areas.shape = (N1, )
    areas = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
    # xmins.shape = (N1, N1)
    xmins = np.maximum(pred_bboxes[None, :, 0], pred_bboxes[:, None, 0])
    xmaxs = np.minimum(pred_bboxes[None, :, 2], pred_bboxes[:, None, 2])
    ymins = np.maximum(pred_bboxes[None, :, 1], pred_bboxes[:, None, 1])
    ymaxs = np.minimum(pred_bboxes[None, :, 3], pred_bboxes[:, None, 3])
    inters = np.clip(xmaxs - xmins, 0, np.inf) * np.clip(ymaxs - ymins, 0, np.inf)
    # ious.shape = (N1, N1)
    ious = inters / areas[:, None]
    # nms
    lower_tri = np.arange(ious.shape[0])[None, :] < np.arange(ious.shape[0])[:, None]
    ious[~lower_tri] = 0
    selected_indexs = np.max(ious, axis=1) < iou_thres
    return pred_bboxes[selected_indexs]


def main():
    # (xmin, ymin, xmax, ymax, cofi)
    pred_bboxes = np.array(
        [
            [10, 10, 20, 20, 0.2],
            [15, 15, 25, 25, 0.8],
            [20, 20, 25, 25, 0.3],
            [100, 90, 120, 200, 0.1],
            [10, 20, 10, 20, 0.4],
        ]
    )
    print(pred_bboxes)
    pred_bboxes = faster_nms(pred_bboxes)
    print(pred_bboxes)


if __name__ == '__main__':
    main()

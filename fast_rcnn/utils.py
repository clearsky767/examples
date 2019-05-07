# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import torch
import numpy as np

def _flip_box(boxes, width):
  boxes = boxes.clone()
  oldx1 = boxes[:, 0].clone()
  oldx2 = boxes[:, 2].clone()
  boxes[:, 0] = width - oldx2 - 1
  boxes[:, 2] = width - oldx1 - 1
  return boxes

def bbox_transform(ex_rois, gt_rois):
    """
    gt_rois box is l,r,u,d mode type numpy shape (n,4)
    ex_rois box is l,r,u,d mode type numpy shape (n,4)
    targets type numpy shape (n,4)
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    print("in bbox_transform")
    print(ex_rois.shape)
    print(ex_rois[0:2])
    print(gt_rois.shape)
    print(gt_rois[0:2])
    print(targets.shape)
    print(targets[0:2])
    return targets

def bbox_transform_inv(boxes, deltas):
    """
    boxes box is l,r,u,d mode type numpy shape (n,4)
    deltas box is dx,dy,dw,dh type numpy shape (n,4)
    pred_boxes box is l,r,u,d mode type numpy shape (n,4)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h
    print("in bbox_transform_inv")
    print(boxes.shape)
    print(boxes[0:2])
    print(deltas.shape)
    print(deltas[0:2])
    return pred_boxes

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    boxes box is l,r,u,d mode ,im_shape type tuple shape (w,h)
    """
    x = boxes[:, 0::4]
    y = np.minimum(x, im_shape[1] - 1)
    z = np.maximum(y, 0)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def filter_boxes(boxes, min_size):
    """
    boxes box is l,r,u,d mode
    Remove all boxes with any side smaller than min_size.
    boxes type numpy shape (n,4)
    keep type numpy shape (k,)
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def bbox_overlaps(a, bb):
  """
  a,bb box is l,r,u,d mode,not x,y,w,h mode
  a type torch shape (n,4) ,bb type torch shape (m,4) ,oo typetorch shape (n,m)
  """
  a = to_tensor(a)
  bb = to_tensor(bb)
  oo = []
  for b in bb:
    x1 = a.select(1,0).clone()
    x1[x1.lt(b[0])] = b[0]
    y1 = a.select(1,1).clone()
    y1[y1.lt(b[1])] = b[1]
    x2 = a.select(1,2).clone()
    x2[x2.gt(b[2])] = b[2]
    y2 = a.select(1,3).clone()
    y2[y2.gt(b[3])] = b[3]

    w = x2-x1+1
    h = y2-y1+1
    inter = torch.mul(w,h).float()
    aarea = torch.mul((a.select(1,2)-a.select(1,0)+1), (a.select(1,3)-a.select(1,1)+1)).float()
    barea = (b[2]-b[0]+1) * (b[3]-b[1]+1)
    
    # intersection over union overlap
    o = torch.div(inter , (aarea+barea-inter))
    # set invalid entries to 0 overlap
    o[w.lt(0)] = 0
    o[h.lt(0)] = 0
    oo += [o]
  return torch.cat([o.view(-1,1) for o in oo],1)

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif torch.is_tensor(x):
        return x
    elif isinstance(x, tuple):
        t = []
        for i in x:
            t.append(to_tensor(i))
        return t
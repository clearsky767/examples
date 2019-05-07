import torch
import torch.nn as nn
import numpy as np

from utils import bbox_transform, bbox_transform_inv, clip_boxes, bbox_overlaps

from utils import to_tensor

# should handle multiple scales, how?
class FasterRCNN(nn.Module):

  def __init__(self,
          features, pooler,
          classifier, rpn,
          batch_size=128, fg_fraction=0.25,
          fg_threshold=0.5, bg_threshold=None,
          num_classes=21):
    super(FasterRCNN, self).__init__()
    self.features = features
    self.roi_pooling = pooler
    self.rpn = rpn
    self.classifier = classifier
    
    self.batch_size = batch_size
    self.fg_fraction = fg_fraction
    self.fg_threshold = fg_threshold
    if bg_threshold is None:
        bg_threshold = (0, 0.5)
    self.bg_threshold = bg_threshold
    self._num_classes = num_classes

  # should it support batched images ?
  def forward(self, x):
    print("in FasterRCNN forward")
    #if self.training is True:
    if isinstance(x, tuple):
      im, gt = x
    else:
      im = x
      gt = None

    assert im.size(0) == 1, 'only single element batches supported'

    feats = self.features(to_tensor(im))

    roi_boxes, rpn_prob, rpn_loss = self.rpn(im, feats, gt)

    #if self.training is True:
    if gt is not None:
      # append gt boxes and sample fg / bg boxes
      # proposal_target-layer.py
      all_rois, frcnn_labels, roi_boxes, frcnn_bbox_targets = self.frcnn_targets(roi_boxes, im, gt)

    # r-cnn
    regions = self.roi_pooling(feats, roi_boxes)
    scores, bbox_pred = self.classifier(regions)

    boxes = self.bbox_reg(roi_boxes, bbox_pred, im)

    # apply cls + bbox reg loss here
    #if self.training is True:
    if gt is not None:
      frcnn_loss = self.frcnn_loss(scores, bbox_pred, frcnn_labels, frcnn_bbox_targets)
      loss = frcnn_loss + rpn_loss
      return loss, scores, boxes

    return scores, boxes

  #scores shape (num_clssses,) bbox_pred (num_classes*4,)
  def frcnn_loss(self, scores, bbox_pred, labels, bbox_targets):
    cls_crit = nn.CrossEntropyLoss()
    cls_loss = cls_crit(scores, labels)

    reg_crit = nn.SmoothL1Loss()
    reg_loss = reg_crit(bbox_pred, bbox_targets)

    loss = cls_loss + reg_loss
    return loss

  def frcnn_targets(self, all_rois, im, gt):
    print("in frcnn_targets")
    print(all_rois.shape)
    print(all_rois[0:2])
    print(im.shape)
    print(im[0:2])
    all_rois = all_rois.data.numpy()
    gt_boxes = gt['boxes'].numpy()
    gt_labels = np.array(gt['gt_classes'])
    print(gt_boxes.shape)
    print(gt_boxes[0:2])
    print(gt_labels.shape)
    print(gt_labels[0:2])
    #zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
    #all_rois = np.vstack(
    #    (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
    #)
    all_rois = np.vstack((all_rois, gt_boxes))
    zeros = np.zeros((all_rois.shape[0], 1), dtype=all_rois.dtype)
    all_rois = np.hstack((zeros, all_rois))
    print(all_rois.shape)
    print(all_rois[0:2])
    num_images = 1
    rois_per_image = self.batch_size / num_images
    fg_rois_per_image = int(np.round(self.fg_fraction * rois_per_image))
    print(fg_rois_per_image)
    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, bbox_targets = _sample_rois(self,
        all_rois, gt_boxes, gt_labels, fg_rois_per_image,
        rois_per_image, self._num_classes)
    
    return to_tensor((all_rois, labels, rois, bbox_targets))

  def bbox_reg(self, boxes, box_deltas, im):
    print("in bbox_reg")
    print(boxes.shape)
    print(boxes[0:2])
    print(box_deltas.shape)
    print(box_deltas[0:2])
    print(im.shape)
    print(im[0:2])
    boxes = boxes.data[:,1:].numpy()
    print(boxes.shape)
    print(boxes[0:2])
    box_deltas = box_deltas.data.numpy()
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    print(pred_boxes.shape)
    print(pred_boxes[0:2])
    pred_boxes = clip_boxes(pred_boxes, im.size()[-2:])
    print(pred_boxes.shape)
    print(pred_boxes[0:2])
    return to_tensor(pred_boxes)
    

    #bbox_target_data type numpy.ndarray shape (128,5) num_classes 21
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    print("in _get_bbox_regression_labels")
    print(bbox_target_data.shape)
    print(type(bbox_target_data))
    print(bbox_target_data[0:2])
    print(num_classes)
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    #bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    print(inds.shape)
    print(inds)
    for ind in inds:
        cls = clss[ind]
        start = cls*4
        end = start + 4
        bbox_targets[ind, int(start):int(end)] = bbox_target_data[ind, 1:]
    return bbox_targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    print("in _compute_targets")
    print(ex_rois.shape)
    print(ex_rois[0:2])
    print(gt_rois.shape)
    print(gt_rois[0:2])
    print(targets.shape)
    print(targets[0:2])
    if False: #cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(self, all_rois, gt_boxes, gt_labels, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    print("in _sample_rois")
    print(overlaps.shape)
    print(overlaps.shape[0:2])
    overlaps = overlaps.numpy()
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    #labels = gt_boxes[gt_assignment, 4]
    labels = gt_labels[gt_assignment]
    print(gt_assignment)
    print(max_overlaps)
    print(labels)
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= self.fg_threshold)[0]
    print(fg_inds)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    print(fg_rois_per_this_image)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < self.bg_threshold[1]) &
                       (max_overlaps >= self.bg_threshold[0]))[0]
    print(bg_inds)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    print(keep_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    print(labels)
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]
    print(rois[0:2])

    bbox_target_data = _compute_targets(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)
    bbox_targets = _get_bbox_regression_labels(bbox_target_data, num_classes)
    print(bbox_target_data.shape)
    print(bbox_target_data[0:2])
    print(bbox_targets.shape)
    print(bbox_targets[0:2])
    print("_sample_rois end")
    return labels, rois, bbox_targets
import torch
import numpy as np

def rpn_get_anchors(im,anchors):
    height, width = im.size()[-2:]
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * 16
    shift_y = np.arange(0, height) * 16
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = 9
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    return all_anchors
"""
main.py: Code to calculate box weights. 
Authors       : svp
"""

import numpy as np
np.set_printoptions(suppress=True)

def compute_box_weights(hist):
    
    hist = np.float32(hist)
    hist[0] /= 64.0
    hist[1] /= 16.0
    hist[2] /= 4.0
    hist_copy = hist.copy()
    num_channels = 4
    z_weights = np.sum(hist[:, 0:num_channels-1], axis=1)
    z_weights = np.min(z_weights) / np.float32(z_weights)
    z_weights = np.expand_dims(z_weights, axis=1)
    z_weights = np.repeat(z_weights, num_channels, axis=1)
    
    b_weights = np.max(hist[:, 0:num_channels-1], axis=1)
    b_weights = np.expand_dims(b_weights, axis=1)
    b_weights = np.repeat(b_weights, num_channels, axis=1)
    b_weights[:, 0:num_channels-1] = b_weights[:, 0:num_channels-1] / np.float32(hist[:, 0:num_channels-1])

    weights = b_weights*z_weights*10
    weights[:, num_channels-1] = z_weights[:, 0]
    hist_copy[:, num_channels-1] = z_weights[:, 0]

    for h, _ in enumerate(hist_copy):
        hist_copy[h, 0:num_channels-1] = (hist_copy[h, 0:num_channels-1] / np.max(hist_copy[h, 0:num_channels-1])) * hist_copy[h, num_channels-1] * 10

    loss_weights = z_weights[:, 0]
    loss_weights = loss_weights[:, np.newaxis]
    loss_weights = np.repeat(loss_weights, 4, axis=1)
    loss_weights[:, 0:3] *= 10
    return loss_weights

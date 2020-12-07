import numpy as np
from functools import reduce
from utils import vis_data

def filtering(depth, config, mask=None, num_iter=None):
    vis_depth = depth.copy()

    for i in range(num_iter):
        if isinstance(config["filter_size"], list):
            window_size = config["filter_size"][i]
        else:
            window_size = config["filter_size"]
        u_over, b_over, l_over, r_over = vis_disp_discontinuity(vis_depth, config['disp_threshold'], mask=mask)

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter(vis_depth, config, discontinuity_map=discontinuity_map, mask=mask, window_size=window_size)
    return vis_depth


def vis_disp_discontinuity(depth, disp_threshold, vis_diff=False, mask=None):
    disp = 1./depth
    # something like 1st derivative
    # next pixel - curr in vert dir from 0 to n
    u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
    # curr pixel - next in vert dir from 0 to n
    b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
    # next pixel - curr in horiz dir from 0 to n
    l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
    # curr pixel - next in horiz dir from 0 to n
    r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
    if mask is not None:
        u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
        b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
        l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
        r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
        u_diff = u_diff * u_mask
        b_diff = b_diff * b_mask
        l_diff = l_diff * l_mask
        r_diff = r_diff * r_mask
    u_over = (np.abs(u_diff) > disp_threshold).astype(np.float32)
    b_over = (np.abs(b_diff) > disp_threshold).astype(np.float32)
    l_over = (np.abs(l_diff) > disp_threshold).astype(np.float32)
    r_over = (np.abs(r_diff) > disp_threshold).astype(np.float32)

    #     disp = depth
    #     u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
    #     b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
    #     l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
    #     r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
    #     if mask is not None:
    #         u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
    #         b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
    #         l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
    #         r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
    #         u_diff = u_diff * u_mask
    #         b_diff = b_diff * b_mask
    #         l_diff = l_diff * l_mask
    #         r_diff = r_diff * r_mask
    #     u_over = (np.abs(u_diff) > 0).astype(np.float32)
    #     b_over = (np.abs(b_diff) > 0).astype(np.float32)
    #     l_over = (np.abs(l_diff) > 0).astype(np.float32)
    #     r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode='constant')
    b_over = np.pad(b_over, 1, mode='constant')
    l_over = np.pad(l_over, 1, mode='constant')
    r_over = np.pad(r_over, 1, mode='constant')
    u_diff = np.pad(u_diff, 1, mode='constant')
    b_diff = np.pad(b_diff, 1, mode='constant')
    l_diff = np.pad(l_diff, 1, mode='constant')
    r_diff = np.pad(r_diff, 1, mode='constant')

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]

def bilateral_filter(depth, config, discontinuity_map=None, mask=None, window_size=False):
    sigma_s = config['sigma_s']
    sigma_r = config['sigma_r']
    if window_size == False:
        window_size = config['filter_size']
    midpt = window_size//2

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1,1), (1,1)), 'edge')
    pad_depth = np.pad(depth, (midpt,midpt), 'edge')

    # filtering
    output_depth = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1,1])

    # if mask is not None:
    #     pad_mask = np.pad(mask, (midpt,midpt), 'constant')
    #     pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1,1])

    'weighted median filter here'
    if discontinuity_map is not None:
        # padding
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1, 1), (1, 1)), 'edge')
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt, midpt), 'edge')
        pad_discontinuity_hole = 1 - pad_discontinuity_map

        pad_discontinuity_patches = rolling_window(pad_discontinuity_map, [window_size, window_size], [1, 1])
        pad_discontinuity_hole_patches = rolling_window(pad_discontinuity_hole, [window_size, window_size], [1, 1])

        # filtering
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue

                # if there is no discontinuities on this patch - continue
                if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                    continue
                discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]

                coef = discontinuity_holes.astype(np.float32)
                # if mask is not None:
                #     coef = coef * pad_mask_patches[pi, pj]

                'if the whole patch is a discontinuity'
                if coef.max() == 0:
                    output_depth[pi, pj] = patch_midpt
                else:
                    # coef = sorta probability map that shows where there is NO discontinuity
                    coef = coef/(coef.sum())
                    # probabilities in increasing order of depth intensities (smaller = closer)
                    coef_order = coef.ravel()[depth_order]
                    # sorta like comulative distribution now
                    cum_coef = np.cumsum(coef_order)
                    # and we select index that shows 0.5 percentile
                    ind = np.digitize(0.5, cum_coef)
                    # and then select value at that index as filtered pix value
                    output_depth[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        'if discontinuity_map is None, bilateral filter is used'
        pH, pW = pad_depth_patches.shape[:2]
        ax = np.arange(-midpt, midpt + 1.)
        xx, yy = np.meshgrid(ax, ax)
        spatial_term = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma_s ** 2))

        for pi in range(pH):
            for pj in range(pW):

                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size//2, window_size//2]
                range_term = np.exp(-(depth_patch-patch_midpt)**2 / (2. * sigma_r**2))
                coef = spatial_term * range_term

                if coef.sum() == 0:
                    output_depth[pi, pj] = patch_midpt
                    continue
                else:
                    coef = coef/(coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output_depth[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output_depth

# cuts a into batches of size window
def rolling_window(a, window, strides):
    assert len(a.shape)==len(window)==len(strides), "\'a\', \'window\', \'strides\' dimension mismatch"
    shape_fn = lambda i,w,s: (a.shape[i]-w)//s + 1
    shape = [shape_fn(i,w,s) for i,(w,s) in enumerate(zip(window, strides))] + list(window)
    # multiplies all a's dimentions
    def acc_shape(i):
        if i+1>=len(a.shape):
            return 1
        else:
            return reduce(lambda x,y:x*y, a.shape[i+1:])
    _strides = [acc_shape(i)*s*a.itemsize for i,s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)

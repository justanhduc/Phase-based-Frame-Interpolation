import numpy as np
from scipy import misc
from pyPyrTools import SCFpyr
from skimage import color
from matplotlib import pyplot as plt
import cupy as cp
from skimage import transform


def decompose(img, ht, n_orientations, t_width, scale, n_scales):
    xp = cp.get_array_module(img)
    lab = cp.array(color.rgb2lab(cp.asnumpy(img))) / 255. if xp.__name__ == 'cupy' else color.rgb2lab(img) / 255.
    pyramids = {'pyramids': [], 'high_pass': [], 'low_pass': [], 'phase': [], 'amplitude': [], 'pind': 0}
    for i in range(img.shape[-1]):
    # for i in [0]:
        pyr = SCFpyr(lab[..., i], ht, n_orientations - 1, t_width, scale, n_scales)
        pyramids['pyramids'].append(pyr)
        pyramids['high_pass'].append(pyr.pyrHigh())
        pyramids['low_pass'].append(pyr.pyrLow())
        pyramids['phase'].append([xp.angle(pyr.pyr[level]) for level in range(1, len(pyr.pyr) - 1)])
        pyramids['amplitude'].append([xp.abs(pyr.pyr[level]) for level in range(1, len(pyr.pyr) - 1)])
    return pyramids


def compute_phase_difference(L, R, *args):
    xp = L['pyramids'][0].xp
    phase_diff_out = []
    for i in range(len(L['phase'])):
        phase_diff = [xp.arctan2(xp.sin(R['phase'][i][j] - L['phase'][i][j]), xp.cos(R['phase'][i][j] - L['phase'][i][j]))
                      for j in range(len(L['phase'][i]))]
        phase_diff_new = shift_correction(phase_diff, L['pyramids'][i], *args)
        unwrapped_phase_diff = []
        for j in range(len(phase_diff_new)):
            unwrapped_phase_diff.append(unwrap(xp.stack([phase_diff_new[j], list(phase_diff)[j]], 0))[0])
        phase_diff_out.append(unwrapped_phase_diff)
    return phase_diff_out


def shift_correction(pyr, pyramid, *args):
    n_high_elems = pyramid.pyrSize[0]
    n_low_elems = pyramid.pyrSize[-1]
    corrected_pyr = list(pyr)
    corrected_pyr.insert(0, np.zeros(n_high_elems))
    corrected_pyr.append(np.zeros(n_low_elems))
    n_levels = pyramid.spyrHt()
    n_bands = pyramid.numBands()
    for level in range(n_levels - 1, -1, -1):
        corrected_level = correct_level(corrected_pyr, pyramid, level, *args)
        start_ind = 1 + n_bands * level
        corrected_pyr[start_ind:start_ind+n_bands] = corrected_level
    corrected_pyr = corrected_pyr[1:len(corrected_pyr) - 1]
    return corrected_pyr


def correct_level(pyr, pyramid, level, *args):
    scale = args[0]
    limit = args[1]
    n_levels = pyramid.spyrHt()
    n_bands = pyramid.numBands()
    out_level = []
    if level < n_levels - 1:
        dims = pyramid.pyrSize[1+n_bands*level]
        for band in range(n_bands):
            index_lo = pyramid.bandIndex(level + 1, band)
            low_level_small = pyr[index_lo]
            if pyramid.xp.__name__ == 'numpy':
                low_level = transform.resize(low_level_small, dims, mode='reflect').astype('float32')
            else:
                low_level = pyramid.xp.array(transform.resize(pyramid.xp.asnumpy(low_level_small),
                                                              dims, mode='reflect').astype('float32'))
            index_hi = pyramid.bandIndex(level, band)
            high_level = pyr[index_hi]
            unwrapped = pyramid.xp.stack([low_level.reshape(-1) / scale, high_level.reshape(-1)], 0)
            unwrapped = unwrap(unwrapped)
            high_level = unwrapped[1]
            high_level = pyramid.xp.reshape(high_level, dims)
            angle_diff = pyramid.xp.arctan2(pyramid.xp.sin(high_level-low_level/scale),
                                            pyramid.xp.cos(high_level-low_level/scale))
            to_fix = pyramid.xp.abs(angle_diff) > (np.pi / 2)
            high_level[to_fix] = low_level[to_fix] / scale

            if limit > 0:
                to_fix = pyramid.xp.abs(high_level) > (limit * np.pi / scale ** (n_levels - level))
                high_level[to_fix] = low_level[to_fix] / scale

            out_level.append(high_level)

    if level == n_levels - 1:
        for band in range(n_bands):
            index_lo = pyramid.bandIndex(level, band)
            low_level = pyr[index_lo]
            if limit > 0:
                to_fix = pyramid.xp.abs(low_level) > (limit * np.pi / scale ** (n_levels - level))
                low_level[to_fix] = 0.
            out_level.append(low_level)
    return out_level


def unwrap(p, cutoff=np.pi):
    xp = cp.get_array_module(p)

    def local_unwrap(p, cutoff):
        dp = p[1] - p[0]
        dps = xp.mod(dp + np.pi, 2 * np.pi) - np.pi
        dps[xp.logical_and(dps == -np.pi, dp > 0)] = np.pi
        dp_corr = dps - dp
        dp_corr[xp.abs(dp) < cutoff] = 0.
        p[1] += dp_corr
        return p
    shape = p.shape
    p = xp.reshape(p, (shape[0], np.prod(shape[1:])))
    q = local_unwrap(p, cutoff)
    q = xp.reshape(q, shape)
    return q


def interpolate_pyramid(L, R, phase_diff, alpha):
    new_pyr = []
    for i in range(len(phase_diff)):
        new_pyr.append([])
        high_pass = L['high_pass'][i] if alpha < 0.5 else R['high_pass'][i]
        low_pass = (1 - alpha) * L['low_pass'][i] + alpha * R['low_pass'][i]
        new_pyr[i].append(high_pass)
        for k in range(len(R['phase'][i])):
            new_phase = R['phase'][i][k] + (alpha - 1) * phase_diff[i][k]
            new_amplitude = (1 - alpha) * L['amplitude'][i][k] + alpha * R['amplitude'][i][k]
            mid_band = new_amplitude * np.e ** (1j * new_phase)
            new_pyr[i].append(mid_band)
        new_pyr[i].append(low_pass)
    return new_pyr


def reconstruct_image(pyr):
    xp = pyr['pyramids'][0].xp
    out_img = xp.zeros((pyr['pyramids'][0].pyrSize[0][0], pyr['pyramids'][0].pyrSize[0][1], 3))
    for i, pyr in enumerate(pyr['pyramids']):
        out_img[..., i] = pyr.reconPyr('all', 'all')
    if xp.__name__ == 'numpy':
        out_img = color.lab2rgb(out_img * 255.)
    else:
        out_img = color.lab2rgb(cp.asnumpy(out_img) * 255.)
    return out_img


def interpolate_frame(img1, img2, n_frames=1, n_orientations=8, t_width=1, scale=0.5, limit=.4, min_size=15, max_levels=23):
    h, w, l = img1.shape
    n_scales = min(np.ceil(np.log2(min((h, w))) / np.log2(1. / scale) -
                           (np.log2(min_size) / np.log2(1 / scale))).astype('int'), max_levels)
    step = 1. / (n_frames + 1)

    L = decompose(img1, n_scales, n_orientations, t_width, scale, n_scales)
    R = decompose(img2, n_scales, n_orientations, t_width, scale, n_scales)

    phase_diff = compute_phase_difference(L, R, scale, limit)

    new_frames = []
    for j in range(n_frames):
        new_pyr = interpolate_pyramid(L, R, phase_diff, step * (j + 1))
        for i, pyr in enumerate(L['pyramids']):
            pyr.pyr = new_pyr[i]
        new_frames.append(reconstruct_image(L))
    return new_frames


if __name__ == '__main__':
    cp.cuda.Device(1).use()
    img1 = cp.array(misc.imread('E:/DB/Videos/DAVIS/JPEGImages/480p/surf/00003.jpg'))
    img2 = cp.array(misc.imread('E:/DB/Videos/DAVIS/JPEGImages/480p/surf/00005.jpg'))

    import time

    start = time.time()
    new_frames_gpu = interpolate_frame(img1, img2, n_frames=1, scale=.5**.25)
    print('Took %.2fm on GPU.' % ((time.time() - start) / 60.))

    img1 = cp.asnumpy(img1)
    img2 = cp.asnumpy(img2)
    start = time.time()
    new_frames_cpu = interpolate_frame(img1, img2, n_frames=1, scale=.5**.25)
    print('Took %.2fm on CPU.' % ((time.time() - start) / 60.))

    plt.figure(0)
    plt.imshow(img1)
    plt.figure(1)
    plt.imshow(new_frames_gpu[0])
    plt.figure(2)
    plt.imshow(new_frames_cpu[0])
    plt.figure(3)
    plt.imshow(img2)
    plt.show()

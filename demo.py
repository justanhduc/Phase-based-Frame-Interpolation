import numpy as np
from scipy import misc
import argparse
import time
from matplotlib import pyplot as plt

from frame_interp import interpolate_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str, help='Path to first frame.')
    parser.add_argument('img2', type=str, help='Path to second frame.')
    parser.add_argument('--n_frames', '-n', type=int, default=1, help='Number of new frames.')
    parser.add_argument('--dev', '-d', type=str, default='cpu', help='Choose a device to run on.')
    parser.add_argument('--gpu', type=int, default=0, help='Choose which GPU to use.')
    parser.add_argument('--show', '-sh', type=int, default=0, help='Display result.')
    parser.add_argument('--save', '-s', type=int, default=0, help='Save interpolated images.')
    parser.add_argument('--save_path', '-p', type=str, default='', help='Output path.')
    args = parser.parse_args()

    if args.dev == 'cpu':
        print('Using CPU.')
        xp = np
    elif args.dev == 'gpu':
        try:
            import cupy as cp
            xp = cp
            print('Using GPU.')
            xp.cuda.Device(args.gpu).use()
        except ImportError:
            xp = np
            print('No CUPY available. Using NUMPY instead.')
    else:
        raise NotImplementedError('Unknown choice of device.')

    img1 = misc.imread(args.img1)
    img2 = misc.imread(args.img2)

    start = time.time()
    new_frames = interpolate_frame(img1, img2, n_frames=args.n_frames, scale=.5**.25, xp=xp)
    print('Took %.2fm' % ((time.time() - start) / 60.))

    if args.save:
        import os
        for i in range(args.n_frames):
            misc.imsave(os.path.join(args.save_path, 'output%d.jpg' % (i+1)), new_frames[i])

    if args.show:
        plt.figure(0)
        plt.subplot(1, args.n_frames+2, 1)
        plt.imshow(img1)
        for i in range(args.n_frames):
            plt.subplot(1, args.n_frames+2, i+2)
            plt.imshow(new_frames[i])
        plt.subplot(1, args.n_frames+2, args.n_frames+2)
        plt.imshow(img2)
        plt.show()

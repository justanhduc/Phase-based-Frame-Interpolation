# Phase-based-Frame-Interpolation

This is my personal implementation of the paper "Phase-based frame interpolation for Video". To be precise, this implementation almost matches this [MATLAB implementation](https://github.com/owang/PhaseBasedInterpolation). As claimed by the orginal author of the MATLAB code, the implementation is highly un-optimized. Nevertheless, I added GPU support to reduce the processing time while still maintaining the clarity of the method. 

Also, if you use this implementation, you must cite the following paper:

``` 
S. Meyer, O. Wang, H. Zimmer, M. Grosse and A. Sorkine-Hornung
Phase-based frame interpolation for video
2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
``` 

This implementation uses a version of [pyrtools](https://github.com/justanhduc/pyPyrTools/tree/test-branch) by Eero Simoncelli in Python modified by me. Please check this [website](https://github.com/LabForComputationalVision/matlabPyrTools) for more information.

## Requirements
[pyrtools](https://github.com/justanhduc/pyPyrTools/tree/test-branch)

[cupy](https://github.com/cupy/cupy) v3.0.0a1 (fft is not supported in the earlier versions)

[scipy](https://www.scipy.org/)

[numpy](http://www.numpy.org/)

[skimage](http://scikit-image.org/)

## Usages

```
python demo.py image1_path image2_path --dev cpu --n_frames 1 --save 1 --show 1

```

Use help to see more options.

## Results
![Two interpolated frames](https://github.com/justanhduc/Phase-based-Frame-Interpolation/blob/master/example.png)


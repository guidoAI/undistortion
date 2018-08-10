#!/usr/bin/env python

'''

Adapted script from: https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py

Accompanying license:

---

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2018, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

---

Camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    calibrate.py [--debug <output path>] [--square_size] [<image mask>]

default values:
    --debug:    ./output/
    --square_size: 1.0
    <image mask> defaults to ../data/left*.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local modules
from common import splitfn

# built-in modules
import os

# information on the printed chessboard:
# Whether to estimate "normal" distortion parameters, or the fisheye ones:
FISH_EYE = True;
# number of squares of the chessboard. OpenCV requires the number of interior points.
n_squares_x = 8;
n_squares_y = 10;
n_interior_x = n_squares_x - 1;
n_interior_y = n_squares_y - 1;
# size of the squares in cm, allowing for non-square "squares" in the chessboard
size_x_cm = 2.30;
size_y_cm = 2.45;

if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
    args = dict(args)
    args.setdefault('--debug', './output/')
    args.setdefault('--square_size', 1.0)
    args.setdefault('--threads', 4)
    if not img_mask:
        img_mask = './chessboard/img_*.jpg'  # default
    else:
        img_mask = img_mask[0]

    img_names = glob(img_mask)
    debug_dir = args.get('--debug')
    if debug_dir and not os.path.isdir(debug_dir):
        os.mkdir(debug_dir)
    square_size = float(args.get('--square_size'))

    # the pattern size is the number of interior corners of the checkerboard:
    pattern_size = (n_interior_y, n_interior_x);
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    #pattern_points *= square_size
    
    # we have a non-square checkerboard: (pattern new bebop / JeVois)
    pattern_points[:, 0] *= size_y_cm;
    pattern_points[:, 1] *= size_x_cm;

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], 0).shape[:2]  # TODO: use imquery call to retrieve results

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, 0)
        #img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        
        if(FISH_EYE):
            return (corners, pattern_points);
        else:
            return (corners.reshape(-1, 2), pattern_points)

    print('Start processing');
    threads_num = int(args.get('--threads'))
    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    if(not FISH_EYE):
        # calculate camera distortion
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    else:
        # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
        # https://github.com/opencv/opencv/issues/11085
    
        for i in range(len(obj_points)):
            obj_points[i] = np.reshape(obj_points[i], (1, n_interior_x * n_interior_y,3));
        
        N_OK = len(obj_points)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        # rms, _, _, _, _ = \
        calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW;
        rms, camera_matrix, dist_coefs, rvecs, tvecs = \
            cv.fisheye.calibrate(
                obj_points,
                img_points,
                (w,h),
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    # undistort the image with the calibration
    print('')
    for fn in img_names if debug_dir else []:
        path, name, ext = splitfn(fn)
        img_found = os.path.join(debug_dir, name + '_chess.png')
        outfile = os.path.join(debug_dir, name + '_undistorted.png')

        img = cv.imread(img_found)
        if img is None:
            continue

        h, w = img.shape[:2]
        
        if(FISH_EYE):
            # Fisheye undistortion:    
            # undistortion with fisheye - keep more pixels:
            # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
            
            map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv.CV_16SC2)
                        
            dst = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        else:
            dst = cv.undistort(img, camera_matrix, dist_coefs);

        print('Undistorted image written to: %s' % outfile)
        cv.imwrite(outfile, dst)

    cv.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:57:21 2018

Function to undistort an image. It works by cycling over undistorted, normalized coordinates and 
then distorting them and transforming them to pixel coordinates in the original image. Currently, no
interpolation is used, as the function mainly serves to obtain the right undistortion parameters, instead
of generating visually pleasing images.

@author: Guido de Croon
"""

import cv2
import numpy as np
import pickle
from util_undistort import *

def undistort_image(min_y = -2, max_y = 2, step_y = 0.01, min_x = -2, max_x = 2, step_x = 0.01, image_name = None, 
                    K = None, k = None, DHANE = True, DHANE_k = 1.25, SAVE_TRAINING_SET = False):

    """ Undistorts an image with file name image_name. The min and max undistorted normalized coordinates can be set. 
        There is a choice between the Dhane undistortion model and the openCV fisheye model. Also, a training set of 
        undistorted normalized coordinates and distorted pixel coordinates can be created. Coordinates that cannot be
        inverted with the original (un)distortion function are made red.
    """

    if(image_name == None):
        print('Please provide an image name to undistort_image.');
        return;
    
    # image not used in the calibration:       
    Im = cv2.imread(image_name);

    # we *start* from normalized coordinates and then determine the distorted ones - in the pixel frame of the distorted images
    x = np.arange(min_x, max_x, step_x);
    y = np.arange(min_y, max_y, step_y);
    N = len(x);
    Undistorted = np.zeros([N, N, 3], dtype=np.uint8);

    v = np.zeros([3,1]);

    # show the progress per 10 percent of the image:
    n_steps = 10;
    pct_per_step = 100 / n_steps;
    progress_step = int(N/n_steps);

    # for learning the inverse mapping:
    Samples = [];
    Targets = [];

    for i in range(N):

        if(np.mod(i, progress_step) == 0):
            print('{} percent'.format(int(i/progress_step) * pct_per_step));
        
        for j in range(N):
            
            # normalized coordinates
            x_p = x[i];
            y_p = y[j];
            
            # distorted normalized coordinates:
            undistortion = False; # undistortion means that it can be undistorted by means of an invertible function
            if(DHANE):
                x_dist, y_dist = Dhane_distortion(x_p, y_p, k=DHANE_k);
                #print('({},{})'.format(x_dist, y_dist));
                x_n, y_n = Dhane_undistortion(x_dist, y_dist, k=DHANE_k);
                if(x_n != None):
                    undistortion = True;
            else:
                x_dist, y_dist = fisheye_distortion(x_p, y_p, k);
                    
            # use the camera matrix to retrieve the image coordinate in pixels in the distorted image:
            v[0] = x_dist;
            v[1] = y_dist;
            v[2] = 1;
            hom_coord_dist = np.dot(K, v);
            
            # for now, just round the coordinate:
            x_rounded = int(np.round(hom_coord_dist[0]));
            y_rounded = int(np.round(hom_coord_dist[1]));
            if(x_rounded >= 0 and x_rounded < Im.shape[1] and y_rounded >= 0 and y_rounded < Im.shape[0]):
                Undistorted[j,i,:] = np.mean(Im[y_rounded, x_rounded,:]);
                if(not undistortion):
                    # print('Uninvertible.');
                    Undistorted[j,i,0:2] = 0;
    
            # we add the normalized distorted coordinates to samples, as the step from image coordinates to such coords is simple:
            Samples.append(np.asarray([x_dist, y_dist]));
            Targets.append(np.asarray([x_p, y_p]))

    # save the training set:
    if(SAVE_TRAINING_SET):
        print('Save training set');
        f = open('training_set_undistortion.dat', 'w');
        pickle.dump([Samples, Targets], f);
        f.close();
    
    cv2.imwrite('undistorted.png', Undistorted);
    cv2.imshow('Undistorted', Undistorted);
    
    # Do the opposite: take the image and make pixels red that cannot be mapped:
    for x in range(Im.shape[1]):
        for y in range(Im.shape[0]):
            invertible = False;
            if(DHANE):
                x_n, y_n = from_distorted_pixels_to_normalized_coords(x, y, K, True);
                if(x_n != None):
                    invertible = True;
            if(not invertible):
                #print('uninvertible');
                Im[y,x,0:2] = 0;
            else:
                Im[y,x,:] = np.mean(Im[y,x,:]);
                
    cv2.imshow('Image with uninvertible regions in red', Im);
    

if(__name__ == '__main__'):
    
    image_name = 'img_distorted.jpg';
    
    DHANE = True;

    # Calibrations obtained with:
    #https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html

    # JeVois:
    #K = np.asarray([[189.68831995,   0.,         165.04353486],
    #                 [  0.,         188.59884425, 118.43686575],
    #                 [  0.,           0.,           1.        ]]);

    # Bebop 2:
    K = np.asarray([[311.59304538,   0.        , 158.37457814],
               [  0.        , 313.01338397, 326.49375925],
               [  0.        ,   0.        ,   1.        ]]);

    if(not DHANE):        
        # JeVois:
        #k = np.asarray([-0.36502361,  0.15915929,  0.00464537,  0.00082922, -0.03625312])
        # Bebop 2:
        k = np.asarray([-0.01239914, -0.11794404,  0.21664767, -0.11772531]);        
        undistort_image(image_name = image_name, K = K, DHANE = DHANE, k = k, SAVE_TRAINING_SET = False);
    else:    
        DHANE_k = 1.3; # set by means of trial-and-error with this function
        undistort_image(image_name = image_name, K = K, DHANE = DHANE, DHANE_k = DHANE_k, SAVE_TRAINING_SET = False);

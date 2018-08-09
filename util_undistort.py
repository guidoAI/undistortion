# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:46:59 2018

Supporting functions for image undistortion. 
If not specified, functions use the invertible Dhane distortion model.

@author: Guido de Croon.
"""

import numpy as np

def Dhane_distortion(x_p, y_p, k = 1.25, f = 1.0):
    """ Takes normalized coordinates and outputs distorted normalized coordinates
    """
    R = np.sqrt(x_p**2 + y_p**2);
    r = f * np.tan( np.arcsin( np.sin( np.arctan( R / f) ) * (1.0/k) ))
    reduction_factor = r/R;
    x_dist = reduction_factor * x_p;
    y_dist = reduction_factor * y_p;
    
    return x_dist, y_dist;    

def Dhane_undistortion(x_d, y_d, k = 1.25, f = 1.0):
    """ Takes distorted normalized coordinates and outputs undistorted normalized coordinates.
    """
    r = np.sqrt(x_d**2 + y_d**2);
    inner_part = np.sin( np.arctan( r / f) ) * k;
    if inner_part > 0.99:
        return None, None;
    R = f * np.tan( np.arcsin( inner_part ));
    enlargement_factor = R/r;
    x_p = enlargement_factor * x_d;
    y_p = enlargement_factor * y_d;
    
    return x_p, y_p;

def fisheye_distortion(x_p, y_p, k):
    """ Takes normalized coordinates and outputs distorted coordinates. This is the fisheye function
        as used by openCV: https://docs.opencv.org/trunk/db/d58/group__calib3d__fisheye.html.
        In principle uninvertible, although a lookup-table can be easily made (see plot_distortion_functions).
    """    
    r = np.sqrt(x_p**2 + y_p**2);
    theta = np.arctan(r);
    theta_d = theta * (1 + k[0] * theta**2 + k[1] * theta**4 + k[2] * theta**6 + k[3] * theta**8);
    x_dist = (theta_d / r) * x_p;
    y_dist = (theta_d / r) * y_p;
    
    return x_dist, y_dist;

def get_coords_for_radius(radius):
    """ Given a radius, will return x,y coordinates with x=y. 
        This is useful for plotting the relation between the undistorted and distorted radius.
    """
    sq = radius**2;
    coord = np.sqrt(sq / 2);
    return coord, coord;
    

def to_pixels(dist_coords, K):
    """ Takes distorted normalized coordinates, and outputs pixel coordinates:
    """
    pixel_coords = np.zeros(dist_coords.shape);
    for i in range(dist_coords.shape[0]):
        x_d = dist_coords[i,0];
        y_d = dist_coords[i,1];
        pixel_coords[i,0] = x_d * K[0,0] + K[0,2];
        pixel_coords[i,1] = y_d * K[1,1] + K[1,2];
        
    return pixel_coords;

def to_normalized(pix_coords, K):
    """ Takes distorted pixel coordinates and outputs distorted normalized coordinates.
    """    
    
    norm_coords = np.zeros(pix_coords.shape);
    for i in range(pix_coords.shape[0]):    
        x_px = pix_coords[i,0];
        y_px = pix_coords[i,1];
        norm_coords[i,0] = (x_px - K[0,2]) / K[0,0];
        norm_coords[i,1] = (y_px - K[1,2]) / K[1,1];

    return norm_coords;

def from_distorted_pixels_to_normalized_coords(x_old, y_old, K, DISTORTION=True):
    """ Takes distorted pixel coordinates and outputs undistorted normalized coordinates.
        Returns None, None if undistortion is not possible.
    """
    pix_coords = np.asarray([x_old, y_old]);
    pix_coords = pix_coords.reshape([1,2]);
    dist_norm_coords = to_normalized(pix_coords, K);
    if(DISTORTION):
        x_n, y_n = Dhane_undistortion(dist_norm_coords[0,0], dist_norm_coords[0,1]);
    else:
        x_n = dist_norm_coords[0,0];
        y_n = dist_norm_coords[0,1];
    
    return x_n, y_n;
    
def from_normalized_coords_to_distorted_pixels(x_n, y_n, K, DISTORTION=True):
    """ Takes undistorted normalized coordinates and returns distorted pixel coordinates.
    """
    if(DISTORTION):
        x_d, y_d = Dhane_distortion(x_n, y_n);
    else:
        x_d = x_n;
        y_d = y_n;
    dist_coords = np.asarray([x_d, y_d]).reshape([1,2]);
    pixel_coords = to_pixels(dist_coords, K);
    
    return pixel_coords[0,0], pixel_coords[0,1];


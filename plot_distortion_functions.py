# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:20:05 2018

Plot different distortion functions for insight into how they work.

@author: Guido de Croon.
"""

import numpy as np
from matplotlib import pyplot as plt
from util_undistort import *

def plot_openCV_fisheye(k = [], min_r = 0, max_r = 10, step_r = 0.01):
    """ Plot the openCV fisheye function.
        Note that the function may be non-monotonous.
        Also note that the way in which we plot the function can be used to create a lookup-table for inversion.
    """    
    
    if(k == []):
        print('No k given to plot_openCV_fisheye.');
        return;
    
    # R is undistorted
    # r is distorted
    
    plt.figure();    
    rs = [];
    Rs = [];
    for radius in np.arange(min_r, max_r, step_r):
        x, y = get_coords_for_radius(radius);
        x_d, y_d = fisheye_distortion(x, y, k);
        rs.append(radius);
        R = np.sqrt(x_d**2 + y_d**2);
        Rs.append(R);
        
    plt.plot(rs, Rs);
    plt.xlabel('r');
    plt.ylabel('R');

def plot_Dhane(k = None, min_k = 1.0, max_k = 2.0, step_k = 0.1, f = 1.0, min_r = 0, max_r = 10, step_r = 0.01):
    """ Plot the Dhane function. When k == None, thee function will be plotted for a range of k-values.
        Note that there may be distorted radii that are not reachable with the function (uninvertible coordinates).
    """
    if(k == None):
        plt.figure();
        leg = [];
        for k in np.arange(min_k, max_k, step_k):
            leg.append(str(k));
            rs = [];
            Rs = [];
            for radius in np.arange(min_r, max_r, step_r):
                x, y = get_coords_for_radius(radius);
                x_n, y_n = Dhane_undistortion(x, y, k, f);
                if(x_n != None):
                    rs.append(radius);
                    R = np.sqrt(x_n**2 + y_n**2);
                    Rs.append(R);
            plt.plot(rs, Rs);
        plt.xlabel('r');
        plt.ylabel('R');
        plt.legend(leg);
    else:
        rs = [];
        Rs = [];
        for radius in np.arange(min_r, max_r, step_r):
            x, y = get_coords_for_radius(radius);
            x_n, y_n = Dhane_undistortion(x, y, k, f);
            if(x_n != None):
                rs.append(radius);
                R = np.sqrt(x_n**2 + y_n**2);
                Rs.append(R);
        plt.plot(rs, Rs);
        plt.xlabel('r');
        plt.ylabel('R');

if(__name__ == '__main__'):

    DHANE = False;
    if(not DHANE):
        # specific distortion parameters for a Bebop2 
        k = np.asarray([-0.01239914, -0.11794404,  0.21664767, -0.11772531]);
        plot_openCV_fisheye(k = k);
    else:
        plot_Dhane();

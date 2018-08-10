# undistortion
Some undistortion functions in Python, notably using the invertible "Dhane" function, i.e., the one discussed in the article:

Dhane, P., Kutty, K., & Bangadkar, S. (2012). A generic non-linear method for fisheye correction. International Journal of Computer Applications, 51(10). (invertible fisheye model)

The Python scripts also contain the (non-invertible) distortion function used in OpenCV for fisheye cameras.

Running undistort_image on your image(s) can help to empirically tune the single k-parameter of the Dhane function. It does require a camera calibration matrix K that is to be obtained with normal (e.g., chessboard-based) calibration procedure.

Running plot_distortion_functions gives insight into the distortion functions (be it the Dhane function or the fisheye distortion function used by OpenCV).

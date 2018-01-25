import seaborn
import skimage.io as io
from scipy.signal import sawtooth
from skimage.future import graph as skg
from skimage import filters, color


import numpy as np

import tensorflow as tf
import warnings




def encoding_error(x, interval):
    """
    Returns the encoding error for the value x, considering a given encoding interval.
    Returned error range from 0 to 1.
    Example:
    If the only two meaningful ("correct") values of a pixel are 0 and 255, then the encoding interval (distance between
     two consecutive meaningful values) is 255, and
     encoding_error(0, 255) = 0
     encoding_error(127.5, 255) = 1
     encoding_error(0, 255) = 0

    Example:
     encoding_error(x, 1) = 0 iff x is an integer, 1 if x = k+0.5 for k in Z

    WARNING: This function is not precise if the magnitude of x is more then one order above the magnitude of interval
    and the effect is the more visible the more the two orders differ
    E.G
    encoding_error(10, 1) = 0
    encoding_error(11, 1) = 2.2759572004815709e-15 != 0
    encoding_error(1e8, 1) = 2.2759572004815709e-15 != 0
    encoding_error(1e14, 1) = 0.00057308744474998674 != 0


    :param x: the value for which calculate the error
    :param interval: the interval between two values for which (error = 0)
    :return:
    """
    if np.greater(np.abs(x), 10*interval).any():
        warnings.warn("Encoding error might not be accurate due to approximations", Warning)
    return (sawtooth(2 * np.pi * x / interval, 0.5) + 1) / 2


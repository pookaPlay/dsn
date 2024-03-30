"""
Visualization tools for dsn.
"""
import pdb
import cv2
import matplotlib.pyplot as plt


def overlay_img_with_labels(img, labels):
    """
    Overlay an image with labels and create
    a labeled color image. Each color signifies
    a different label.

    Parameters
    ----------
    img
       A grayscale or color image 
    labels
        A single channel image with different labels.

    Returns
    -------
    overlayed_image
        Image with labels overlayed.

    Todo
    ----
    1. Make this script work if the number of unique labels are less than or
       equal to 255.
    """

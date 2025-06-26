import os
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.transform
import scipy.ndimage.morphology
from functools import partial
import scipy.ndimage
import numpy as np
import pandas as pd

from . import calc_optimal_centers
#
# Crops mammography images using the method described in
# https://cs.nyu.edu/~kgeras/reports/datav1.0.pdf
#
# See also https://github.com/nyukat/breast_cancer_classifier/
#

N_ITER = 100
N_THRESHOLD = -0.999
N_BUFFER = 50


def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels+1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)
        
    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end


def get_bottommost_pixels(img, largest_mask, y_edge_bottom):
    """
    Gets the bottommost nonzero pixels of dilated mask before cropping. 
    """
    bottommost_nonzero_y = y_edge_bottom - 1
    bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right):
    """
    If we fail to recover the original shape as a result of erosion-dilation 
    on the side where the breast starts to appear in the image, 
    we record this information.
    """
    if mode == "left":
        return img.shape[1] - x_edge_right
    else:
        return x_edge_left


def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom     


def include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size):
    """
    Includes buffer in only one side of the image in x-direction
    """
    if mode == "left":
        if x_edge_left > 0:
            x_edge_left -= min(x_edge_left, buffer_size)
    else:
        if x_edge_right < img.shape[1]:
            x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
    return x_edge_left, x_edge_right


def convert_bottommost_pixels_wrt_cropped_image(mode, bottommost_nonzero_y, bottommost_nonzero_x,
                                                y_edge_top, x_edge_right, x_edge_left):
    """
    Once the image is cropped, adjusts the bottommost pixel values which was originally w.r.t. the original image
    """
    bottommost_nonzero_y -= y_edge_top
    if mode == "left":
        bottommost_nonzero_x = x_edge_right - bottommost_nonzero_x  # in this case, not in sorted order anymore.
        bottommost_nonzero_x = np.flip(bottommost_nonzero_x, 0)
    else:
        bottommost_nonzero_x -= x_edge_left
    return bottommost_nonzero_y, bottommost_nonzero_x


def get_rightmost_pixels_wrt_cropped_image(mode, largest_mask_cropped, find_rightmost_from_ratio):
    """
    Ignores top find_rightmost_from_ratio of the image and searches the rightmost nonzero pixels
    of the dilated mask from the bottom portion of the image.
    """
    ignore_height = int(largest_mask_cropped.shape[0] * find_rightmost_from_ratio)
    rightmost_pixel_search_area = largest_mask_cropped[ignore_height:, :]
    rightmost_pixel_search_area_has_value = np.any(rightmost_pixel_search_area, axis=0)
    rightmost_nonzero_x = np.arange(rightmost_pixel_search_area.shape[1])[
        rightmost_pixel_search_area_has_value][-1 if mode == 'right' else 0]
    rightmost_nonzero_y = np.arange(rightmost_pixel_search_area.shape[0])[
        rightmost_pixel_search_area[:, rightmost_nonzero_x] > 0] + ignore_height

    # rightmost pixels are already found w.r.t. newly cropped image, except that we still need to
    #   reflect horizontal_flip
    if mode == "left":
        rightmost_nonzero_x = largest_mask_cropped.shape[1] - rightmost_nonzero_x
        
    return rightmost_nonzero_y, rightmost_nonzero_x

def crop_img_from_largest_connected(img, mode, erode_dialate=True, iterations=100, 
                                    buffer_size=50, find_rightmost_from_ratio=1/3):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component, and draws a bounding box for the result
    with buffers

    input:
        - img:   2D numpy array
        - mode:  breast pointing left or right

    output: a tuple of (window_location, rightmost_points, 
                        bottommost_points, distance_from_starting_side)
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        - rightmost_points: rightmost nonzero pixels after correctly being flipped in the format of 
                            ((y_start, y_end), x)
        - bottommost_points: bottommost nonzero pixels after correctly being flipped in the format of
                             (y, (x_start, x_end))
        - distance_from_starting_side: number of zero columns between the start of the image and start of
           the largest connected component w.r.t. original dicom image.
    """
    assert mode in ("left", "right")

    img_mask = img > 0

    # Erosion in order to remove thin lines in the background
    if erode_dialate:
        img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    if erode_dialate:
        largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)
    
    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # extract bottommost pixel info
    bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)

    # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)
    
    # If cropped image not starting from corresponding edge, they are wrong. Record the distance, will reject if not 0.
    distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)

    # include more pixels on either side just to make sure we don't miss anything, if the next column
    #   contains non-zero value and isn't noise
    x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

    # convert bottommost pixel locations w.r.t. newly cropped image. Flip if necessary.
    bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
        mode,
        bottommost_nonzero_y,
        bottommost_nonzero_x,
        y_edge_top,
        x_edge_right,
        x_edge_left
    )

    # calculate rightmost point from bottom portion of the image w.r.t. cropped image. Flip if necessary.
    rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
        mode,
        largest_mask[y_edge_top: y_edge_bottom, x_edge_left: x_edge_right],
        find_rightmost_from_ratio
    )

    # save window location in medical mode, but everything else in training mode
    return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right), \
        ((rightmost_nonzero_y[0], rightmost_nonzero_y[-1]), rightmost_nonzero_x), \
        (bottommost_nonzero_y, (bottommost_nonzero_x[0], bottommost_nonzero_x[-1])), \
        distance_from_starting_side


def find_crop_center(x, crop_size=[1024, 1024], side='left', view='cc'):
    if side == 'right':
        # flip left-right
        x = x[:, ::-1]

    # find nonzero pixels
    x_nonzero = x > N_THRESHOLD

    # erode N_ITER times
    x_obj_mask = scipy.ndimage.morphology.binary_erosion(x_nonzero, iterations=N_ITER)

    # find the largest connected component
    x_obj_mask, n_objects = skimage.morphology.label(x_obj_mask, return_num=True)
    largest_object_label = np.argmax(np.sum(x_obj_mask == label + 1) for label in range(n_objects))

    # mask the largest connected component
    x_obj_mask = (x_obj_mask == largest_object_label + 1)

    # dilate N_ITER times
    x_obj_mask = scipy.ndimage.morphology.binary_dilation(x_obj_mask, iterations=N_ITER)

    # find bounding box
    rprops = skimage.measure.regionprops(x_obj_mask.astype(int))[0]

    # add buffer
    ymin = max(0, min(rprops.bbox[0] - N_BUFFER, x_obj_mask.shape[0]))
    xmin = max(0, min(rprops.bbox[1] - N_BUFFER, x_obj_mask.shape[1]))
    ymax = max(0, min(rprops.bbox[2] + N_BUFFER, x_obj_mask.shape[0]))
    xmax = max(0, min(rprops.bbox[3] + N_BUFFER, x_obj_mask.shape[1]))

    # add constraints based on view
    if view == 'cc':
        tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(rightmost_x=xmax)
    else:
        assert view == 'mlo'
        tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(rightmost_x=xmax,
                                                                                     bottommost_y=ymax)

    # compute optimal center
    wininfo = calc_optimal_centers.get_image_optimal_window_info(
        image=x_obj_mask,
        com=np.array(x_obj_mask.shape) // 2,
        window_dim=np.array(crop_size),
        tl_br_constraint=tl_br_constraint)

    best_center_y = wininfo['best_center_y']
    best_center_x = wininfo['best_center_x']

    if side == 'right':
        # flip left-right
        best_center_x = x.shape[1] - best_center_x

    return (best_center_y, best_center_x), rprops.bbox, wininfo
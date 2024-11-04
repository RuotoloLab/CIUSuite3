"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Methods for raw data pre-processing (smoothing, interpolation, etc) for CIU-type data
Author: Dan Polasky
Date: 10/6/2017
"""

import numpy as np
import os
import scipy.signal
import scipy.interpolate
import scipy.stats
import pygubu
import logging
from tkinter import messagebox
import tkinter as tk

import CIU_analysis_obj
import CIU_raw
import CIU2_Main

logger = logging.getLogger('main')


def process_raw_obj(raw_obj, params_obj, short_filename=None):
    """
    Run all initial raw processing stages (data import, smoothing, interpolation, cropping)
    on a raw file using the parameters provided in a Parameters object. Returns a NEW
    analysis object with the processed data
    :param raw_obj: the CIURaw object containing the raw data to process
    :type raw_obj: CIURaw
    :param params_obj: Parameters object containing processing parameters
    :type params_obj: Parameters
    :param short_filename: filename to save into the new object (e.g. if reconstructing)
    :rtype: CIUAnalysisObj
    :return: CIUAnalysisObj with processed data
    :raises: ValueError
    """
    # check for empty raw data and don't initialize object if empty
    if np.max(raw_obj.rawdata) == 0:
        logger.warning('Empty raw data in file {}'.format(raw_obj.filepath))
        raise ValueError

    # normalize data and save axes information
    norm_data = normalize_by_col(raw_obj.rawdata)
    axes = (raw_obj.dt_axis, raw_obj.cv_axis)

    # save a CIUAnalysisObj with the information above
    analysis_obj = CIU_analysis_obj.CIUAnalysisObj(raw_obj, norm_data, axes, params_obj, short_filename=short_filename)
    analysis_obj = smooth_main(analysis_obj, params_obj)

    return analysis_obj


def get_data(filename):
    """
    Read _raw.csv file and generate a CIURaw object containing its raw data and filename
    :param filename: string - path to _raw.csv file to read
    :rtype: CIURaw
    :return: CIURaw object with rawdata, axes, and filename initialized
    """
    try:
        rawdata = np.genfromtxt(filename, missing_values=[""], filling_values=[0], delimiter=",")
    except ValueError:
        # bad characters in file, or other numpy import errors
        logger.error('Data import error in File: {}. Illegal characters or other error in importing data. This file will NOT be processed'.format(os.path.basename(filename)))
        raise ValueError('Data import error in File: ', os.path.basename(filename), 'Illegal characters or other error in importing data. This file will NOT be processed')
    row_axis = rawdata[1:, 0]
    col_axis = rawdata[0, 1:]

    # check for decreasing values in either axis
    prev_val = 0
    for value in row_axis:
        if value < prev_val:
            logger.error('Data import error in File: {}. Mobility axis has decreasing or negative values. All axis values must be positive and increasing. This file will NOT be processed'.format(os.path.basename(filename)))
            raise ValueError('Data import error in File: ', os.path.basename(filename), 'Mobility axis has decreasing or negative values. All axis values must be positive and increasing. This file will NOT be processed')
    prev_val = 0
    for value in col_axis:
        if value < prev_val:
            logger.error('Data import error in File: {}. Activation axis has decreasing or negative values. All axis values must be positive and increasing. This file will NOT be processed'.format(os.path.basename(filename)))
            raise ValueError('Data import error in File: ', os.path.basename(filename), 'Activation axis has decreasing or negative values. All axis values must be positive and increasing. This file will NOT be processed')

    # check for duplicate values in axes
    unique_row, row_counts = np.unique(row_axis, return_counts=True)
    if np.max(row_counts) > 1:
        logger.error('Data import error in File: {}. Duplicate row (DT) values. This file will NOT be processed'.format(os.path.basename(filename)))
        raise ValueError('Data import error in File: ', os.path.basename(filename), 'Duplicate row (DT) values. This file will NOT be processed')
    unique_col, col_counts = np.unique(col_axis, return_counts=True)
    if np.max(col_counts) > 1:
        logger.error('Data import error in File: {}. Duplicate column (CV) values. This file will NOT be processed.'.format(os.path.basename(filename)))
        raise ValueError('Data import error in File: ', os.path.basename(filename), 'Duplicate column (CV) values. This file will NOT be processed.')

    raw_obj = CIU_raw.CIURaw(rawdata[1:, 1:], row_axis, col_axis, filename)

    return raw_obj


def normalize_by_col(raw_data_matrix):
    """
    Generate a normalized dataset where each column has been normalized to 1 individually. Returns
    normalized 2D matrix only.
    :param raw_data_matrix: CIURaw.rawdata - 2D numpy array with no axes
    :return: Normalized 2D numpy array of raw data
    """
    maxint = np.amax(raw_data_matrix, axis=0)  # find the maximum value for each energy
    if 0 in maxint:
        # set any columns with 0 total intensity to have normalized intensity of 0 as well
        norm = raw_data_matrix / maxint

        index_tup = np.where(maxint == 0)
        index_list = index_tup[0]
        for index in index_list:
            # zero the column at this index (where the max value is 0 - replace all NANs with 0s)
            norm[:, index] = np.zeros(np.shape(norm[:, index]))
    else:
        # no zero values, so simply divide
        norm = raw_data_matrix / maxint
    return norm


def sav_gol_smooth(ciu_data_matrix, smooth_window, smooth_order):
    """
    Apply savitsky-golay smoothing to each column (CV) of the 2D matrix supplied
    :param ciu_data_matrix: input matrix (2D, columns (axis 1) gets smoothed) - supply without axes
    :param smooth_window: Savitsky golay smoothing window to apply
    :param smooth_order: polynomial order to use for smoothing
    :return: smoothed data (same size/format as input)
    """
    # ensure window length is odd
    if smooth_window % 2 == 0:
        smooth_window += 1

    # swap axes to access the columns
    cv_data = np.swapaxes(ciu_data_matrix, 0, 1)
    output_data = np.ndarray(np.shape(cv_data))

    # smooth each column and return the data (axes swapped back to normal)
    index = 0
    while index < len(cv_data):
        smoothed_col = scipy.signal.savgol_filter(cv_data[index], smooth_window, polyorder=smooth_order)
        output_data[index] = smoothed_col
        index += 1
    output_data = np.swapaxes(output_data, 0, 1)

    return output_data


def smooth_main(analysis_obj, params_obj):
    """
    Integrated smoothing method for analysis object/parameter object combinations for general use
    :param analysis_obj: CIU container with data to smooth
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: parameter container with smoothing parameters
    :type params_obj: Parameters
    :return: updated analysis_obj (.ciu_data updated, no other changes)
    """
    norm_data = analysis_obj.ciu_data

    if params_obj.smoothing_1_method is not None:
        # ensure window size is odd
        if params_obj.smoothing_2_window % 2 == 0:
            params_obj.smoothing_2_window += 1

        i = 0
        while i < params_obj.smoothing_3_iterations:
            if params_obj.smoothing_1_method.lower() == '2d savitzky-golay':
                # catch data with only 1 row or column
                if len(analysis_obj.axes[0]) < params_obj.smoothing_2_window or len(analysis_obj.axes[1]) < 2:
                    logger.error('File {} has at least one axis too small for 2D smoothing. 2D smoothing requires at least 2 activation axis bins and at least the smoothing window size mobility bins. The file was NOT smoothed'.format(analysis_obj.short_filename))
                    break
                norm_data = sgolay2d(norm_data, params_obj.smoothing_2_window, order=2)

            elif params_obj.smoothing_1_method.lower() == '1d savitzky-golay':
                # catch data with only 1 row
                if len(analysis_obj.axes[0]) < params_obj.smoothing_2_window:
                    logger.error('The mobility axis in file {} has fewer bins than the smoothing window and cannot be smoothed. The file was NOT smoothed'.format(
                            analysis_obj.short_filename))
                    break
                norm_data = sav_gol_smooth(norm_data, params_obj.smoothing_2_window, smooth_order=2)

            else:
                logger.error('Invalid smoothing method, no smoothing applied')
                break
            i += 1

    # renormalize data
    norm_data = normalize_by_col(norm_data)

    analysis_obj.ciu_data = norm_data
    return analysis_obj


def sgolay2d(z, window_size, order, derivative=None):
    """
    ADAPTED FROM THE SCIPY COOKBOOK, at http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    accessed 2/14/2018. Performs a 2D Savitzky-Golay smooth on the the provided 2D array z using parameters
    window_size and polynomial order.
    :param z: 2D numpy array of data to smooth
    :param window_size: filter size (int), must be odd to use for smoothing
    :param order: polynomial order (int) to use
    :param derivative: optional (string), values = row, col, both, or None
    :return: smoothed 2D numpy array
    """

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        logger.error('2D SG smooth window_size ({}) must be odd'.format(window_size))
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        logger.error('2D SG smooth order is too high for the window size of {}'.format(window_size))
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [(k-n, n) for k in range(order + 1) for n in range(k+1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    a_mat = np.empty((window_size ** 2, len(exps)))
    for i, exp in enumerate(exps):
        a_mat[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    z_mat = np.zeros(new_shape)
    # top band
    band = z[0, :]
    z_mat[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1:half_size+1, :]) - band)
    # bottom band
    band = z[-1, :]
    z_mat[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size-1:-1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    z_mat[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1:half_size+1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    z_mat[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size-1:-1]) - band)
    # central band
    z_mat[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    z_mat[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1:half_size+1, 1:half_size+1])) - band)
    # bottom right corner
    band = z[-1, -1]
    z_mat[-half_size:, -half_size:] = band + np.abs(np.flipud(np.fliplr(z[-half_size-1:-1, -half_size-1:-1])) - band)
    # top right corner
    band = z_mat[half_size, -half_size:]
    z_mat[:half_size, -half_size:] = band - np.abs(np.flipud(z_mat[half_size+1:2*half_size+1, -half_size:]) - band)
    # bottom left corner
    band = z_mat[-half_size:, half_size].reshape(-1, 1)
    z_mat[-half_size:, :half_size] = band - np.abs(np.fliplr(z_mat[-half_size:, half_size+1:2*half_size+1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(a_mat)[0].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(a_mat)[1].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(a_mat)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(a_mat)[1].reshape((window_size, -1))
        r = np.linalg.pinv(a_mat)[2].reshape((window_size, -1))
        return scipy.signal.fftconvolve(z_mat, -r, mode='valid'), scipy.signal.fftconvolve(z_mat, -c, mode='valid')


def find_nearest(array, value):
    """
    Get the index in the array nearest to the input value. Handles values outside the
    array by returning the end value in the correct direction.
    :param array: array-like object to search within
    :param value: value to find nearest index in the array
    :return: index (int) of closest match in the array
    """
    idx = (np.abs(np.asarray(array) - value)).argmin()
    return idx


def crop(analysis_obj, crop_vals):
    """
    Crops the data and axes arrays to the nearest values specified in the crop vals list.
    If provided crop values are outside the axes, will crop to the nearest value.
    :param analysis_obj: CIUAnalysisObj with data to crop
    :type analysis_obj: CIUAnalysisObj
    :param crop_vals: list of values to crop to in form [cv_low, cv_high, dt_low, dt_high]
    :rtype: CIUAnalysisObj
    :return: New CIUAnalysisObj with cropped data and new axes
    """
    # Determine the indices corresponding to the values nearest to those entered by the user
    dt_axis = analysis_obj.axes[0]
    cv_axis = analysis_obj.axes[1]
    ciu_data_matrix = analysis_obj.ciu_data

    # Crop
    dt_low = find_nearest(dt_axis, crop_vals[0])
    dt_high = find_nearest(dt_axis, crop_vals[1])
    cv_low = find_nearest(cv_axis, crop_vals[2])
    cv_high = find_nearest(cv_axis, crop_vals[3])

    # allow single axis cropping - ignore one axis if its provided values are equal
    crop_dt = True
    crop_cv = True
    if dt_high == dt_low:
        crop_dt = False
    if cv_low == cv_high:
        crop_cv = False

    # crop the data and axes
    if crop_dt:
        ciu_data_matrix = ciu_data_matrix[dt_low:dt_high + 1]  # crop the rows
        dt_axis = dt_axis[dt_low:dt_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    if crop_cv:
        ciu_data_matrix = ciu_data_matrix[cv_low:cv_high + 1]  # crop the columns
        cv_axis = cv_axis[cv_low:cv_high + 1]
    ciu_data_matrix = np.swapaxes(ciu_data_matrix, 0, 1)
    new_axes = [dt_axis, cv_axis]

    # crop the data for the rare case that the highest value in a column is cropped out
    ciu_data_matrix = normalize_by_col(ciu_data_matrix)

    # save output to a new analysis object (clears fitting results/etc that can fail if axes are different)
    new_obj = CIU_analysis_obj.CIUAnalysisObj(analysis_obj.raw_obj, ciu_data_matrix, new_axes, analysis_obj.params)
    new_obj.filename = analysis_obj.filename
    return new_obj


def interpolate_axes(analysis_obj, new_axes):
    """
    interpolate along the collision voltage (x) axis to allow for unevenly spaced data collection
    :param analysis_obj: input data object
    :type analysis_obj: CIUAnalysisObj
    :param new_axes: new axes onto which to interpolate in form [dt_axis, cv_axis]
    :return: updated object with interpolate CIUData and axes
    :rtype: CIUAnalysisObj
    """
    # interpolate the existing data, then reframe on new axes
    interp_func = scipy.interpolate.RectBivariateSpline(analysis_obj.axes[0],
                                             analysis_obj.axes[1],
                                             analysis_obj.ciu_data.T)
    interp_data = interp_func(new_axes[0], new_axes[1])
    ciu_interp_data = np.asarray(interp_data).T

    # save to analysis object and return
    analysis_obj.ciu_data = ciu_interp_data
    analysis_obj.axes = new_axes
    analysis_obj.refresh_data()
    return analysis_obj


def interpolate_axis_1d(analysis_obj, dt_axis_bool, new_axis):
    """
    Interpolate an axis in one dimension iteratively across the CIU dataset. 2D interpolation is
    recommended in most cases, but sometimes (e.g. if only one CV) isn't good
    :param analysis_obj: data container
    :type analysis_obj: CIUAnalysisObj
    :param dt_axis_bool: Whether to interpolate DT axis (True) or CV axis (False)
    :param new_axis: new axis to interpolate, taken from compute_new_axes
    :return: updated analysis_obj with ciu_data and axes edited to new values
    """
    new_data = []
    if dt_axis_bool:
        # interpolate along the drift axis within each CV column using the new_axis provided
        cv_col_data = analysis_obj.ciu_data.T
        new_axes = [new_axis, analysis_obj.axes[1]]
        for cv_col in cv_col_data:
            interp_fn = scipy.interpolate.interp1d(analysis_obj.axes[0], cv_col)
            new_col = interp_fn(new_axis)
            new_data.append(new_col)
        new_data = np.asarray(new_data).T
    else:
        # interpolate along the CV axis within each drift bin using the new_axis provided
        dt_row_data = analysis_obj.ciu_data
        new_axes = [analysis_obj.axes[0], new_axis]
        for dt_row in dt_row_data:
            interp_fn = scipy.interpolate.interp1d(analysis_obj.axes[1], dt_row)
            new_row = interp_fn(new_axis)
            new_data.append(new_row)
        new_data = np.asarray(new_data)

    # save output to analysis object and return
    analysis_obj.axes = new_axes
    analysis_obj.ciu_data = new_data
    analysis_obj.refresh_data()
    return analysis_obj


def compute_new_axes(old_axes, interpolation_scaling, interp_cv=True, interp_dt=False):
    """
    Determine new (interpolated) axes based on the old axes and a scaling factor. Designed
    to preserve the original axes values as much as possible (e.g. if axes scaled by 2,
    all the original axis points will remain along with new points halfway in between).
    New bin num calculation: axis length is one more than the number of steps, so subtract 1,
    double the number of steps, then add 1 back to include the axis endpoint from linspace.
    :param old_axes: [dt_axis, cv_axis] list of numpy arrays
    :param interpolation_scaling: scaling factor to apply to the data. Equal to the new_num_bins / old_num_bins
    :param interp_cv: whether to interpolate the CV axis
    :param interp_dt: whether to interpolate the DT axis
    :return: new axes [dt_axis, cv_axis] (list of numpy arrays)
    """
    # interpolate CV if requested
    if interp_cv:
        cv_start = old_axes[1][0]
        cv_end = old_axes[1][-1]
        new_num_bins = (len(old_axes[1]) - 1) * interpolation_scaling + 1
        new_cv_axis = np.linspace(cv_start, cv_end, new_num_bins)
    else:
        new_cv_axis = old_axes[1]

    # interpolate DT if requested
    if interp_dt:
        dt_start = old_axes[0][0]
        dt_end = old_axes[0][-1]
        new_num_bins = (len(old_axes[0]) - 1) * interpolation_scaling + 1
        new_dt_axis = np.linspace(dt_start, dt_end, new_num_bins)
    else:
        new_dt_axis = old_axes[0]

    return [new_dt_axis, new_cv_axis]


def check_axes_crop(analysis_obj_list):
    """
    Interrogate all CIUAnalysisObjs in the provided list to determine the shared dimensions in all files to use for
    cropping to a shared axis. Interpolation done separately to avoid interpolating in cases where objects have
    different size axes but the same voltage step.
    :param analysis_obj_list: list of CIUAnalysis objects
    :type analysis_obj_list: list[CIUAnalysisObj]
    :return: [dt_start, dt_end, cv_start, cv_end], [min_dt_spacing, min_cv_spacing] Maximum dimension DT/CV axes that are shared amongst all files, and the minimum axes sizes
    """
    # Dimensions to crop to (maximum shared area amongst all fingerprints)
    dt_start_max = 0
    dt_end_min = np.inf
    cv_start_max = 0
    cv_end_min = np.inf

    # min_dt_bins = np.inf
    # min_cv_bins = np.inf
    min_dt_spacing = np.inf
    min_cv_spacing = np.inf

    # take the smaller dimension as the shared area (e.g. if a file only goes to 100V and others go to 110V, sharing stops at 100V)
    for analysis_obj in analysis_obj_list:
        if analysis_obj.axes[0][0] > dt_start_max:
            dt_start_max = analysis_obj.axes[0][0]
        if analysis_obj.axes[0][-1] < dt_end_min:
            dt_end_min = analysis_obj.axes[0][-1]
        if analysis_obj.axes[1][0] > cv_start_max:
            cv_start_max = analysis_obj.axes[1][0]
        if analysis_obj.axes[1][-1] < cv_end_min:
            cv_end_min = analysis_obj.axes[1][-1]

        # check bin SPACING for later interpolation (if needed).
        bin_spacings_dt = [analysis_obj.axes[0][x + 1] - analysis_obj.axes[0][x] for x in range(len(analysis_obj.axes[0]) - 1)]
        bin_spacings_cv = [analysis_obj.axes[1][x + 1] - analysis_obj.axes[1][x] for x in range(len(analysis_obj.axes[1]) - 1)]

        print(bin_spacings_dt)
        print(scipy.stats.mode(bin_spacings_dt)[0])
        print(bin_spacings_cv)
        print(scipy.stats.mode(bin_spacings_cv)[0])

        # using most common (mode) spacing in case of unevenly spaced data.
        if np.median(bin_spacings_dt) < min_dt_spacing:
            min_dt_spacing = scipy.stats.mode(bin_spacings_dt)[0]
        if np.median(bin_spacings_cv) < min_cv_spacing:
            min_cv_spacing = scipy.stats.mode(bin_spacings_cv)[0]

    crop_vals_all = [dt_start_max, dt_end_min, cv_start_max, cv_end_min, min_dt_spacing, min_cv_spacing]
    for val in crop_vals_all:
        if val < 0:
            logger.error('Negative value found in axes checking. Errors possible. Please check the raw data')
    return [dt_start_max, dt_end_min, cv_start_max, cv_end_min], [min_dt_spacing, min_cv_spacing]


def check_axes_interp(crop_vals, axes_spacings):
    """
    Check all objects in the provided list for whether their axes have the same spacing. Increases the number of bins
    to the maximum in all files if there are any differences.
    NOTE: should be called AFTER check_axes_crop to ensure that axes are beginning and ending in the same place.
    :param crop_vals: list [dt_start, dt_end, cv_start, cv_end] in the finalized axes (previously run through check_axes_crop)
    :param axes_spacings: [dt_spacing, cv_spacing] - list of determined spacings (distances) between axes values
    :return: [dt_axis, cv_axis] optimized axes starting and ending in the same place and with identical bin spacing
    """
    # assemble final axes using start/end values and spacing
    final_dt_axis = []
    current_dt = crop_vals[0]
    # use tolerance of 1e-12 to prevent weird floating point equality issues
    while current_dt - crop_vals[1] < 1e-12:
        final_dt_axis.append(current_dt)
        current_dt += axes_spacings[0]

    final_cv_axis = []
    current_cv = crop_vals[2]
    while current_cv - crop_vals[3] < 1e-12:
        final_cv_axis.append(current_cv)
        current_cv += axes_spacings[1]

    return [np.asarray(final_dt_axis), np.asarray(final_cv_axis)]


def equalize_obj(analysis_obj, final_axes):
    """
    Adjust the axes of a CIUAnalysis object to a set of finalized axes (provided). Crops first, then interpolates
    :param analysis_obj: object to adjust
    :type analysis_obj: CIUAnalysisObj
    :param final_axes: [dt_axis, cv_axis] final axes to adjust to.
    :return: updated analysis_obj, boolean (True if adjustments were made)
    :rtype: CIUAnalysisObj, bool
    :raises: ValueError
    """
    # compute bin spacing on DT axis to determine relative difference acceptable tolerance (set to 1/10 of a bin)
    # dt_spacing = scipy.stats.mode([final_axes[0][x + 1] - final_axes[0][x] for x in range(len(final_axes[0]) - 1)])[0][0]
    # dt_tol = dt_spacing * 0.1
    # dt_tol deprecated because CCS-calibrated files can succeed given this tolerance when they should fail
    dt_tol = 0.001
    try:
        if np.allclose(analysis_obj.axes[0], final_axes[0], rtol=dt_tol) \
                and np.allclose(analysis_obj.axes[1], final_axes[1]):
            return analysis_obj, False
        else:
            # precise adjustment - use exact final axes provided rather than nearest approx (typical) cropping method
            analysis_obj = interpolate_axes(analysis_obj, final_axes)
            logger.info('equalized axes to match in file {}'.format(analysis_obj.short_filename))
            return analysis_obj, True
    except ValueError:
        # if axes are different shapes, allclose raises a ValueError. Adjust axes.
        try:
            analysis_obj = interpolate_axes(analysis_obj, final_axes)
        except ValueError:
            logger.error('Axis equalization failed in file {}. Axes to be equalized must overlap at least partially.'.format(analysis_obj.short_filename))
            raise ValueError('Axis equalization failed in file: {} '.format(analysis_obj.short_filename), 'Axes to be equalized must overlap at least partially')
            # return analysis_obj, False

        logger.info('equalized axes to match in file {}'.format(analysis_obj.short_filename))
        return analysis_obj, True


def equalize_axes_main(list_of_analysis_objs):
    """
    Unified entry point for axis checking and equalization for many methods (crop/avg, classification, etc).
    Determines the best shared axes for all files, then adjusts any files that aren't matched to them.
    Attempts to preserve the original CV axis values as much as possible, but may not in all cases (e.g. uneven
    spacings between CVs). Also attempts to preserve DT original values, but since (at least for Waters data)
    the drift time spacings are inherently uneven due to the data compression scheme they use, drift time
    will typically not be preserved.
    :param list_of_analysis_objs: list of CIUAnalysisObjs to equalize axes across
    :type list_of_analysis_objs: list[CIUAnalysisObj]
    :return: list of analysis objects with axes updated, final axes
    """
    # check axes for cropping (ensure that region of interest is equal)
    crop_vals, axes_spacings = check_axes_crop(list_of_analysis_objs)
    final_axes = check_axes_interp(crop_vals, axes_spacings)

    # perform any adjustments required
    adjust_flags = []
    for analysis_obj in list_of_analysis_objs:
        try:
            analysis_obj, current_flag = equalize_obj(analysis_obj, final_axes)
            adjust_flags.append(current_flag)
        except ValueError as err:
            list_of_analysis_objs.remove(analysis_obj)
            messagebox.showerror('Axes Could Not Be Equalized',
                                 message='{}. \nProblem: {}. This file will NOT be included in comparison. Please make sure axes overlap at least partially. Press OK to continue'.format(*err.args))

    # check if any adjustments were made to flag for methods that want it
    if True in adjust_flags:
        any_adjust_flag = True
    else:
        any_adjust_flag = False

    return list_of_analysis_objs, final_axes, any_adjust_flag


def equalize_axes_2d_list_subclass(cl_input_list_by_label):
    """
    Axis checking method for 2D lists (e.g. in classification) where list order/shape must be
    preserved. Axes are equalized across ALL sublists (every object anywhere in the 2D list)
    :param cl_input_list_by_label: list of lists of Classification.ClInput
    :type cl_input_list_by_label: list[list[Classification.ClInput]]
    :return: updated list of lists with axes equalized, final axes list
    :rtype: list[list[CIUAnalysisObj]], output_axes_list
    """
    flat_cl_input_list = [x for cl_input_list in cl_input_list_by_label for x in cl_input_list]
    flat_obj_list = []
    for cl_input in flat_cl_input_list:
        for subclass_label, subclass_obj in cl_input.subclass_dict.items():
            flat_obj_list.append(subclass_obj)

    # check axes for cropping (ensure that region of interest is equal)
    crop_vals, axes_spacings = check_axes_crop(flat_obj_list)
    final_axes = check_axes_interp(crop_vals, axes_spacings)

    for cl_input_list in cl_input_list_by_label:
        for cl_input in cl_input_list:
            for subclass_label, subclass_obj in cl_input.subclass_dict.items():
                try:
                    subclass_obj, adj_flag = equalize_obj(subclass_obj, final_axes)
                except ValueError as err:
                    # Raise error and cancel classification. Don't want to try to guarantee that data will still match, and user shouldn't be fitting things that don't overlap anyway.
                    raise ValueError(*err.args)

    return cl_input_list_by_label, final_axes


def equalize_unk_axes_clinput(flat_unknown_list, final_axes, scheme_feats_list, gaussian_mode=False):
    """
    Specialized equalization method for unknown classification data. Equalizes DT axis by cropping,
    and CHECKS cv axis (does NOT equalize/reduce) to ensure all required feature inputs are present.
    :param flat_unknown_list: list of CIUAnalysis objects to be equalized
    :type flat_unknown_list: list[Classification.ClInput]
    :param final_axes: crop values from the Scheme to equalize to
    :param scheme_feats_list: List of CFeature containers from the classification scheme being used
    :type scheme_feats_list: list[Classification.CFeature]
    :param gaussian_mode: (optional) if True, does NOT adjust DT axis (only CV), as Gaussian fitted data doesn't need DT adjustment
    :return: updated object list with axes cropped (if needed)
    :rtype: list[CIUAnalysisObj]
    """
    valid_inputs = []
    # adjust ALL files to the same final axes
    for cl_input in flat_unknown_list:
        for subclass_label, analysis_obj in cl_input.subclass_dict.items():
            # no alterations to "DT" data if using Gaussian mode, as the raw Gaussian data will be gathered elsewhere
            if not gaussian_mode:
                # interpolate DT axis ONLY - leave CV axis alone by passing the existing CV axis as the "new" axis
                adjusted_axes = [final_axes[0], analysis_obj.axes[1]]
                try:
                    analysis_obj, adj_flag = equalize_obj(analysis_obj, adjusted_axes)
                except ValueError as err:
                    flat_unknown_list.remove(cl_input)
                    messagebox.showerror('Axes Could Not Be Equalized',
                                         message='{}. \nProblem: {}. This file will be skipped Press OK to continue'.format(*err.args))

        # Check that all required feature CVs (including subclass if required) are present
        valid_input = True
        for feature in scheme_feats_list:
            # Get CIU container for the feature's subclass label
            if feature.subclass_label is not None:
                try:
                    subclass_obj = cl_input.subclass_dict[feature.subclass_label]
                    filename = subclass_obj.short_filename
                    cv_axis = subclass_obj.axes[1]
                except KeyError:
                    valid_input = False
                    cv_axis = []
                    filename = '<No Subclass File!>'
                    logger.error(
                        'Unknown replicate {} does not have the required subclass label {} from this classification scheme. Classification cannot be performed'.format(cl_input.name, feature.subclass_label))
                    # raise ValueError(
                    #     'Unknown replicate {} does not have the required subclass label {} from this classification scheme. Classification cannot be performed'.format(cl_input.name, feature.subclass_label))
            else:
                subclass_obj = cl_input.subclass_dict['0']
                filename = subclass_obj.short_filename
                cv_axis = subclass_obj.axes[1]

            # Confirm that the Feature CV is in the subclass container's CV axis
            if feature.cv not in cv_axis:
                valid_input = False
                logger.error(
                    'CIU dataset {} does not have CV value {} required for this classification scheme. Classification cannot be performed'.format(filename, feature.cv))
                # raise ValueError(
                #     'CIU dataset {} does not have CV value {} required for this classification scheme. Classification cannot be performed'.format(subclass_obj.short_filename, feature.cv))
        if valid_input:
            valid_inputs.append(cl_input)
    return valid_inputs


class CropUI(object):
    """
    Simple dialog with several fields build with Pygubu for inputting crop values
    """
    def __init__(self, init_axes, crop_ui_file):
        # Get crop input from the Crop_vals UI form
        self.builder = pygubu.Builder()

        # load the UI file
        self.builder.add_from_file(crop_ui_file)
        # create widget using provided root (Tk) window
        self.mainwindow = self.builder.get_object('Crop_toplevel')
        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        callbacks = {
            'on_button_cancel_clicked': self.on_button_cancel_clicked,
            'on_button_crop_clicked': self.on_button_crop_clicked
        }
        self.builder.connect_callbacks(callbacks)
        self.crop_vals = []

        self.init_values(init_axes)

    def run(self):
        """
        Run the UI and return the output values
        :return: List of crop values [dt low, dt high, cv low, cv high]
        """
        self.builder.get_object('Crop_toplevel').grab_set()     # prevent users from clicking other stuff while crop is active
        self.mainwindow.mainloop()
        self.builder.get_object('Crop_toplevel').grab_release()
        return self.return_values()

    def on_close_window(self):
        """
        Quit the mainwindow to stop the mainloop and get it to return
        the crop values, then destroy it to remove it from screen.
        :return: the provided crop values, or None if none were provided
        """
        self.mainwindow.quit()
        self.mainwindow.destroy()
        return self.return_values()

    def init_values(self, axes):
        """
        Display starting axes values in the object for the user to edit
        :param axes: list of [dt_axis, cv_axis] (each their own list)
        :return: void
        """
        dt_axis = axes[0]
        cv_axis = axes[1]

        self.builder.get_object('Entry_dt_start').delete(0, tk.END)
        self.builder.get_object('Entry_dt_start').insert(0, dt_axis[0])
        self.builder.get_object('Entry_dt_end').delete(0, tk.END)
        self.builder.get_object('Entry_dt_end').insert(0, dt_axis[len(dt_axis) - 1])

        self.builder.get_object('Entry_cv_start').delete(0, tk.END)
        self.builder.get_object('Entry_cv_start').insert(0, cv_axis[0])
        self.builder.get_object('Entry_cv_end').delete(0, tk.END)
        self.builder.get_object('Entry_cv_end').insert(0, cv_axis[len(cv_axis) - 1])

    def on_button_cancel_clicked(self):
        """
        Close the UI window and return None
        :return: will return None from on_close_window
        """
        return self.on_close_window()

    def on_button_crop_clicked(self):
        """
        Return the cropping values entered in the fields, after checking for invalid values.
        If any values are invalid, returns without setting the crop_vals list
        :return: sets crop_vals list [dt_low, dt_high, cv_low, cv_high] to the object's field for retrieval
        """
        dt_low = self.builder.get_object('Entry_dt_start').get()
        dt_high = self.builder.get_object('Entry_dt_end').get()
        cv_low = self.builder.get_object('Entry_cv_start').get()
        cv_high = self.builder.get_object('Entry_cv_end').get()

        try:
            dt_low = float(dt_low)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid starting DT: must be a number (decimals OK)')
            return -1
        try:
            dt_high = float(dt_high)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid ending DT: must be a number (decimals OK)')
            return -1
        try:
            cv_low = float(cv_low)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid starting CV: must be a number (decimals OK)')
            return -1
        try:
            cv_high = float(cv_high)
        except ValueError:
            messagebox.showwarning('Error', 'Invalid ending CV: must be a number (decimals OK)')
            return -1

        # if all values are valid, save the list and close the window
        self.crop_vals = [dt_low, dt_high, cv_low, cv_high]
        # self.crop_vals = [dt_low, dt_high, cv_low, cv_high, dt_length, cv_length]
        return self.on_close_window()

    def return_values(self):
        """
        Returns the crop_values if they are specified
        :return: crop_vals list [dt_low, dt_high, cv_low, cv_high] or None if they're not set
        """
        if len(self.crop_vals) > 0:
            return self.crop_vals
        else:
            return None


def run_crop_ui(axes, crop_ui_file):
    """
    Run the crop UI widget from PyGuBu to get cropping values and return them
    :param axes: starting axes to display in the widget
    :param crop_ui_file: full system path to crop_vals.ui file
    :return: List of crop values [dt_low, dt_high, cv_low, cv_high], or None if none were specified or
    the user hit the 'cancel' button
    """
    crop_app = CropUI(axes, crop_ui_file)
    crop_vals = crop_app.run()
    return crop_vals



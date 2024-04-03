"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Module with updated versions of the original CIUSuite modules to preserve (and improve)
functionality of the original CIUSuite. Designed to operate in the framework of CIUSuite2,
with CIUAnalysisObj objects providing the primary basis for handling data.
"""
import numpy as np
import os
import scipy.interpolate
from tkinter import messagebox
import logging

from CIU_analysis_obj import CIUAnalysisObj
from CIU_Params import Parameters
from CIU_raw import CIURaw

# use a non-interactive backend to prevent background windows from getting created and causing error messages
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
logger = logging.getLogger('main')


def ciu_plot(analysis_obj, params_obj, output_dir):
    """
    Generate a CIU plot in the provided directory
    :param analysis_obj: preprocessed CIUAnalysisObj with data to be plotted
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param output_dir: directory in which to save the plot
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    else:
        plot_title = ''

    # use filename and provided extension as output path with specific figure size and DPI
    output_title = analysis_obj.short_filename
    output_path = os.path.join(output_dir, output_title + params_obj.plot_02_extension)

    # Generate contours. Aiming for levels of ~ 0 - 1.0 in steps of 0.01, but merge the bottom 10 levels together for easier editing.
    levels = get_contour_levels(analysis_obj.ciu_data)
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, levels=levels, cmap=params_obj.ciuplot_cmap_override)
    # plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, 100, cmap=params_obj.ciuplot_cmap_override)

    # set x/y limits if applicable, allowing for partial limits
    if params_obj.plot_16_xlim_lower is not None:
        if params_obj.plot_17_xlim_upper is not None:
            plt.xlim((params_obj.plot_16_xlim_lower, params_obj.plot_17_xlim_upper))
        else:
            plt.xlim(xmin=params_obj.plot_16_xlim_lower)
    elif params_obj.plot_17_xlim_upper is not None:
        plt.xlim(xmax=params_obj.plot_17_xlim_upper)
    if params_obj.plot_18_ylim_lower is not None:
        if params_obj.plot_19_ylim_upper is not None:
            plt.ylim((params_obj.plot_18_ylim_lower, params_obj.plot_19_ylim_upper))
        else:
            plt.ylim(ymin=params_obj.plot_18_ylim_lower)
    elif params_obj.plot_19_ylim_upper is not None:
        plt.ylim(ymax=params_obj.plot_19_ylim_upper)

    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()

    return 'returning a value so that the mainloop doesnt stop'


def get_contour_levels(ciu_data, merge_cutoff=10, num_contours=100):
    """
    Generates contours for CIU plots with the bottom 10% of data merged into a single contour for
    easier post-processing. Detects min/max values to ensure that contours match the data.
    :param ciu_data: analysis_obj.ciu_data - 2D numpy array of DT/CV ciu data
    :param merge_cutoff: Percent (int) value below which to merge all contours together. Default 10% for data scaled to 100
    :param num_contours: approximate number of contour levels to generate. Default 100
    :return: list of ints (contour levels). Pass returned list directly to pyplot.contourf as 'levels' arg
    """
    possible_steps = np.asarray([0.001, 0.01, 0.1, 1, 10, 100])

    max_val = int(round(np.max(ciu_data) * 100)) + 1  # +1 and -1 for max/min vals are to prevent white spots from appearing after rounding
    min_val = int(round(np.min(ciu_data) * 100)) - 1
    step = (max_val - min_val) / float(num_contours)      # round magnitude step to get close to 100 contours/bins total
    step = possible_steps[(np.abs(possible_steps - step)).argmin()]

    # ensure the min value in the data is above the merge cutoff
    if merge_cutoff < min_val:
        merge_cutoff = min_val
        levels = [x for x in np.arange(merge_cutoff, max_val, step)]
    else:
        levels = [x for x in np.arange(merge_cutoff, max_val, step)]
        levels.insert(0, min_val)
    levels = [x / 100.0 for x in levels]  # convert from integers (percent) back to float (relative intensity)
    return levels


def rmsd_difference(ciu_matrix_1, ciu_matrix_2, noise_cutoff):
    """
        Compute RMSD between an arbitrary number of fingerprints. Returns RMSD and either pairwise difference
    matrix or standard deviation matrix depending on number of files passed.
    :param noise_cutoff: minimum relative intensity to consider (all values below cutoff set to 0)
    :param ciu_matrix_1: 2D numpy array of floats
    :param ciu_matrix_2: 2D numpy array of floats
    :return: difference matrix (ndarray), rmsd (float) in percent
    """
    # First, noise filter data by setting anything below noise cutoff (relative intensity) to 0
    ciu_matrix_1[ciu_matrix_1 < noise_cutoff] = 0
    ciu_matrix_2[ciu_matrix_2 < noise_cutoff] = 0

    # Calculate difference matrix
    dif = ciu_matrix_1 - ciu_matrix_2
    num_values = np.count_nonzero(dif)

    rmsd = (np.sum(dif ** 2) / num_values) ** 0.5 * 100
    return dif, rmsd


def rmsd_plot(difference_matrix, axes, rtext, outputdir, params_obj,
              file1, file2, blue_label=None, red_label=None, filename_append=''):
    """
    Make a CIUSuite comparison RMSD plot with provided parameters
    :param difference_matrix: 2D ndarray with differences to plot
    :param axes: [DT axis, CV axis] - axes labels to use for plot
    :param rtext: RMSD label to apply to plot
    :param outputdir: directory in which to save plot
    :param file1: filename of first file
    :param file2: filename of second file
    :param blue_label: (optional) custom colorbar label for the second file
    :param red_label: (optional) custom colorbar label for the first file
    :param params_obj: Parameters container with plotting information
    :type params_obj: Parameters
    :param filename_append: optional string to append to output title immediately before the file extenstion
    :return: void
    """
    # initial plot setup
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = 'Red: {} \nBlue: {}'.format(file1, file2)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')

    # scale plot to max difference value in high contrast mode, or max value (1) in default mode
    if params_obj.compare_3_high_contrast:
        rmsd_plot_scaling = np.amax(difference_matrix, axis=None)
        rmsd_plot_scaling = round(rmsd_plot_scaling, 2)  # round to nearest hundredth
        rmsd_plot_scaling += 0.01  # ensure that we didn't round down below the max value by adding 1/100
    else:
        rmsd_plot_scaling = 1
    contour_scaling = np.linspace(-rmsd_plot_scaling, rmsd_plot_scaling, 50, endpoint=True)
    colorbar_scaling = np.linspace(-rmsd_plot_scaling, rmsd_plot_scaling, 3, endpoint=True)

    # make the RMSD contour plot
    plt.contourf(axes[1], axes[0], difference_matrix, contour_scaling, cmap="bwr", ticks="none")
    plt.tick_params(axis='x', which='both', bottom='off', top='off', left='off', right='off')
    plt.tick_params(axis='y', which='both', bottom='off', top='off', left='off', right='off')

    # plot labels and legends
    if params_obj.plot_07_show_legend:
        plt.annotate(rtext, xy=(200, 10), xycoords='axes points', fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_06_show_colorbar:
        colorbar = plt.colorbar(ticks=colorbar_scaling)
        if blue_label is not None and red_label is not None:
            colorbar.ax.set_yticklabels([red_label, 'Equal', blue_label])
        colorbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)

    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    # set x/y limits if applicable, allowing for partial limits
    if params_obj.plot_16_xlim_lower is not None:
        if params_obj.plot_17_xlim_upper is not None:
            plt.xlim((params_obj.plot_16_xlim_lower, params_obj.plot_17_xlim_upper))
        else:
            plt.xlim(xmin=params_obj.plot_16_xlim_lower)
    elif params_obj.plot_17_xlim_upper is not None:
        plt.xlim(xmax=params_obj.plot_17_xlim_upper)
    if params_obj.plot_18_ylim_lower is not None:
        if params_obj.plot_19_ylim_upper is not None:
            plt.ylim((params_obj.plot_18_ylim_lower, params_obj.plot_19_ylim_upper))
        else:
            plt.ylim(ymin=params_obj.plot_18_ylim_lower)
    elif params_obj.plot_19_ylim_upper is not None:
        plt.ylim(ymax=params_obj.plot_19_ylim_upper)

    # save and close
    output_path = os.path.join(outputdir, '{}-{}{}{}'.format(file1, file2, filename_append, params_obj.plot_02_extension))
    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()


def std_dev_plot(analysis_obj, std_dev_matrix, pairwise_rmsds, params_obj, output_dir):
    """
    Plot the original CIUSuite-style standard deviation plot from averaged fingerprints
    :param analysis_obj: Averaged analysis object (for axes and filename)
    :type analysis_obj: CIUAnalysisObj
    :param std_dev_matrix: standard deviation data in same shape as analysis_obj.ciu_data
    :param params_obj: Parameters information
    :type params_obj: Parameters
    :param pairwise_rmsds: list of pairwise RMSD values from replicate analyses for plot annotation. Set to [] to ignore
    :param output_dir: directory in which to save output
    :return: void
    """
    # initial plot setup
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # save filename as plot title, unless a specific title is provided
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename + ' Replicate Standard Deviation Plot'
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')

    # plot standard deviation contour plot, normalized to the maximum std dev observed
    max_std_dev = np.max(std_dev_matrix)
    cutoff = int(round(0.05 * max_std_dev * 100))   # combine lowest 5% into single contour
    contour_scale = get_contour_levels(std_dev_matrix, merge_cutoff=cutoff)
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], std_dev_matrix, contour_scale, cmap=params_obj.plot_01_cmap)

    # set x/y limits if applicable, allowing for partial limits
    if params_obj.plot_16_xlim_lower is not None:
        if params_obj.plot_17_xlim_upper is not None:
            plt.xlim((params_obj.plot_16_xlim_lower, params_obj.plot_17_xlim_upper))
        else:
            plt.xlim(xmin=params_obj.plot_16_xlim_lower)
    elif params_obj.plot_17_xlim_upper is not None:
        plt.xlim(xmax=params_obj.plot_17_xlim_upper)
    if params_obj.plot_18_ylim_lower is not None:
        if params_obj.plot_19_ylim_upper is not None:
            plt.ylim((params_obj.plot_18_ylim_lower, params_obj.plot_19_ylim_upper))
        else:
            plt.ylim(ymin=params_obj.plot_18_ylim_lower)
    elif params_obj.plot_19_ylim_upper is not None:
        plt.ylim(ymax=params_obj.plot_19_ylim_upper)

    # plot desired labels and legends
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_06_show_colorbar:
        colorbar_scale = np.linspace(0, max_std_dev, 6, endpoint=True)   # plot colorbar
        cbar = plt.colorbar(ticks=colorbar_scale, format='%.2f')
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        mean_rmsd = np.mean(pairwise_rmsds)
        std_text = 'Average Pairwise RMSD: {:.2f}'.format(mean_rmsd)
        plt.annotate(std_text, xy=(150, 10), xycoords='axes points', fontsize=params_obj.plot_13_font_size)

    # save and close
    output_path = os.path.join(output_dir, analysis_obj.short_filename + '_stdev' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()


def average_ciu(analysis_obj_list):
    """
    Generate and save replicate object (a CIUAnalysisObj with averaged ciu_data and a list
    of raw_objs) that can be used for further analysis
    :param analysis_obj_list: list of CIUAnalysisObj's to average
    :type analysis_obj_list: list[CIUAnalysisObj]
    :rtype: CIUAnalysisObj, numpy array of std data
    :return: averaged analysis object, standard deviation matrix, and replicate rmsd
    """
    raw_obj_list = [analysis_obj.raw_obj for analysis_obj in analysis_obj_list]
    ciu_data_list = [analysis_obj.ciu_data for analysis_obj in analysis_obj_list]

    # generate the average object and averaged Raw container
    avg_data = np.mean(ciu_data_list, axis=0)
    std_data = np.std(ciu_data_list, axis=0)

    # Avg raw data uses the averaged CIU data (NOT raw) to prevent crashes on uneven axes.
    raw_filename = raw_obj_list[0].filename.rstrip('_raw.csv') + '_Avg_raw.csv'
    avg_raw_obj = CIURaw(avg_data, analysis_obj_list[0].axes[0], analysis_obj_list[0].axes[1], raw_filename)

    averaged_obj = CIUAnalysisObj(avg_raw_obj, avg_data, analysis_obj_list[0].axes, analysis_obj_list[0].params)
    averaged_obj.raw_obj_list = raw_obj_list

    return averaged_obj, std_data


def get_pairwise_rmsds(analysis_obj_list, params_obj):
    """
    Helper method to compute pairwise RMSD values for each replicate in an averaging analysis. Also
    generates strings for saving to CSV.
    :param analysis_obj_list: list of analysis objects being averaged
    :type analysis_obj_list: list[CIUAnalysisObj]
    :param params_obj: parameter container
    :type params_obj: Parameters
    :return: list of RMSD values (floats), list of RMSD strings
    """
    f1_index = 0
    rmsd_strings = ''
    rmsds = []
    for analysis_obj in analysis_obj_list:
        f2_index = 0
        while f2_index < len(analysis_obj_list):
            # skip reverse and self comparisons
            if f2_index >= f1_index:
                f2_index += 1
                continue

            rmsd = compare_basic_raw(analysis_obj, analysis_obj_list[f2_index], params_obj, outputdir='', no_plots=True)
            rmsds.append(rmsd)
            rmsd_strings += '{},{},{:.2f}\n'.format(analysis_obj.short_filename, analysis_obj_list[f2_index].short_filename, rmsd)
            f2_index += 1
        f1_index += 1
    return rmsds, rmsd_strings


def save_avg_rmsd_data(analysis_obj_list, params_obj, avg_filename, output_dir):
    """
    Generate a CSV file with information about the averaged file, including the input files,
    total replicate RMSD, and pairwise RMSDs from each file.
    NOTE: should be called AFTER the averaged object has been saved with its updated filename
    :param analysis_obj_list: list of analysis objects averaged
    :param params_obj: parameters container
    :type params_obj: Parameters
    :param avg_filename: filename of the average .ciu file
    :param output_dir: directory in which to save output.
    :return: void
    """
    # Determine pairwise comparison RMSDs for all file and save
    rmsds, rmsd_strings = get_pairwise_rmsds(analysis_obj_list, params_obj)

    # Format string output
    output_string = ''
    output_string += 'Avg File:,{}\n'.format(avg_filename)
    output_string += 'Input Files:,{}\n'.format(','.join([x.short_filename for x in analysis_obj_list]))
    output_string += 'Replicate RMSD (%):,{:.2f}\n'.format(np.mean(rmsds))
    output_string += 'RMSD Std Dev:,{:.2f}\n'.format(np.std(rmsds))
    output_string += 'Pairwise RMSDs:\n'
    output_string += 'File 1,File 2,RMSD (%)\n'
    output_string += rmsd_strings

    # save to file while catching permission errors
    save_path = os.path.join(output_dir, avg_filename + '_RMSDs.csv')
    try:
        with open(save_path, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(save_path))
        with open(save_path, 'w') as outfile:
            outfile.write(output_string)


def interpolate_axes(axis1, axis2, num_bins):
    """
    Method to interpolate different length axes to the same length/scale to enable subtractive
    (or other) comparison of data. The interpolated axis will have twice the number of bins as the
    larger of the initial axes and will go from the minimum to maximum value of both axes to ensure
    no data is missed.
    :param axis1: numpy array (1D) of values
    :param axis2: numpy array (1D) of values
    :param num_bins: number of bins onto which to interpolate
    :return: interpolated axis (numpy array)
    """
    # Determine axis sizes and ranges
    min_val = np.min([np.min(axis1), np.min(axis2)])  # minimum value between both axes
    max_val = np.max([np.max(axis1), np.max(axis2)])  # max value between both axes
    return np.linspace(min_val, max_val, num_bins)


def compare_basic_raw(analysis_obj1, analysis_obj2, params_obj, outputdir, no_plots=False):
    """
    Basic CIU comparison between two raw files. Prints compare plot and csv to output dir.
    Adapted to use CIUAnalysis objects as inputs, so now assumes data will be preprocessed prior
    to running this method.
    :param analysis_obj1: preprocessed (and loaded) CIUAnalysisObj with data to be used as file 1
    :type analysis_obj1: CIUAnalysisObj
    :param analysis_obj2: preprocessed CIUAnalysisObj with data to be used as file 2
    :type analysis_obj2: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :param no_plots: Turn off plot outputs
    :return: RMSD value (writes plot to output directory)
    """
    norm_data_1 = analysis_obj1.ciu_data
    norm_data_2 = analysis_obj2.ciu_data
    axes = analysis_obj1.axes

    interp_flag = False
    dt_axis = analysis_obj1.axes[0]
    cv_axis = analysis_obj1.axes[1]
    # ensure that the data are the same in both dimensions, and interpolate if not matched in either
    if not len(analysis_obj1.axes[0]) == len(analysis_obj2.axes[0]) or not len(analysis_obj1.axes[1]) == len(analysis_obj2.axes[1]):
        interp_flag = True
        logger.warning('axes in files {}, {} did not match; interpolating to compare'.format(
            os.path.basename(analysis_obj1.filename),
            os.path.basename(analysis_obj2.filename)))
        num_bins_dt = np.max([len(analysis_obj1.axes[0]), len(analysis_obj2.axes[0])])  # length of the DT axis
        dt_axis = interpolate_axes(analysis_obj1.axes[0], analysis_obj2.axes[0], num_bins_dt)
        num_bins_cv = np.max([len(analysis_obj1.axes[1]), len(analysis_obj2.axes[1])])
        cv_axis = interpolate_axes(analysis_obj1.axes[1], analysis_obj2.axes[1], num_bins_cv)

    if interp_flag:
        # interpolate the original CIU data from each object onto the new (matched) axes
        interp_fn1 = scipy.interpolate.interp2d(analysis_obj1.axes[1],
                                                analysis_obj1.axes[0],
                                                analysis_obj1.ciu_data)
        interp_fn2 = scipy.interpolate.interp2d(analysis_obj2.axes[1],
                                                analysis_obj2.axes[0],
                                                analysis_obj2.ciu_data)
        norm_data_1 = interp_fn1(cv_axis, dt_axis)
        norm_data_2 = interp_fn2(cv_axis, dt_axis)
        axes = [dt_axis, cv_axis]

    # dif, rmsd = rmsd_difference(norm_data_1, norm_data_2)
    dif, rmsd = rmsd_difference(norm_data_1, norm_data_2, params_obj.compare_4_int_cutoff)

    if not no_plots:
        rtext = "RMSD = " + '%2.2f' % rmsd
        rmsd_plot(difference_matrix=dif,
                  axes=axes,
                  file1=analysis_obj1.short_filename,
                  file2=analysis_obj2.short_filename,
                  rtext=rtext,
                  outputdir=outputdir,
                  params_obj=params_obj,
                  blue_label=params_obj.compare_1_custom_blue,
                  red_label=params_obj.compare_2_custom_red)

    # if params_obj.output_1_save_csv:
    #     title = '{} - {}'.format(analysis_obj1.short_filename,
    #                              analysis_obj2.short_filename)
    #     save_path = os.path.join(outputdir, title)
    #     save_path += '_raw.csv'
    #     write_ciu_csv(save_path, dif, axes)

    return rmsd


def delta_dt(analysis_obj):
    """
    Converts a CIU dataset to delta-drift time by moving the x-axis (DT) so that the max value
    (or centroid if gaussian fitting has been performed) of the first CV column is at 0.
    :param analysis_obj: CIUAnalysisObj to be shifted
    :type analysis_obj: CIUAnalysisObj
    :rtype: CIUAnalysisObj
    :return: new CIUAnalysisObj with shifted data
    """
    # gaussian fitting not done, simply use max of first column
    first_col = analysis_obj.ciu_data[:, 0]
    index_of_max = np.argmax(first_col)
    centroid_xval = analysis_obj.axes[0][index_of_max]

    # Shift the DT axis (ONLY) so that the max value of the column is at DT = 0
    old_dt_axis = analysis_obj.axes[0]
    new_dt_axis = old_dt_axis - centroid_xval
    new_axes = [new_dt_axis, analysis_obj.axes[1]]

    analysis_obj.axes = new_axes
    return analysis_obj


def write_ciu_csv(save_path, ciu_data, axes=None):
    """
    Method to write an _raw.csv file for CIU data. If 'axes' is provided, assumes that the ciu_data
    array does NOT contain axes and if 'axes' is None, assumes ciu_data contains axes.
    :param save_path: Full path to save location (SHOULD end in _raw.csv)
    :param ciu_data: 2D numpy array containing CIU data in standard format (rows = DT bins, cols = CV)
    :param axes: (optional) axes labels, provided as (row axis, col axis). if provided,
    assumes the data array does not contain axes labels.
    :return: void
    """
    output_string = ''
    if axes is not None:
        # write axes first if they're provided
        args = ['{}'.format(x) for x in axes[1]]    # get the cv-axis now to write to the header
        line = ','.join(args)
        line = ',' + line
        output_string += line + '\n'

        index = 0
        for row in ciu_data:
            # insert the axis label at the start of each row
            args = ['{}'.format(x) for x in row]
            args.insert(0, str(axes[0][index]))
            index += 1
            line = ','.join(args)
            output_string += line + '\n'
    else:
        # axes are included, so just write everything to file with comma separation
        args = ['{}'.format(x) for x in ciu_data]
        line = ','.join(args)
        output_string += line + '\n'

    # save to file while catching permission errors
    try:
        with open(save_path, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(save_path))
        with open(save_path, 'w') as outfile:
            outfile.write(output_string)

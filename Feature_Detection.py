"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Module for feature detection. Relies on CIUAnalysisObj from Gaussian fitting module
Author: DP
Date: 10/10/2017
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import scipy.interpolate
import os
import math
import logging
import Raw_Processing
import Gaussian_Fitting
import Original_CIU
from tkinter import messagebox
import warnings

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj
    from CIU_Params import Parameters

warnings.filterwarnings('ignore')
logger = logging.getLogger('main')
TRANS_COLOR_DICT = {6: 'white',
                    0: 'red',
                    5: 'blue',
                    1: 'green',
                    4: 'yellow',
                    2: 'orange',
                    3: 'purple'}


def feature_detect_col_max(analysis_obj, params_obj):
    """
    Uses max values of each CV column to assign flat features to data. Should be roughly
    analogous to the changepoint detection + flat features from column maxes in CIU-50 analysis,
    but without reliance on the (somewhat fickle) changepoint detection
    :param analysis_obj: CIUAnalysisObj container for CIU data
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: analysis object with features saved
    """
    features = []

    # interpolate CV axis if the spacing is not equal
    cv_axis = analysis_obj.axes[1]
    bin_spacings = [cv_axis[x + 1] - cv_axis[x] for x in range(len(cv_axis) - 1)]
    unique_spacings = set(bin_spacings)
    if len(unique_spacings) > 1:
        # uneven CV spacing - interpolate axis to even spacing (smallest previous bin spacing) before analysis
        new_num_bins = len(np.arange(cv_axis[0], cv_axis[-1], min(unique_spacings))) + 1
        cv_axis = np.linspace(cv_axis[0], cv_axis[-1], new_num_bins)
        analysis_obj = Raw_Processing.interpolate_axes(analysis_obj, [analysis_obj.axes[0], cv_axis])
        logger.warning('NOTE: CV axis in file {} was not evenly spaced; Feature Detection requires even spacing. Axis has been interpolated to fit. Use "Restore Original Data" button to undo interpolation'.format(analysis_obj.short_filename))

    # compute width tolerance in DT units, CV gap in bins (NOT cv axis units)
    width_tol_dt = params_obj.feature_t2_2_width_tol  # * analysis_obj.bin_spacing
    cv_spacing = analysis_obj.axes[1][1] - analysis_obj.axes[1][0]
    gap_tol_cv = params_obj.feature_t2_3_ciu50_gap_tol * cv_spacing

    # Search each gaussian for features it matches (based on centroid)
    for cv_index, col_max_dt in enumerate(analysis_obj.col_max_dts):
        # check if any current features will accept this max value
        found_feature = False
        for feature in features:
            if feature.accept_centroid(col_max_dt, width_tol_dt, cv_axis[cv_index], gap_tol_cv, cv_spacing):
                feature.cvs.append(cv_axis[cv_index])
                feature.dt_max_vals.append(col_max_dt)
                feature.cv_indices.append(cv_index)

                found_feature = True
                break

        if not found_feature:
            # no feature was found for this Gaussian, so create a new feature
            new_feature = Feature(gaussian_bool=False)
            new_feature.cvs.append(cv_axis[cv_index])
            new_feature.dt_max_vals.append(col_max_dt)
            new_feature.cv_indices.append(cv_index)
            features.append(new_feature)

    # filter features to remove 'loners' without a sufficient number of points
    filtered_features = filter_features(features, params_obj.feature_t2_1_min_length, mode='changept')

    # exclude features to only include the features with the highest and lowest median
    if params_obj.feature_t3_1_exclude_feature:
        medians = list(map(lambda x: x.get_median(), filtered_features))
        filtered_features = list(filter(lambda x: x.get_median() == max(medians) or x.get_median() == min(medians), filtered_features))

    analysis_obj.features_changept = filtered_features
    return analysis_obj


def feature_detect_gaussians(analysis_obj, params_obj):
    """
    Uses fitted (and filtered) multi-gaussians to assign flat features to data. Should be roughly
    analogous to the changepoint detection + flat features from column maxes in CIU-50 analysis,
    but using gaussian data enables seeing all features instead only the most intense one(s).
    Features returned will be gap-filled (if specified) and in order. They may NOT cover every CV
    in the CV axis and they MAY include data not at column max - those need to be adjusted
    for classification and CIU50 analysis, respectively.
    :param analysis_obj: CIUAnalysisObj with Gaussians previously fitted
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: CIUAnalysisObj
    :return: analysis object with features saved
    """
    features = []

    cv_axis = analysis_obj.axes[1]
    bin_spacings = np.around([cv_axis[x + 1] - cv_axis[x] for x in range(len(cv_axis) - 1)], 6)
    unique_spacings = set(bin_spacings)
    if len(unique_spacings) > 1:
        # uneven CV spacing - tell user to interpolate axes and re-do gaussian fitting
        raise ValueError

    # compute width tolerance in DT units and gap tolerance in CV units
    width_tol_dt = params_obj.feature_t2_2_width_tol  # * analysis_obj.bin_spacing
    cv_spacing = analysis_obj.axes[1][1] - analysis_obj.axes[1][0]
    gap_tol_cv = params_obj.feature_t2_3_ciu50_gap_tol * cv_spacing

    # Search each protein gaussian for features it matches (based on centroid)
    for cv_index, protein_gauss_list in enumerate(analysis_obj.raw_protein_gaussians):
        # First, assign protein Gaussians to features
        for gaussian in protein_gauss_list:
            # check if any current features will accept the Gaussian
            found_feature = False
            for feature in features:
                if feature.accept_centroid(gaussian.centroid, width_tol_dt, gaussian.cv, gap_tol_cv, cv_spacing):
                    feature.gaussians.append(gaussian)
                    feature.cvs.append(gaussian.cv)
                    found_feature = True
                    break

            # no feature was found for this Gaussian, so create a new feature
            if not found_feature:
                new_feature = Feature(gaussian_bool=True)
                new_feature.gaussians.append(gaussian)
                features.append(new_feature)

        # After protein features are done, check if any nonprotein peaks match any features that don't already have a protein peak at this CV
        if params_obj.feature_t2_5_gauss_allow_nongauss:
            if analysis_obj.raw_nonprotein_gaussians is not None:
                for nonprot_gaussian in analysis_obj.raw_nonprotein_gaussians[cv_index]:
                    current_cv = nonprot_gaussian.cv
                    for feature in features:
                        protein_cvs = [x.cv for x in feature.gaussians]
                        if current_cv not in protein_cvs:
                            # use 2x feature standard deviation as width tolerance for non-protein peaks (95% conf lvl analogy)
                            nonprot_width_tol = feature.get_std_dev() * 2

                            # this feature does not currently have any entries at this CV value, making it available to add a non-prot peak
                            if feature.accept_centroid(nonprot_gaussian.centroid, nonprot_width_tol, current_cv, gap_tol_cv, cv_spacing):
                                # Change this "non-protein" Gaussian to a protein - likely misassigned
                                nonprot_gaussian.is_protein = True
                                feature.gaussians.append(nonprot_gaussian)

    # perform a second pass to add to features that were created after the CV at which these non-protein peaks were considered
    if params_obj.feature_t2_5_gauss_allow_nongauss:
        if analysis_obj.raw_nonprotein_gaussians is not None:
            for cv_index, protein_gauss_list in reversed(list(enumerate(analysis_obj.raw_protein_gaussians))):
                for nonprot_gaussian in analysis_obj.raw_nonprotein_gaussians[cv_index]:
                    current_cv = nonprot_gaussian.cv
                    for feature in features:
                        protein_cvs = [x.cv for x in feature.gaussians]
                        if current_cv not in protein_cvs:
                            # use 2x feature standard deviation as width tolerance for non-protein peaks (95% conf lvl analogy)
                            nonprot_width_tol = feature.get_std_dev() * 2

                            # this feature does not currently have any entries at this CV value, making it available to add a non-prot peak
                            if feature.accept_centroid(nonprot_gaussian.centroid, nonprot_width_tol, current_cv, gap_tol_cv, cv_spacing):
                                # Change this "non-protein" Gaussian to a protein - likely misassigned
                                if nonprot_gaussian not in feature.gaussians:
                                    # make sure this protein isn't already in this feature
                                    nonprot_gaussian.is_protein = True
                                    feature.gaussians.append(nonprot_gaussian)

    # ensure cvs and Gaussians are sorted in ascending order (only necessary if appending non-protein peaks AND a nonprotein peak is added out of order last)
    for feature in features:
        feature.cvs = sorted(feature.cvs)
        feature.gaussians = sorted(feature.gaussians, key=lambda x: x.cv)

    # filter features to remove 'loners' without a sufficient number of points
    filtered_features = filter_features(features, params_obj.feature_t2_1_min_length, mode='gaussian')

    # fill feature gaps (if specified) and check order
    if params_obj.feature_t2_4_gauss_fill_gaps:
        filtered_features = fill_feature_gaps(filtered_features, cv_spacing)
    filtered_features = check_feature_order(filtered_features)

    # save filtered gaussians into analysis object as feat_protein_gaussians
    analysis_obj.features_gaussian = filtered_features
    assigned_gaussians = gaussians_by_cv_from_feats(filtered_features, cv_axis)
    analysis_obj.feat_protein_gaussians = assigned_gaussians

    return analysis_obj


def gaussians_by_cv_from_feats(feature_list, cv_axis):
    """
    Generate a list of Gaussians by CV from a list of features
    :param feature_list: list of Features
    :type feature_list: list[Feature]
    :param cv_axis: CV axis to ensure Gaussians are placed in the correct location
    :return: list of lists of Gaussian objects at each CV
    """
    gauss_lists_by_cv = [[] for _ in range(len(cv_axis))]

    # place all Gaussians in each in feature into the final list of Gaussians
    for feature in feature_list:
        for gaussian in feature.gaussians:
            insert_index = np.where(cv_axis == gaussian.cv)[0][0]
            gauss_lists_by_cv[insert_index].append(gaussian)

    return gauss_lists_by_cv


def ciu50_main(features_list, analysis_obj, params_obj, outputdir, gaussian_bool):
    """
    Primary feature detection runner method. Calls appropriate sub-methods using data and
    parameters from the passed analysis object
    :param features_list: list of Feature objects to fit transitions between
    :type features_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with initial data processed and parameters
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :param gaussian_bool: (bool) whether to use Gaussian or raw data Features for CIU-50 fitting
    :rtype: CIUAnalysisObj
    :return: updated analysis object with feature detect information saved
    """
    if len(features_list) <= 1:
        logger.warning('Not enough features (<=1) in file {}. No transition analysis performed'.format(
            analysis_obj.short_filename))
        return analysis_obj

    # Adjust features (remove long gaps, check for non-max data) if features are from Gaussian mode
    if gaussian_bool:
        adjusted_features = adjust_gauss_features(features_list, analysis_obj, params_obj)
        adjusted_features = check_feature_order(adjusted_features)
        plot_features(adjusted_features, analysis_obj, params_obj, outputdir, filename_append='_adjusted')
    else:
        adjusted_features = features_list

    # compute transitions
    transitions_list = compute_transitions(analysis_obj, params_obj, adjusted_features, gaussian_bool)
    if len(transitions_list) == 0:
        logger.info('No transitions found for file {}'.format(os.path.basename(analysis_obj.filename).rstrip('.ciu')))

    # generate output plot
    plot_transitions(transitions_list, analysis_obj, params_obj, outputdir)
    return analysis_obj


def compute_transitions(analysis_obj, params_obj, adjusted_features, gaussian_bool):
    """
    Fit logistic/sigmoidal transition functions to the transition between each sequential pair
    of features in the provided gaussian feature list. Saves Transition objects containing combined
    feature pair info, fit, and resulting CIU-50 value.
    :param adjusted_features: List of Features adjusted to only include CV data where col max is close to centroid
    :param analysis_obj: CIU analysis object with gaussian features already prepared
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param gaussian_bool: Gaussian mode (True) or standard (False)
    :return: list of Transition objects (also saves to analysis_obj)
    :rtype: list[Transition]
    """
    # initialize transition fitting information for gaussian feature lists
    for feature in adjusted_features:
        # Get indices of each CV relative to the complete fingerprint and corresponding DT max values
        cv_indices = []
        cv_axis = list(analysis_obj.axes[1])
        for cv in feature.cvs:
            overall_index = cv_axis.index(cv)
            cv_indices.append(overall_index)
        if len(feature.dt_max_vals) == 0:
            dt_max_vals = analysis_obj.col_max_dts[cv_indices[0]: cv_indices[-1] + 1]
        else:
            dt_max_vals = feature.dt_max_vals
        feature.init_feature_data(cv_indices, dt_max_vals)

    # Fit sigmoids for transition calculations
    index = 0
    transition_list = []
    while index < len(adjusted_features) - 1:
        current_transition = Transition(adjusted_features[index],
                                        adjusted_features[index + 1],
                                        analysis_obj,
                                        gaussian_bool)
        # check to make sure this is a transition that should be fitted (upper feature has a col max)
        # if current_transition.check_features(analysis_obj, params_obj):
        current_transition.fit_transition(params_obj)
        transition_list.append(current_transition)
        # else:
        #     print('feature {} never reaches 50% intensity, '
        #           'skipping transition between {} and {}'.format(index + 2, index + 1, index + 2))
        index += 1
    analysis_obj.transitions = transition_list
    return transition_list


def filter_features(features, min_feature_length, mode):
    """
    Remove any features below the specified minimum feature length from the feature list
    :param features: list of Features
    :type features: list[Feature]
    :param min_feature_length: minimum length (number of gaussians) to be included in a feature
    :param mode: gaussian or changepoint: whether features are from gaussian fitting or changept detection
    :return: filtered feature list with too-small features removed
    """
    filtered_list = []
    for feature in features:
        if mode == 'gaussian':
            if len(feature.gaussians) >= min_feature_length:
                filtered_list.append(feature)
        elif mode == 'changept':
            if len(feature.cvs) >= min_feature_length:
                filtered_list.append(feature)
        else:
            logger.error('invalid mode')
    return filtered_list


def fill_feature_gaps(features_list, cv_spacing):
    """
    Assumes that any 'gaps' within features (CVs that lack a centroid within the feature, but are surrounded
    on both sides by correct centroids) are simply missed by the data analysis and not a true lack of signal.
    For use in Gaussian fitting mode only. Fills in the gaps by adding a
    Gaussian at each CV in the gap corresponding to the median centroid/width/amp
    of surrounding feature points.
    :param features_list: list of Features in which to close gaps
    :type features_list: list[Feature]
    :param cv_spacing: spacing between points along CV axis
    :return: updated features list with features edited to have gaps filled
    :rtype: list[Feature]
    """
    for feature in features_list:
        index = 1
        while index < len(feature.cvs):
            current_spacing = feature.cvs[index] - feature.cvs[index - 1]
            if not math.isclose(current_spacing, cv_spacing, rel_tol=1e-5):
                # a gap is present, fill it
                gap_size = int(np.round(current_spacing / cv_spacing)) - 1
                for gap_fill_index in range(0, gap_size):
                    new_index = index + gap_fill_index
                    new_cv = feature.cvs[index - 1] + cv_spacing * (gap_fill_index + 1)
                    feature.cvs.insert(new_index, new_cv)

                    # create a new Gaussian to append here, using data from previous 3 points along the feature
                    if index > 3:
                        new_centroid = np.median([x.centroid for x in feature.gaussians[index - 4: index - 1]])
                        new_width = np.average([x.width for x in feature.gaussians[index - 4: index - 1]])
                        new_amplitude = np.average([x.amplitude for x in feature.gaussians[index - 4: index - 1]])
                    else:
                        # we're early in the feature and there aren't enough previous points for a good median. Simply use the preceeding point
                        new_centroid = feature.gaussians[index - 1].centroid
                        new_width = feature.gaussians[index - 1].width
                        new_amplitude = feature.gaussians[index - 1].amplitude

                    new_gaussian = Gaussian_Fitting.Gaussian(new_amplitude, new_centroid, new_width, new_cv, pcov=None, protein_bool=True)
                    feature.gaussians.insert(new_index, new_gaussian)

            index += 1
    return features_list


def adjust_gauss_features(features_list, analysis_obj, params_obj):
    """
    Run setup method to prepare Gaussian features for transition fitting. Removes any CV values
    from features for which the max DT value at that CV is outside the width tolerance from the
    feature's median DT. This is necessary because Gaussian features start/persist well before/after
    they are the most abundant peak - which can cause bad fitting if incorrect CV's are included.
    :param features_list: list of Features to adjust
    :type features_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with gaussian fitting and gaussian feature detect performed
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :rtype: list[Feature]
    :return: list of adjusted Features
    """
    adjusted_features = []
    cv_spacing = analysis_obj.axes[1][1] - analysis_obj.axes[1][0]

    for index, feature in enumerate(features_list):
        final_cvs = []
        for cv in feature.cvs:
            # check if the ciu_data column max value is appropriate for this feature at this CV
            cv_index = list(analysis_obj.axes[1]).index(cv)
            dt_diff = abs(analysis_obj.col_max_dts[cv_index] - feature.gauss_median_centroid)
            if dt_diff < params_obj.ciu50_t2_3_gauss_width_adj_tol:
                # also check if a gap has formed and exclude features after the gap if so
                if len(final_cvs) > 0:
                    if cv - final_cvs[-1] <= (params_obj.feature_t2_3_ciu50_gap_tol * cv_spacing):
                        # difference is within tolerances; include this CV in the adjusted feature
                        final_cvs.append(cv)
                else:
                    final_cvs.append(cv)

        # initialize the new feature using the CV list (it will only have CV and centroid data)
        if len(final_cvs) > 0:
            adj_feature = Feature(gaussian_bool=True)
            adj_feature.gauss_median_centroid = feature.gauss_median_centroid
            adj_feature.cvs = final_cvs
            adjusted_features.append(adj_feature)
            adj_feature.gaussians = feature.gaussians
        else:
            logger.info('Feature {} (range {}-{}) never reaches max relative intensity, no transition will be fit'.format(index + 1, feature.cvs[0], feature.cvs[-1]))
    return adjusted_features


def check_feature_order(features_list):
    """
    Ensure that features are in a reasonable order for transition fitting. Specifically,
    make sure that the end of feature n+1 does not come before the start of feature n, as this
    will cause transition fitting to crash/fail.
    NOTE: initial fitting of features places them in order of starting CV, ensuring that this check
    is not needed. However, after adjusting Gaussian features it is possible to get out of order, making
    this check necessary for Gaussian features only.
    :param features_list: list of Feature objects to sort
    :return: sorted features list, swapping ONLY features that are completely out of order as described above
    """
    if len(features_list) == 0:
        # no features detected, likely because filter settings were too strict. return empty list
        return features_list

    new_list = [features_list[0]]
    index = 1
    while index < len(features_list):
        if new_list[index - 1].cvs[0] > features_list[index].cvs[-1]:
            # the start of the previous feature comes AFTER the end of this feature - swap them
            new_list.insert(index - 1, features_list[index])
            new_list = check_feature_order(new_list)    # Recurse to ensure the reordered feature(s) match previous order
        else:
            # feature in order, add to the new list
            new_list.append(features_list[index])
        index += 1

    return new_list


def plot_features(feature_list, analysis_obj, params_obj, outputdir, filename_append=None):
    """
    Generate a plot of features using previously saved (into the analysis_obj) feature fitting data
    :param feature_list: list of Features to plot
    :type feature_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with fitting data previously saved to obj.features_gaussian
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :param filename_append: additional string to append to filename (optional)
    :return: void
    """
    # initialize plot
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # plot the initial CIU contour plot for reference
    levels = Original_CIU.get_contour_levels(analysis_obj.ciu_data)
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, levels=levels, cmap=params_obj.plot_01_cmap)

    # prepare and plot the actual Features using saved data
    feature_index = 1
    if params_obj.feature_t1_1_ciu50_mode == 'gaussian':
        # plot the raw data to show what was fit
        for feature in feature_list:
            for gaussian in feature.gaussians:
                plt.plot(gaussian.cv, gaussian.centroid, 'wo', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')

        for feature in feature_list:
            feature_x = feature.cvs
            feature_y = [feature.gauss_median_centroid for _ in feature.cvs]
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            feature.get_median()))
            feature_index += 1
            plt.setp(lines, linewidth=3)
    elif params_obj.feature_t1_1_ciu50_mode == 'standard':
        # plot the raw data to show what was fit
        plt.plot(analysis_obj.axes[1], analysis_obj.col_max_dts, 'wo', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')

        for feature in feature_list:
            feature_x = feature.cvs
            feature_y = feature.dt_max_vals
            lines = plt.plot(feature_x, feature_y, label='Feature {} median: {:.2f}'.format(feature_index,
                                                                                            feature.get_median()))
            feature_index += 1
            plt.setp(lines, linewidth=3)
    else:
        logger.error('invalid mode')

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

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

    # save plot
    if filename_append is None:
        output_path = os.path.join(outputdir, analysis_obj.short_filename + '_features' + params_obj.plot_02_extension)
    else:
        output_path = os.path.join(outputdir, analysis_obj.short_filename + filename_append + '_features' + params_obj.plot_02_extension)

    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()


def save_features_main(feature_list, outputpath, filename, mode, concise_mode, combine):
    """
    Method to direct output feature saving to detailed or concise mode
    :param feature_list: list of Feature objects
    :type feature_list: list[Feature]
    :param outputpath: directory in which to save output
    :param filename: short filename to save
    :param mode: gaussian or changepoint
    :param concise_mode: whether to save detailed or concise output (string: 'concise' or 'detailed')
    :param combine: whether to save an output file immediately or return the information as a string
    :return: void or string if using 'combine=True'
    """
    if concise_mode == 'concise':
        output_med, output_cv = print_features_concise(feature_list, outputpath, filename, combine)
    else:
        output_med = print_features_list(feature_list, outputpath, filename, mode, combine)
        output_cv = ''
    return output_med, output_cv


def print_features_list(feature_list, outputpath, filename, mode, combine):
    """
    Write feature information to file, OR return it as a string to be saved into a final file if combining
    :param feature_list: list of Feature objects
    :type feature_list: list[Feature]
    :param filename: short filename to save
    :param outputpath: directory in which to save output
    :param mode: gaussian or changepoint
    :param combine: whether to save an output file immediately or return the information as a string
    :return: void or string if using 'combine=True'
    """
    index = 1
    outputstring = '{}'.format(filename)
    for feature in feature_list:
        if mode == 'gaussian':
            outputstring += ',Feature {},Median centroid:,{:.2f},CV range:,{} - {}\n'.format(index,
                                                                                             feature.get_median(),
                                                                                             feature.cvs[0],
                                                                                             feature.cvs[len(feature.cvs) - 1])
            outputstring += ',CV (V), Amplitude, Centroid, Width\n'
            for gaussian in feature.gaussians:
                outputstring += ',' + gaussian.print_info() + '\n'
        else:
            outputstring += ',Feature {},Median centroid:,{:.2f},CV range:,{} - {}\n'.format(index,
                                                                                             feature.get_median(),
                                                                                             feature.cvs[0],
                                                                                             feature.cvs[len(feature.cvs) - 1])
            outputstring += ',CV (V),Peak Drift Time\n'
            cv_index = 0
            for cv in feature.cvs:
                outputstring += ',{},{:.2f}\n'.format(cv, feature.dt_max_vals[cv_index])
                cv_index += 1

        index += 1

    if not combine:
        try:
            with open(outputpath, 'w') as outfile:
                outfile.write(outputstring)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving',
                                 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(
                                     outputpath))
            with open(outputpath, 'w') as outfile:
                outfile.write(outputstring)
        return ''
    else:
        return outputstring


def print_features_concise(feature_list, outputpath, filename, combine):
    """
    Concise version. Write feature information to file, OR return it as a string to be saved into a final file if combining
    :param feature_list: list of Feature objects
    :type feature_list: list[Feature]
    :param outputpath: directory in which to save output
    :param filename: output filename to save
    :param combine: whether to save an output file immediately or return the information as a string
    :return: void or string if using 'combine=True'
    """
    # assemble concise output info
    median_outputs = '{}'.format(filename)
    cv_range_outputs = '{}'.format(filename)
    for feature in feature_list:
        median_outputs += ',{:.2f}'.format(feature.get_median())
        cv_range_outputs += ',{}-{}'.format(feature.cvs[0], feature.cvs[len(feature.cvs) - 1])
    median_outputs += '\n'
    cv_range_outputs += '\n'

    # save to file immediately if not combining
    if not combine:
        try:
            with open(outputpath, 'w') as outfile:
                outfile.write('Feature Median Centroids\nFilename,Feature 1,Feature 2,Feature 3,(etc)\n')
                outfile.write(median_outputs)
                outfile.write('Feature CV ranges\nFilename,Feature 1,Feature 2,Feature 3,(etc)\n')
                outfile.write(cv_range_outputs)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving',
                                 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(
                                     outputpath))
            with open(outputpath, 'w') as outfile:
                outfile.write('Feature Median Centroids\nFilename,Feature 1,Feature 2,Feature 3,(etc)\n')
                outfile.write(median_outputs)
                outfile.write('Feature CV ranges\nFilename,Feature 1,Feature 2,Feature 3,(etc)\n')
                outfile.write(cv_range_outputs)
        return '', ''
    else:
        # combining - return both outputs to be combined later
        return median_outputs, cv_range_outputs


def save_ciu50_outputs_main(analysis_obj, outputpath, concise_mode, combine=False):
    """
    Method to direct output saving to concise or detailed output handlers for CIU50
    output CSVs
    :param analysis_obj: CIU container with transition information to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param concise_mode: whether to save concise or detailed file (string: 'concise' or 'detailed')
    :param combine: whether to output directly for this file or return a string for combining
    :return: output string if combining or void if not
    :return: string if combine True, void if combine False
    """
    if concise_mode == 'concise':
        output = save_ciu50_short(analysis_obj, outputpath, combine)
    else:
        output = save_ciu50_outputs(analysis_obj, outputpath, combine)
    return output


def save_ciu50_outputs(analysis_obj, outputpath, combine=False):
    """
    Print feature detection outputs to file. Must have feature detection already performed.
    **NOTE: currently, feature plot is still in the feature detect module, but could (should?)
    be moved here eventually.
    :param analysis_obj: CIU container with transition information to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param combine: whether to output directly for this file or return a string for combining
    :return: output string if combining or void if not
    """
    output_name = os.path.join(outputpath, analysis_obj.short_filename + '_CIU50.csv')
    output_string = '{},max DT (ms),min DT (ms),CIU-50 (V),k (steepness),r_squared\n'.format(analysis_obj.short_filename)
    trans_index = 1
    for transition in analysis_obj.transitions:
        output_string += 'transition {} -> {},'.format(trans_index, trans_index + 1)
        output_string += '{:.2f},{:.2f},{:.2f},{:.2f}'.format(*transition.fit_params)
        output_string += ',{:.3f}\n'.format(transition.rsq)
        trans_index += 1

    if combine:
        # return the output string to be written together with many files
        return output_string
    else:
        try:
            with open(output_name, 'w') as outfile:
                outfile.write(output_string)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
            with open(output_name, 'w') as outfile:
                outfile.write(output_string)
        return ''


def save_ciu50_short(analysis_obj, outputpath, combine=False):
    """
    Helper method to also save a shortened version of feature information
    :param analysis_obj: CIU container with transition information to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param combine: If True, return a string to be combined with other files instead of saving to file
    :return: output string if combining or void if not
    """
    output_name = os.path.join(outputpath, analysis_obj.short_filename + '_CIU50.csv')
    output_string = ''

    # assemble the output
    output_string += analysis_obj.short_filename
    for transition in analysis_obj.transitions:
        output_string += ',{:.2f}'.format(transition.fit_params[2])
    output_string += '\n'

    if combine:
        # return the output string to be written together with many files
        return output_string
    else:
        try:
            with open(output_name, 'w') as outfile:
                outfile.write('Filename,CIU50 1,CIU50 2,(etc)\n')
                outfile.write(output_string)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
            with open(output_name, 'w') as outfile:
                outfile.write(output_string)


def plot_transitions(transition_list, analysis_obj, params_obj, outputdir):
    """
    Provide a plot of provided transitions overlaid on top of the CIU contour plot for the
    provided analysis object
    :param transition_list: list of Transitions to plot
    :type transition_list: list[Transition]
    :param analysis_obj: object with CIU data to plot
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with parameter information
    :type params_obj: Parameters
    :param outputdir: directory in which to save output
    :return: void
    """
    plt.clf()
    x_axis = analysis_obj.axes[1]
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # plot the initial CIU contour plot for reference
    levels = Original_CIU.get_contour_levels(analysis_obj.ciu_data)
    plt.contourf(analysis_obj.axes[1], analysis_obj.axes[0], analysis_obj.ciu_data, levels=levels, cmap=params_obj.plot_01_cmap)

    # plot all transitions
    transition_num = 0
    for transition in transition_list:
        # plot markers for the max/average/median values used in fitting for reference
        for index, cv in enumerate(transition.combined_x_axis):
            if params_obj.ciu50_t2_1_centroiding_mode == 'max':
                plt.plot(cv, transition.combined_y_vals[index], 'wo', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
            elif params_obj.ciu50_t2_1_centroiding_mode == 'average':
                plt.plot(cv, transition.combined_y_avg_raw[index], 'wo', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
            elif params_obj.ciu50_t2_1_centroiding_mode == 'median':
                plt.plot(cv, transition.combined_y_median_raw[index], 'wo', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')

        # prepare and plot the actual transition using fitted parameters
        interp_x = np.linspace(x_axis[0], x_axis[len(x_axis) - 1], 200)
        y_fit = logistic_func(interp_x, *transition.fit_params)

        # use different colors for plotting the transition (up to 6 provided)
        if transition_num <= 6:
            trans_line_color = TRANS_COLOR_DICT[transition_num]
        else:
            trans_line_color = TRANS_COLOR_DICT[6]
        trans_plot = plt.plot(interp_x, y_fit, color=trans_line_color, label='CIU50: {:.1f}, r2=: {:.2f}'.format(transition.ciu50, transition.rsq))
        plt.setp(trans_plot, linewidth=2)
        transition_num += 1

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

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_06_show_colorbar:
        cbar = plt.colorbar(ticks=[0, .25, .5, .75, 1])
        cbar.ax.tick_params(labelsize=params_obj.plot_13_font_size)
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    # save plot to file
    filename = analysis_obj.short_filename + '_transition' + params_obj.plot_02_extension
    output_path = os.path.join(outputdir, filename)
    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()


def bin_to_dt(bin_val, min_dt, bin_spacing):
    """
    Convert a bin value to a fingerprint-relative drift time. Adjusts for the minimum DT of
    the fingerprint to give accurate drift axis results. Should NOT be used for conversion
    of absolute bin -> DT
    :param bin_val: (int) bin number to convert to DT space
    :param min_dt: minimum DT of fingerprint
    :param bin_spacing: spacing between DT bins (conversion factor)
    :return: DT in drift axis units
    """
    dt = min_dt + (bin_val - 1) * bin_spacing
    return dt


def bin_to_ms(bin_val, bin_spacing):
    """
    Conversion from a number of bins to the corresponding time in ms (or other drift units)
    for the provided spacing. Differs from bin to dt in that the output time is NOT adjusted
    for the minimum of the fingerprint (an absolute conversion, rather than a fingerprint-relative
    conversion).
    :param bin_val: (int) number of bins to convert
    :param bin_spacing: distance between bins in time units (conversion factor)
    :return: time in drift axis units corresponding to bin val
    """
    dt = bin_val * bin_spacing
    return dt


def logistic_func(x, c, y0, x0, k):
    """
    Generalized logistic function for fitting to feature transitions
    :param x: x value (independent variable)
    :param c: height of the maximum/upper asymptote of the curve
    :param y0: height of the minimum/lower asymptote of the curve
    :param x0: centroid/midpoint of the curve (also CIU-50 value of the transition)
    :param k: steepness of the curve/transition
    :return: y = f(x)
    """
    y = y0 + ((c - y0) / (1 + np.exp(-k * (x - x0))))
    return y


def fit_logistic(x_axis, y_data, guess_center, guess_min, guess_max, steepness_guess):
    """
    Fit a general logistic function (defined above) to data using the SciPy.optimize
    curve_fit module.
    :param x_axis: x data to fit (list or ndarray)
    :param y_data: y data to fit (list of ndarray)
    :param guess_center: initial guess for x0 - should be around the midpoint of the transition
    :param guess_max: initial guess for c - should be around the max y-value for the transition
    :param guess_min: initial guess for y0 - should be around the min y-value for the transition
    :param steepness_guess: initial guess for k. 0.1 works well for shallower transitions and still gets steep ones
    :return: popt, pcov: popt = optimized parameters [c, y0, x0, k] from fitting, pcov = covariance matrix
    """
    # guess initial params: [c, y0, x0, k], default guess k=1
    p0 = [guess_max, guess_min, guess_center, steepness_guess]
    # constrain all parameters to positive values
    fit_bounds_lower = [1e-5, 1e-5, 1e-5, 1e-10]
    fit_bounds_upper = [np.inf, np.inf, np.inf, np.inf]
    try:
        popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0,
                                              bounds=(fit_bounds_lower, fit_bounds_upper))
    except ValueError:
        logger.warning('Error: fitting failed due to bad input values. Please try additional smoothing and/or interpolating data')
        popt, pcov = [0, 0, 0, 0], []
    # popt, pcov = scipy.optimize.curve_fit(logistic_func, x_axis, y_data, p0=p0, maxfev=5000)
    return popt, pcov


def find_nearest(array, value):
    """
    Get the index in the array nearest to the input value. Handles values outside the
    array by returning the end value in the correct direction.
    :param array: array-like object to search within
    :param value: value to find nearest index in the array
    :return: index (int) of closest match in the array
    """
    idx = (np.abs(array - value)).argmin()
    return idx


class Feature(object):
    """
    Holder for feature information while doing feature detection
    """
    def __init__(self, gaussian_bool):
        """
        Create a new feature object to hold feature information. Intended to add to cv/centroid info over time
        :param gaussian_bool: Whether this feature was constructed from Gaussian fit data or not
        """
        self.cvs = []
        self.cv_indices = []
        self.centroids = []
        self.gauss_median_centroid = None
        self.gaussians = []     # NOT necessarily sorted in CV order
        self.gaussian_bool = gaussian_bool

        # attributes to handle conversion for use with Transitions. Will be set after fitting by a method
        self.start_cv_index = None
        self.end_cv_index = None
        self.start_cv_val = None
        self.end_cv_val = None
        self.dt_max_bins = None
        self.dt_max_vals = []

    def __str__(self):
        # display either the gaussian or changepoint version data, including median and length of list
        # if self.gaussian_bool:
        #     return '<Feature> Med: {:.1f} Len: {}'.format(self.get_median(), len(self.cvs))
        # else:
        return '<Feature> Med: {:.1f} Len: {}'.format(self.get_median(), len(self.cvs))
    __repr__ = __str__

    def refresh(self):
        """
        Refresh the centroid median and cvs using the gaussians that have been added to the feature
        :return: void
        """
        self.gauss_median_centroid = np.median([x.centroid for x in self.gaussians])
        # get all CVs included (without repeats)
        for gaussian in self.gaussians:
            if gaussian.cv not in self.cvs:
                self.cvs.append(gaussian.cv)
        self.cvs = sorted(self.cvs)

    def accept_centroid(self, centroid, width_tol, collision_voltage, cv_tol, cv_spacing):
        """
        Determine whether the provided centroid is within tolerance of the feature or not. Uses
        feature detection parameters (flat width tolerance) to decide.
        :param centroid: the centroid (float) to compare against Feature
        :param width_tol: tolerance in DT units (float) to compare to centroid
        :param collision_voltage: CV position of the gaussian to compare against feature for gaps
        :param cv_tol: distance in collision voltage space that can be skipped and still accept a gaussian
        :param cv_spacing: distance between discrete points along collision voltage axis of CIU data
        :return: boolean
        """
        # Refresh current median and cvs in case more gaussians have been added since last calculation
        self.refresh()

        # ensure cv_tol is at least the cv bin spacing
        if cv_tol < cv_spacing:
            cv_tol = cv_spacing

        if abs(self.get_median() - centroid) <= width_tol:
            # centroid is within the Feature's bounds, check for gaps
            nearest_cv_index = (np.abs(np.asarray(self.cvs) - collision_voltage)).argmin()
            nearest_cv = self.cvs[nearest_cv_index]

            # check for duplicate at this CV
            if nearest_cv == collision_voltage:
                # A peak is already present at this CV. Use whichever peak is closer to nearby data as the correct one
                nearby_median = np.median([x.centroid for x in self.gaussians[nearest_cv_index - 4: nearest_cv_index - 1]])
                if abs(centroid - nearby_median) < abs(self.gaussians[nearest_cv_index].centroid - nearby_median):
                    # the new peak is closer to the nearby data - replace the existing peak with this one
                    self.gaussians.remove(self.gaussians[nearest_cv_index])
                    self.cvs.remove(nearest_cv)
                    return True
                else:
                    # the existing peak is closer - keep it
                    return False

            # if collision voltage is within tolerance of the nearest CV in the feature already, return True
            cv_diff = abs(collision_voltage - nearest_cv)
            within_tol_bool = cv_diff <= cv_tol
            return within_tol_bool

    def get_median(self):
        """
        Return the median centroid (DT units) of this Feature uniformly for Gaussian and non-Gaussian
        Features.
        :return: (float) feature median
        """
        if self.gaussian_bool:
            return self.gauss_median_centroid
        else:
            return np.median(self.dt_max_vals)

    def get_std_dev(self):
        """
        Return the standard deviation (in drift axis units) of centroids in this feature
        :return: (float) std deviation
        """
        if self.gaussian_bool:
            return np.std([x.centroid for x in self.gaussians])
        else:
            return np.std(self.centroids)

    def get_gaussian_at_cv(self, cv):
        """
        Return the Gaussian at the provided cv, or None if one is not present
        :param cv: collision voltage (float) at which to look for the Gaussian
        :return: Gaussian object found at the provided CV, or a Gaussian with 0 amplitude if none found
        :rtype: Gaussian
        """
        for gaussian in self.gaussians:
            if gaussian.cv == cv:
                return gaussian
        return Gaussian_Fitting.Gaussian(amplitude=0, width=1e-5, centroid=0, collision_voltage=cv, pcov=None, protein_bool=False)

    def init_feature_data(self, cv_index_list, dt_val_list):
        """
        Import and set data to use with Transition class. Adapted to removed ChangeptFeature subclass
        Note: *requires gaussian feature detection to have been performed previously*
        :param cv_index_list: list of indices of collision voltages that make up this feature (args for sublist of CV axis)
        :param dt_val_list: list of max_dt values in ms for each collision voltage in the feature
        """
        self.start_cv_val = self.cvs[0]
        self.end_cv_val = self.cvs[len(self.cvs) - 1]

        self.start_cv_index = cv_index_list[0]
        self.end_cv_index = cv_index_list[len(cv_index_list) - 1]

        self.dt_max_vals = dt_val_list


class Transition(object):
    """
    Store information about a CIU transition, including the starting and ending feature,
    their combined CV/index range and DT data, and fitted logistic function parameters
    and CIU50.
    """
    def __init__(self, feature1, feature2, analysis_obj, gaussian_bool):
        """
        Create a combined Transition object from two identified features. Features MUST be
        adjacent in CV space for this to make sense.
        :param feature1: Lower CV ("earlier/starting") Feature object
        :param feature2: Higher CV ("later/ending") Feature object
        :param analysis_obj: CIUAnalysisObj with data to be fitted
        :type analysis_obj: CIUAnalysisObj
        :param gaussian_bool: Whether to fit in Gaussian mode (True) or standard (False)
        """
        # initialize data from analysis_obj
        self.filename = analysis_obj.short_filename

        self.feature1 = feature1    # type: Feature
        self.feature2 = feature2    # type: Feature

        self.start_cv = feature1.start_cv_val
        self.end_cv = feature2.end_cv_val
        self.start_index = feature1.start_cv_index
        self.end_index = feature2.end_cv_index
        self.feat_distance = self.feature2.start_cv_val - self.feature1.end_cv_val

        if self.feat_distance < 0:
            # overlapping features: flip sign to make feature distance the overlap distance
            self.feat_distance = self.feat_distance * -1

        self.combined_x_axis = analysis_obj.axes[1][self.start_index: self.end_index + 1]     # +1 b/c slicing
        self.combined_y_vals, self.combined_y_avg_raw, self.combined_y_median_raw = self.compute_spectral_yvals(analysis_obj.ciu_data, analysis_obj.axes[0], gaussian_bool)

        self.min_guess = None
        self.max_guess = None

        self.ciu50 = None
        self.fit_params = None
        self.fit_covariances = None
        self.rsq = None
        self.fit_param_errors = None

    def __str__(self):
        # display the transition range and CIU50 and R2 if they have been calculated
        if self.ciu50 is not None and self.rsq is not None:
            return '<Transition> range: {}-{}, ciu50: {:.1f}, rsq: {:.2f}'.format(self.start_cv,
                                                                                  self.end_cv,
                                                                                  self.ciu50,
                                                                                  self.rsq)
        else:
            return '<Transition> range: {}-{}'.format(self.start_cv, self.end_cv)
    __repr__ = __str__

    def compute_spectral_yvals(self, ciu_data, dt_axis, gaussian_bool):
        """
        Determine spectral average and median values in standard and Gaussian modes to be
        used in CIU50 fitting. Assumes that self.combined_y_vals has been set to the column
        maxes of the provided ciu_data
        :param ciu_data: CIUAnalysisObj.ciu_data: CIU data array in standard form
        :param dt_axis: IM axis from CIUAnalysisObj
        :param gaussian_bool: True if Gaussian, False if standard
        :return: lists of max, average, and median values at each CV in the Transition
        """
        y_max_cols = []
        y_wtd_avg_cols = []
        y_median_cols = []

        # prepare CIU column data for analysis
        if not gaussian_bool:
            # standard mode, compute y values based on raw data within the feature range
            y_col_data = np.swapaxes(ciu_data, 0, 1)
            y_col_data = y_col_data[self.start_index: self.end_index + 1]
        else:
            # Gaussian mode - use filtered Gaussian data to compute y-values
            gaussian_ciu_cols = []

            # Reconstruct Gaussian data for ONLY the two features being considered
            for index, cv in enumerate(self.combined_x_axis):
                gauss1 = self.feature1.get_gaussian_at_cv(cv)
                gauss2 = self.feature2.get_gaussian_at_cv(cv)
                if gauss1.amplitude == 0 and gauss2.amplitude == 0:
                    # There is a gap between features where no Gaussians are fit - use raw column max data to fill it
                    col_data = np.swapaxes(ciu_data, 0, 1)[index]
                    gaussian_ciu_cols.append(col_data)
                else:
                    # At least one Gaussian was found at this CV, so use it as recon data
                    all_params = gauss1.return_popt()
                    all_params.extend(gauss2.return_popt())
                    recon_data = Gaussian_Fitting.multi_gauss_func(dt_axis, *all_params)
                    gaussian_ciu_cols.append(recon_data)

            y_col_data = gaussian_ciu_cols

        # compute spectral max, average, and median for each CIU column
        for cv_col in y_col_data:
            max_index = np.argmax(cv_col)
            y_max_cols.append(dt_axis[max_index])

            # the weighted sum of a column is the product of intensity * drift time for each bin
            wtd_sum_dt = 0
            int_sum = 0
            wtd_sum_dts = []
            for i in range(len(cv_col)):
                # cv_col[i] is the intensity at drift bin i; axes[0][i] is the drift time at bin i
                wtd_value = cv_col[i] * dt_axis[i]
                wtd_sum_dt += wtd_value
                int_sum += cv_col[i]
                wtd_sum_dts.append(wtd_sum_dt)

            # spectral centroid (average) = (sum of (DT_value * intensity)) / (sum of intensity)
            y_wtd_avg_cols.append(wtd_sum_dt / int_sum)

            # spectral median - center of weighted sum distribution
            med_value = wtd_sum_dt / 2.0
            wtd_sum_dts = np.asarray(wtd_sum_dts)
            med_index = (np.abs(wtd_sum_dts - med_value)).argmin()
            med_dt = dt_axis[med_index]
            y_median_cols.append(med_dt)
        return y_max_cols, y_wtd_avg_cols, y_median_cols

    def fit_transition(self, params_obj):
        """
        Fit a logistic function to the transition using the feature information. Requires
        bin_spacing and dt_min to do conversion to DT space from bin space for plotting.
        :param params_obj: Parameters object with fit settings information
        :type params_obj: Parameters
        :return: void (saves fit parameters to object)
        """
        # initial fitting guesses: center is in between the features, min/max are median DTs of features 1 and 2
        center_guess = self.feature2.start_cv_val - (self.feat_distance / 2.0)   # halfway between features
        self.min_guess = self.feature1.get_median()
        self.max_guess = self.feature2.get_median()

        # guess steepness as a function of between feature1 end and feature2 start
        steepness_guess = 2 * 1 / (self.feat_distance + 1)
        if steepness_guess < 0:
            steepness_guess = -1 * steepness_guess

        # for interpolation of transition modes - determine transition region to interpolate
        pad_cv = params_obj.ciu50_t2_2_pad_transitions_cv
        if self.feature1.end_cv_val < self.feature2.start_cv_val:
            # features do NOT overlap, go from end of first feature to start of second
            trans_start_cv = self.feature1.end_cv_val - pad_cv
            trans_end_cv = self.feature2.start_cv_val + pad_cv
        else:
            # features DO overlap - use overlap region as transition region
            trans_start_cv = self.feature2.start_cv_val - pad_cv
            trans_end_cv = self.feature1.end_cv_val + pad_cv

        # interpolate whole dataset by a factor of 2 for improved fitting quality
        interp_x_vals = np.linspace(self.combined_x_axis[0], self.combined_x_axis[len(self.combined_x_axis) - 1],
                                    len(self.combined_x_axis) * 2)
        interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_vals)
        interp_y_vals = interp_function_raw(interp_x_vals)
        # interp_x_vals = self.combined_x_axis
        # interp_y_vals = self.combined_y_vals
        # interp_function_raw = None

        if params_obj.ciu50_t2_1_centroiding_mode == 'average':
            # use spectral average for y-values
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_avg_raw)
            interp_y_vals = interp_function_raw(interp_x_vals)
        elif params_obj.ciu50_t2_1_centroiding_mode == 'median':
            # spectral median for y-values
            interp_function_raw = scipy.interpolate.interp1d(self.combined_x_axis, self.combined_y_median_raw)
            interp_y_vals = interp_function_raw(interp_x_vals)

        # Skip transition data assembly for Gaussian mode, keep standard mode unchanged
        if params_obj.feature_t1_1_ciu50_mode == 'gaussian':
            final_x_vals = interp_x_vals
            final_y_vals = interp_y_vals
        else:
            if params_obj.ciu50_t2_1_centroiding_mode == 'max':
                # no spectral centroiding
                final_x_vals, final_y_vals = self.assemble_transition_data(interp_x_vals, interp_y_vals, trans_start_cv,
                                                                           trans_end_cv, self.feat_distance)
            else:
                final_x_vals, final_y_vals = self.assemble_transition_data(interp_x_vals, interp_y_vals, trans_start_cv,
                                                                           trans_end_cv, self.feat_distance,
                                                                           interp_trans_factor=2,
                                                                           trans_interp_fn=interp_function_raw)

        # run the logistic fitting
        try:
            popt, pcov = fit_logistic(final_x_vals, final_y_vals, center_guess, self.min_guess, self.max_guess,
                                      steepness_guess)
            perr = np.sqrt(np.diag(pcov))
            self.fit_param_errors = perr
        except RuntimeError:
            logger.error('fitting failed for {} in file {}'.format(self, self.filename))
            popt = [0, 0, 0, 0]
            pcov = []

        # check goodness of fit
        yfit = logistic_func(interp_x_vals, *popt)
        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(interp_y_vals, yfit)
        rsq = rvalue ** 2

        if popt[2] < 0:
            logger.warning('WARNING: poor performance from logistic fitting for {} in file {}'.format(self.__str__(), self.filename))
        self.ciu50 = popt[2]
        self.fit_params = popt
        self.fit_covariances = pcov
        self.rsq = rsq

    def assemble_transition_data(self, interpd_x_vals, fit_y_vals, trans_start_cv, trans_end_cv,
                                 trans_distance, interp_trans_factor=1, trans_interp_fn=None):
        """

        :param interpd_x_vals: Interpolated combined x-axis data from both features. Runs from feature1 start to feature2 end
        :param fit_y_vals: y-axis data corresponding to interpd_x_vals array (same length)
        :param trans_start_cv: Starting point of transition region (including padding)
        :param trans_end_cv: End point of transition region (including padding)
        :param trans_distance: length of transition region (in CV), including padding
        :param trans_interp_fn:
        :param interp_trans_factor: Factor by which to interpolate transition region (e.g. 2 would double the number of points considered)
        :return: final x_data, final y_data (equal length 1D ndarrays)
        """
        interpd_cv_step = interpd_x_vals[1] - interpd_x_vals[0]
        trans_start_index = find_nearest(interpd_x_vals, trans_start_cv)
        trans_end_index = find_nearest(interpd_x_vals, trans_end_cv)

        if trans_interp_fn is not None:
            transition_x_vals = np.linspace(trans_start_cv, trans_end_cv, (trans_distance / interpd_cv_step) * interp_trans_factor)
            transition_y_vals = trans_interp_fn(transition_x_vals)
        else:
            # cannot interpolate because no function provided, simply use the corresponding x and y data
            transition_x_vals = interpd_x_vals[trans_start_index: trans_end_index + 1]
            transition_y_vals = fit_y_vals[trans_start_index: trans_end_index + 1]

        final_x_vals = interpd_x_vals[0: trans_start_index]
        final_x_vals = np.append(final_x_vals, transition_x_vals)
        second_half_xvals = interpd_x_vals[trans_end_index + 1: len(interpd_x_vals) - 1]
        final_x_vals = np.append(final_x_vals, second_half_xvals)

        final_y_vals = [self.min_guess for _ in range(0, trans_start_index)]
        final_y_vals = np.append(final_y_vals, transition_y_vals)
        second_half_yvals = [self.max_guess for _ in range(trans_end_index + 1, len(fit_y_vals) - 1)]
        final_y_vals = np.append(final_y_vals, second_half_yvals)

        return final_x_vals, final_y_vals

    def check_features(self, analysis_obj, params_obj):
        """
        Method containing checks for Gaussian feature based transitions. Confirms that the
        second feature contains at least one column max value (i.e. the transition to the feature does
        take place).
        :param analysis_obj: CIUAnalysisObj with axis and column max information
        :type analysis_obj: CIUAnalysisObj
        :param params_obj: Parameters object with parameter information
        :type params_obj: Parameters
        :return: True if checks are satisfied, False if not
        """
        # non-Gaussian features don't need checking - return True
        if not self.feature1.gaussian_bool and not self.feature2.gaussian_bool:
            return True

        if self.start_cv > self.end_cv:
            return False

        feature2_cv_indices = np.arange(self.feature2.start_cv_index, self.feature2.end_cv_index)
        width_tol_dt = params_obj.feature_t2_2_width_tol   # * analysis_obj.bin_spacing
        for cv_index in feature2_cv_indices:
            # check if a column max is within tolerance of the feature median
            current_max_dt = analysis_obj.col_max_dts[cv_index]
            if abs(current_max_dt - self.feature2.gauss_median_centroid) <= width_tol_dt:
                return True

        # no CV found with column max within tolerance - return false
        return False


# testing
# if __name__ == '__main__':
    # import tkinter
    # from tkinter import filedialog
    #
    # # open a filechoose to choose .ciu files (change to .csv and add generate_raw_obj/process_raw_obj from CIU2 main if analyzing _raw.csv data)
    # root = tkinter.Tk()
    # root.withdraw()
    # files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    # main_dir = os.path.dirname(files[0])
    #
    # # initialize parameters to defaults
    # params = CIU_Params.Parameters()
    # params.set_params(CIU_Params.parse_params_file(CIU_Params.hard_descripts_file))
    #
    # # load files and run feature detection and/or CIU-50
    # for file in files:
    #     with open(file, 'rb') as analysis_file:
    #         obj = pickle.load(analysis_file)
    #
    #     obj = feature_detect_col_max(obj, params)
    #     obj = ciu50_main(obj, params, main_dir, gaussian_bool=False)

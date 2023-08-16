"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky and Sugyan Dixit

Module for classification schemes for CIU data groups
Authors: Dan Polasky, Sugyan Dixit
Date: 1/11/2018
"""
from Gaussian_Fitting import Gaussian
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches
import pickle
import os
import itertools
import random
import logging
import time
import tkinter
from tkinter import messagebox
from tkinter import ttk
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import precision_score, roc_curve, auc
from sklearn.feature_selection import f_classif, GenericUnivariateSelect
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier


from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj
    from CIU_Params import Parameters
    from Feature_Detection import Feature

# load main logger
logger = logging.getLogger('main')


def main_build_classification_new(cl_inputs_by_label, subclass_labels, params_obj, output_dir, manual_feat_select):
    """
    Main method for classification. Performs feature selection followed by LDA and classification
    and generates output and plots. Returns a ClassificationScheme object to be saved for future
    classification of unknowns. Allows use of subclasses.
    NOTE: requires that input data has had axes equalized previously (assumes all axes are identical at this point)
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param subclass_labels: list of subclass labels (strings). If no subclasses present, default to ['0']
    :param output_dir: directory in which to save plots/output
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param manual_feat_select: boolean - whether to allow the user to manually choose features
    :return: ClassificationScheme object with the generated scheme
    :rtype: ClassificationScheme
    """
    start_time = time.time()

    # Data preparation for Gaussians (if applicable) and standardization
    max_num_gaussians = prep_data_2d(cl_inputs_by_label, params_obj)
    cl_inputs_by_label, means, stdevs = standardize_all_2d(cl_inputs_by_label, params_obj)

    # convert to subclass oriented data (if no subclasses, will be a single entry in a list)
    class_labels = [class_list[0].class_label for class_list in cl_inputs_by_label]
    list_classif_inputs = subclass_inputs_from_class_inputs(cl_inputs_by_label, subclass_labels, class_labels)

    logger.debug('standardization finished: {:.2f}s'.format(time.time() - start_time))

    if not manual_feat_select:
        # run feature selection and crossvalidation to select best features automatically
        all_features = multi_subclass_ufs(list_classif_inputs, params_obj, output_dir, subclass_labels)
        logger.debug('UFS finished: {:.2f}s'.format(time.time() - start_time))

        # assess all features to determine which to use in the final scheme
        best_features, crossval_score, all_crossval_data = crossval_main_new(cl_inputs_by_label, output_dir, params_obj, all_features, subclass_labels)
    else:
        sorted_features = multi_subclass_ufs(list_classif_inputs, params_obj, output_dir, subclass_labels)
        sorted_features = sorted_features[:params_obj.classif_7_max_feats_for_crossval]  # only display set number of features for selection
        selected_features = get_manual_classif_feats(sorted_features)

        # Manual mode: use the provided features and run limited crossvalidation
        best_features, crossval_score, all_crossval_data = crossval_main_new(cl_inputs_by_label, output_dir, params_obj, selected_features, subclass_labels)
        best_features = selected_features
        all_features = selected_features
    logger.debug('crossval finished: {:.2f}s'.format(time.time() - start_time))

    # perform LDA and classification on the selected/best features
    shaped_subsets = rearrange_ciu_by_feats(cl_inputs_by_label, best_features, params_obj)
    flat_subsets = [x for class_list in shaped_subsets for x in class_list]
    constructed_scheme = lda_svc_best_feats(flat_subsets, best_features, all_features, output_dir, subclass_labels, max_num_gaussians)
    constructed_scheme.crossval_test_score = crossval_score
    constructed_scheme.all_crossval_data = all_crossval_data
    constructed_scheme.standard_means = means
    constructed_scheme.standard_stdevs = stdevs

    # plot output here for now, will probably move eventually
    plot_classification_decision_regions(constructed_scheme, params_obj, output_dir)
    logger.debug('classif (scheme = {}) finished: {:.2f}s'.format(constructed_scheme.name, time.time() - start_time))
    return constructed_scheme


def multi_subclass_ufs(subclass_input_list, params_obj, output_path, subclass_labels):
    """
    Perform univariate feature selection across classes using multiple subclasses (e.g. different charge states
    of CIU fingerprints). Essentially performs standard UFS on each subclass separately and combines
    all output features/scores into a single list, from which the best features can be chosen for
    LDA/SVM classification. Inputs are structured like in standard UFS method, except provided
    as a list of inputs for each subclass rather than a single input (as the standard method takes only
    1 "subclass")
    :param subclass_input_list: list of ClassifInput containers for each subclass
    :type subclass_input_list: list[ClassifInput]
    :param params_obj: parameters information
    :type params_obj: Parameters
    :param output_path: directory in which to save plot
    :param subclass_labels: list of strings for scheme name
    :return: list of all features, sorted in decreasing order of score from ALL subclasses
    :rtype: list[CFeature]
    """
    # Iterate over all subclass lists to generate feature score information
    features = []
    features_by_subclass = []
    for subclass_input in subclass_input_list:
        # generate all combinations of replicate datasets within the labels
        shaped_label_list = subclass_input.shaped_label_list
        scores = generate_products_for_ufs(subclass_input.analysis_objs_by_label, shaped_label_list, params_obj)

        # Create a CFeature object to hold the information for this CV (feature)
        mean_score = np.mean(scores, axis=0)
        std_score = np.std(scores, axis=0)

        cv_axis = subclass_input.analysis_objs_by_label[0][0].axes[1]
        subclass_features = []
        for cv_index, cv in enumerate(cv_axis):
            feature = CFeature(cv, cv_index, mean_score[cv_index], std_score[cv_index], subclass_label=subclass_input.subclass_label)
            features.append(feature)
            subclass_features.append(feature)
        features_by_subclass.append(subclass_features)

    # sort feature scores either by mean - stdev ("error mode") or just mean alone.
    if params_obj.classif_6_ufs_use_error_mode:
        sorted_features = sorted(features, key=lambda x: (x.mean_score - x.std_dev_score), reverse=True)
    else:
        sorted_features = sorted(features, key=lambda x: x.mean_score, reverse=True)

    unique_labels = get_unique_labels([x for label_list in subclass_input_list[0].shaped_label_list for x in label_list])
    scheme_name = generate_scheme_name(unique_labels, subclass_labels)
    plot_feature_scores_subclass(features_by_subclass, params_obj, scheme_name, output_path)
    save_feature_scores(features_by_subclass, scheme_name, output_path)
    return sorted_features


def generate_products_for_ufs(analysis_obj_list_by_label, shaped_label_list, params_obj):
    """
    Generate all combinations of replicate data across classes for feature selection. Will
    create a DataProduct object with the key information for each combination.
    :param analysis_obj_list_by_label: list of lists of CIUAnalysisObj's, sorted by class label
    :type analysis_obj_list_by_label: list[list[CIUAnalysisObj]]
    :param shaped_label_list: list of lists of class labels with matching shape of analysis_obj_by_label
    :param params_obj: parameter info
    :type params_obj: Parameters
    :return: list of DataProduct objects for each combination
    :rtype: list[UFSResult]
    """
    scores = []
    for object_tuple, label_tuple in zip(itertools.product(*analysis_obj_list_by_label), itertools.product(*shaped_label_list)):
        # create a UFSResult object for this combination
        # data_list = [x.ciu_data for x in object_tuple]
        data_list = [get_classif_data(x, params_obj, ufs_mode=True) for x in object_tuple]

        label_list = [x for x in label_tuple]
        product = UFSResult(data_list, label_list)

        # Run feature selection for this combination
        select = GenericUnivariateSelect(score_func=f_classif, mode='percentile', param=100)
        select.fit(product.combined_data, product.numeric_label_arr)

        product.fit_pvalues = select.pvalues_
        product.fit_scores = select.scores_
        product.fit_sc = -np.log10(select.pvalues_)

        # scores.append(product.fit_scores)
        scores.append(product.fit_sc)   # don't save whole product to reduce memory load
    return scores


def roc_curve_area_multiclass(x_train, y_train, x_test, y_test, svc=None):
    """
    creates a one vs all classifer
    calculates the roc fpr, tpr, for each class with one vs all, micro-average roc, and macro-average roc
    :param x_test: x_test lda
    :param x_train: x_train lda
    :param y_train: y_train labels
    :param y_test: y_test labels
    :param svc: SVC classifier from cross validation to use for binary classifications. None for multiclass
    :return: fpr, tpr, and roc_auc for class specific, micro-average, and macro-average
    :rtype: dict
    """
    output_dict = {}
    temp_fpr = np.linspace(0, 1, 100)
    unique_class_labels = np.unique(y_train)

    if len(unique_class_labels) == 2:
        # only 2 classes, and thus only 1 classifier. No OneVsRest classification needed
        y_train_binary = binarize_2class(unique_class_labels, y_train)
        y_test_binary = binarize_2class(unique_class_labels, y_test)
        # use saved classifier for binary if available
        if svc is not None:
            clf = svc
        else:
            clf = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
            clf.fit(x_train, y_train_binary)
        y_score = clf.decision_function(x_test)

        # generate ROC curve
        fpr_class, tpr_class, thr = roc_curve(y_test_binary, y_score)
        interp_tpr = np.interp(temp_fpr, fpr_class, tpr_class)
        interp_tpr[0] = 0
        roc_auc_interp = auc(temp_fpr, interp_tpr)
        interp_tpr_class = [[interp_tpr]]

        # save micro/macro avg as just regular fpr/tpr
        interp_tpr_micro = interp_tpr
        interp_tpr_macro = interp_tpr
        roc_auc_micro_interp = roc_auc_interp
        roc_auc_macro_interp = roc_auc_interp
        roc_auc_class_interp = [roc_auc_interp]
    else:
        # multiclass: must be binarized (converted to one vs rest classifiers) for ROC analysis
        y_train_binary = label_binarize(y_train, classes=unique_class_labels)
        y_test_binary = label_binarize(y_test, classes=unique_class_labels)
        n_classes = y_train_binary.shape[1]
        clf = OneVsRestClassifier(SVC(kernel='linear', C=1, probability=True, max_iter=1000)).fit(x_train, y_train_binary)
        y_score = clf.decision_function(x_test)

        # create dicts for fpr, tpr, and roc_auc
        fpr_class = [[] for _ in range(n_classes)]
        tpr_class = [[] for _ in range(n_classes)]
        interp_tpr_class = [[] for _ in range(n_classes)]
        roc_auc_class_interp = [[] for _ in range(n_classes)]
        for index in range(n_classes):
            fpr_class[index], tpr_class[index], _ = roc_curve(y_test_binary[:, index], y_score[:, index])
            interp_tpr = np.zeros(np.shape(temp_fpr))
            interp_tpr += np.interp(temp_fpr, fpr_class[index], tpr_class[index])
            interp_tpr[0] = 0
            interp_tpr_class[index].append(interp_tpr)
            roc_auc_class_interp[index] = auc(temp_fpr, interp_tpr_class[index][0])
        roc_auc_class_interp = np.asarray(roc_auc_class_interp)
        interp_tpr_class = np.asarray(interp_tpr_class)

        # compute micro-average ROC curve and ROC area
        fpr_micro, tpr_micro, _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
        interp_tpr_micro = np.zeros(np.shape(temp_fpr))
        interp_tpr_micro += np.interp(temp_fpr, fpr_micro, tpr_micro)
        interp_tpr_micro[0] = 0
        roc_auc_micro_interp = auc(temp_fpr, interp_tpr_micro)

        # compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr_class[x] for x in range(n_classes)]))
        # interpolate all ROC curves at these fpr points
        mean_tpr = np.zeros_like(all_fpr)
        for index in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_class[index], tpr_class[index])
        # average tpr and compute AUC
        mean_tpr /= n_classes
        interp_tpr_macro = np.zeros(np.shape(temp_fpr))
        interp_tpr_macro += np.interp(temp_fpr, all_fpr, mean_tpr)
        interp_tpr_macro[0] = 0
        roc_auc_macro_interp = auc(temp_fpr, interp_tpr_macro)

    # save final outputs
    output_dict['tmp_fpr'] = temp_fpr
    output_dict['tpr_class'] = interp_tpr_class
    output_dict['roc_auc_class'] = roc_auc_class_interp
    output_dict['tpr_micro'] = interp_tpr_micro
    output_dict['roc_auc_micro'] = roc_auc_micro_interp
    output_dict['tpr_macro'] = interp_tpr_macro
    output_dict['roc_auc_macro'] = roc_auc_macro_interp
    return output_dict


def binarize_2class(unique_class_labels, label_data):
    """
    Binarize labels for 2-class case for ROC curve (convert labels to 0 or 1). Generally just
    subtracting 1 from unique numeric labels, but more general support provided.
    :param unique_class_labels: list of unique class labels (usually numeric)
    :param label_data: data to convert
    :return: binarized label_data
    """
    label_encoding = {}
    bin_index = 0
    for class_label in unique_class_labels:
        label_encoding[class_label] = bin_index
        bin_index += 1

    output_data = []
    for label in label_data:
        output_data.append(label_encoding[label])
    return output_data


def plot_roc_cuve(roc_data, class_labels, schem_name, dirpath, params_obj, selected_features=None):
    """
    Plot ROC curves for each number of features in a single PDF document. roc_data input is the
    saved results dictionary from a CrossValRun container.
    :param roc_data: saved results dictionary from a CrossValRun container.
    :param class_labels: list of strings - labels for each class
    :param schem_name:
    :param dirpath:
    :param selected_features:
    :param params_obj: parameters
    :type params_obj: Parameters
    :return:
    """
    tmp_fpr = roc_data['tmp_fpr'][0]
    tpr_class_mean, tpr_class_std, roc_auc_class_mean, roc_auc_class_std = roc_data['tpr_class_mean'], roc_data['tpr_class_std'], roc_data['roc_auc_class_mean'], roc_data['roc_auc_class_std']
    tpr_micro_mean, tpr_micro_std, roc_auc_micro_mean, roc_auc_micro_std = roc_data['tpr_micro_mean'], roc_data['tpr_micro_std'], roc_data['roc_auc_micro_mean'], roc_data['roc_auc_micro_std']
    tpr_macro_mean, tpr_macro_std, roc_auc_macro_mean, roc_auc_macro_std = roc_data['tpr_macro_mean'], roc_data['tpr_macro_std'], roc_data['roc_auc_macro_mean'], roc_data['roc_auc_macro_std']

    if selected_features is None:
        pdf_output = os.path.join(dirpath, schem_name + '_ROC_curves.pdf')
        try:
            testfile = open(pdf_output, 'a')
            testfile.close()
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving',
                                 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(
                                     pdf_output))

        with PdfPages(pdf_output) as pdf:
            for index in range(len(tpr_class_mean)):
                for num, (tpr_class_mean_, tpr_class_std_, roc_auc_class_mean_, roc_auc_class_std_) in enumerate(zip(tpr_class_mean[index], tpr_class_std[index], roc_auc_class_mean[index], roc_auc_class_std[index])):
                    plt.plot(tmp_fpr, tpr_class_mean_[0], linestyle=':', label='{0} {1:0.2f} +/- {2:0.2f}'.format(class_labels[num], roc_auc_class_mean_, roc_auc_class_std_))
                    plt.fill_between(tmp_fpr, tpr_class_mean_[0] + tpr_class_std_[0], tpr_class_mean_[0] - tpr_class_std_[0], alpha=0.2)

                # plot macro/micro averages only if more than one classifier
                if len(tpr_class_mean[0]) > 1:
                    plt.plot(tmp_fpr, tpr_micro_mean[index], color='navy', label='ROC_micro {0:0.2f} +/- {1:0.2f}'.format(roc_auc_micro_mean[index], roc_auc_micro_std[index]))
                    plt.fill_between(tmp_fpr, tpr_micro_mean[index] + tpr_micro_std[index], tpr_micro_mean[index] - tpr_micro_std[index], color='black', alpha=0.4)
                    plt.plot(tmp_fpr, tpr_macro_mean[index], color='red', label='ROC_macro {0:0.2f} +/- {1:0.2f}'.format(roc_auc_macro_mean[index], roc_auc_macro_std[index]))
                    plt.fill_between(tmp_fpr, tpr_macro_mean[index] + tpr_macro_std[index], tpr_macro_mean[index] - tpr_macro_std[index], color='red', alpha=0.4)
                plt.plot([0, 1], [0, 1], color='black', linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])

                # plot titles, labels, and legends
                if params_obj.plot_12_custom_title is not None:
                    plot_title = params_obj.plot_12_custom_title
                    plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
                elif params_obj.plot_11_show_title:
                    plot_title = 'ROC: {} Features'.format(index + 1)
                    plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
                if params_obj.plot_08_show_axes_titles:
                    plt.xlabel('False Positive Rate', fontsize=params_obj.plot_13_font_size, fontweight='bold')
                    plt.ylabel('True Positive Rate', fontsize=params_obj.plot_13_font_size, fontweight='bold')
                plt.xticks(fontsize=params_obj.plot_13_font_size)
                plt.yticks(fontsize=params_obj.plot_13_font_size)
                if params_obj.plot_07_show_legend:
                    plt.legend(loc='best', fontsize='small')

                try:
                    pdf.savefig()
                except PermissionError:
                    messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(pdf_output))
                    plt.savefig()
                plt.close()
    else:
        # plot ROC curve for single number of selected features
        index = len(selected_features) - 1  # indexed from 0 in the lists, not 1
        for num, (tpr_class_mean_, tpr_class_std_, roc_auc_class_mean_, roc_auc_class_std_) in enumerate(zip(tpr_class_mean[index], tpr_class_std[index], roc_auc_class_mean[index], roc_auc_class_std[index])):
            plt.plot(tmp_fpr, tpr_class_mean_[0], linestyle=':', label='{0} {1:0.2f} +/- {2:0.2f}'.format(class_labels[num], roc_auc_class_mean_, roc_auc_class_std_))
            plt.fill_between(tmp_fpr, tpr_class_mean_[0] + tpr_class_std_[0], tpr_class_mean_[0] - tpr_class_std_[0], alpha=0.2)

        # plot macro/micro averages only if more than one classifier
        if len(tpr_class_mean[0]) > 1:
            plt.plot(tmp_fpr, tpr_micro_mean[index], color='navy', label='ROC_micro {0:0.2f} +/- {1:0.2f}'.format(roc_auc_micro_mean[index], roc_auc_micro_std[index]))
            plt.fill_between(tmp_fpr, tpr_micro_mean[index] + tpr_micro_std[index], tpr_micro_mean[index] - tpr_micro_std[index], color='black', alpha=0.4)
            plt.plot(tmp_fpr, tpr_macro_mean[index], color='red', label='ROC_macro {0:0.2f} +/- {1:0.2f}'.format(roc_auc_macro_mean[index], roc_auc_macro_std[index]))
            plt.fill_between(tmp_fpr, tpr_macro_mean[index] + tpr_macro_std[index], tpr_macro_mean[index] - tpr_macro_std[index], color='red', alpha=0.4)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # plot titles, labels, and legends
        if params_obj.plot_12_custom_title is not None:
            plot_title = params_obj.plot_12_custom_title
            plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        elif params_obj.plot_11_show_title:
            plot_title = 'ROC: {} Features'.format(len(selected_features))
            plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel('False Positive Rate', fontsize=params_obj.plot_13_font_size, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks(fontsize=params_obj.plot_13_font_size)
        if params_obj.plot_07_show_legend:
            plt.legend(loc='best', fontsize='small')
        output_name = os.path.join(dirpath, schem_name + '_final-ROC' + params_obj.plot_02_extension)
        try:
            plt.savefig(output_name)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
            plt.savefig(output_name)
        plt.close()


def crossval_main_new(cl_inputs_by_label, outputdir, params_obj, features_list, subclass_labels):
    """
    Updated crossval method to allow multiple subclasses. Updated to reduce datasets to selected
    features first, then perform crossval with modular methods.
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param outputdir: directory in which to save output plots
    :param params_obj: parameters container
    :type params_obj: Parameters
    :param features_list: List of CFeatures, sorted in decreasing order of score
    :type features_list: list[CFeature]
    :param subclass_labels: list of strings for scheme naming purposes
    :return: list of selected features, test score for that # features, and all cross validation data
    """
    # determine training size as size of the smallest class - 1 (1 test file at a time)
    min_class_size = np.min([len(x) for x in cl_inputs_by_label])
    training_size = min_class_size - params_obj.classif_91_test_size
    if training_size < 2:
        logger.warning('WARNING! Testing size provided ({}) was too large: at least one class had less than 2 replicates of training data. Test size of 1 used instead.'.format(params_obj.classif_91_test_size))
        training_size = min_class_size - 1
    label_list = [class_list[0].class_label for class_list in cl_inputs_by_label]

    # optional max number of features to consider
    if params_obj.classif_7_max_feats_for_crossval > 0:
        max_features = params_obj.classif_7_max_feats_for_crossval
        if max_features > len(features_list):
            max_features = len(features_list) + 1
    else:
        max_features = len(features_list) + 1

    # determine number of products. If less than # iterations (or if # iterations = 0), calculate out all products. Otherwise, randomly select to save memory
    num_products = 1
    for cl_input_list in cl_inputs_by_label:
        num_permutations = len(list(itertools.permutations(cl_input_list, params_obj.classif_91_test_size)))
        num_products *= num_permutations
    if num_products < params_obj.classif_8_max_crossval_iterations or params_obj.classif_8_max_crossval_iterations == 0:
        calc_all_products = True
    else:
        calc_all_products = False

    current_features_list = []
    all_results_by_feats = {}
    for ind, feature in enumerate(features_list[:max_features]):
        # Generate all combinations - NOTE: assumes that if subclasses are present, features know their subclass (should always be true)
        logger.info('Performing cross validation for {} of {} features'.format(ind + 1, len(features_list[:max_features])))
        current_features_list.append(feature)

        # format all data
        shaped_data_list = rearrange_ciu_by_feats(cl_inputs_by_label, current_features_list, params_obj)

        # perform the cross validation for this feature combination
        crossval_obj = CrossValRun(shaped_data_list, label_list, training_size, current_features_list)
        if calc_all_products:
            crossval_obj.divide_data_and_run_lda()
            crossval_obj.assemble_class_products(params_obj.classif_8_max_crossval_iterations)
        else:
            crossval_obj.random_sample_run_lda(params_obj.classif_8_max_crossval_iterations)

        # save results into a dictionary with same keys as original crossval dict
        for key, value in crossval_obj.results.items():
            try:
                all_results_by_feats[key].append(crossval_obj.results[key])
            except KeyError:
                all_results_by_feats[key] = [crossval_obj.results[key]]

    all_results_by_feats['train_scores_mean'] = np.asarray(all_results_by_feats['train_scores_mean'])
    all_results_by_feats['train_scores_std'] = np.asarray(all_results_by_feats['train_scores_std'])
    all_results_by_feats['test_scores_mean'] = np.asarray(all_results_by_feats['test_scores_mean'])
    all_results_by_feats['test_scores_std'] = np.asarray(all_results_by_feats['test_scores_std'])

    # save and plot crossvalidation score information
    crossval_acc_data = [all_results_by_feats['train_scores_mean'], all_results_by_feats['train_scores_std'], all_results_by_feats['test_scores_mean'], all_results_by_feats['test_scores_std'], all_results_by_feats['roc_auc_micro_mean'], all_results_by_feats['roc_auc_micro_std']]

    unique_labels = get_unique_labels(label_list)
    scheme_name = generate_scheme_name(unique_labels, subclass_labels)
    crossval_file = save_crossval_score(crossval_acc_data, scheme_name, outputdir)
    save_roc_data(all_results_by_feats, crossval_file)
    plot_crossval_scores(crossval_acc_data, scheme_name, params_obj, outputdir)
    plot_roc_cuve(all_results_by_feats, unique_labels, scheme_name, outputdir, params_obj)
    # plot_crossval_auc(all_results_by_feats['roc_auc_micro_mean'], all_results_by_feats['roc_auc_micro_std'], scheme_name, params_obj, outputdir)

    # determine best features list from crossval scores
    if params_obj.classif_4_score_mode == 'auc':
        score_list = all_results_by_feats['roc_auc_micro_mean']
    else:
        score_list = all_results_by_feats['test_scores_mean']
    best_num_feats, best_score = peak_crossval_score_detect(score_list, params_obj.classif_2_score_dif_tol)
    output_features = features_list[0: best_num_feats]
    plot_roc_cuve(all_results_by_feats, unique_labels, scheme_name, outputdir, params_obj, selected_features=output_features)
    logger.info('Cross validation complete!')
    return output_features, best_score, crossval_acc_data


def rearrange_ciu_by_feats(shaped_inputs_list, features_list, params_obj):
    """
    For each CIU dataset in the original input, generate a rearranged (and possibly shrunken)
    matrix of data in order of features in the provided features list. Designed to allow easy
    access to the CV subset data of interest prior to doing crossval/LDA/etc. Handles subclasses.
    :param features_list: list of features in descending order of score from UFS
    :type features_list: list[CFeature]
    :param shaped_inputs_list: lists by class of SubclassUnknown containers with classification input info
    :type shaped_inputs_list: list[list[ClInput]]
    :param params_obj: parameters container for getting classif data (raw data or gaussian mode)
    :type params_obj: Parameters
    :return: shaped output data with selected feature data in order
    :rtype: list[list[DataSubset]]
    """
    # Loop over the shaped input list, extracting data and maintaining the same organization
    shaped_output_list = []
    class_numeric_label = 1
    for class_list in shaped_inputs_list:
        class_outputs = []

        for rep_obj in class_list:
            # generate a subset container to hold the extracted data and associated metadata
            rep_subset = rearrange_ciu_by_feats_helper(rep_obj, params_obj, features_list, class_numeric_label)
            class_outputs.append(rep_subset)

        shaped_output_list.append(class_outputs)
        class_numeric_label += 1

    return shaped_output_list


def rearrange_ciu_by_feats_helper(rep_obj, params_obj, features_list, class_numeric_label, num_gaussian_override=None):
    """
    Rearrange CIU data in feature order for a single replicate to generate a single DataSubset
    container to return.
    :param rep_obj: input ClInput container with raw data
    :type rep_obj: ClInput
    :param params_obj: paramters
    :type params_obj: Parameters
    :param features_list: list of features in decreasing order of score
    :type features_list: list[CFeature]
    :param class_numeric_label: (int) numeric label for scheme. Set to 0 for unknown data
    :param num_gaussian_override: For Gaussian mode with unknowns - require that the max num gaussians be that of the previously saved classification scheme
    :return: DataSubset container with initialized data
    :rtype: DataSubset
    """
    data_output = []
    if features_list[0].subclass_label is not None:
        # multiple subclasses, so file_id is the 'best' subclass object plus the number of subclasses
        len_subclasses = len(rep_obj.subclass_dict.keys())
        file_id = rep_obj.subclass_dict[features_list[0].subclass_label].short_filename + '_{}SubCl'.format(len_subclasses)
    else:
        # only one object, so just use its filename
        file_id = rep_obj.get_subclass_obj().short_filename

    for feature in features_list:
        if feature.subclass_label is not None:
            subclass_obj = rep_obj.get_subclass_obj(feature.subclass_label)
        else:
            subclass_obj = rep_obj.get_subclass_obj()

        # Determine the correct CV column to append to the growing matrix and do so
        current_cv_axis = subclass_obj.axes[1]
        this_cv_index = (np.abs(np.asarray(current_cv_axis) - feature.cv)).argmin()
        raw_data = get_classif_data(subclass_obj, params_obj)
        cv_col = raw_data.T[this_cv_index]
        data_output.append(cv_col)

    # generate a subset container to hold the extracted data and associated metadata
    rep_subset = DataSubset(data_output, rep_obj.class_label, class_numeric_label, file_id, rep_obj.all_filenames, features_list)
    return rep_subset


def arrange_lda_new(subset_list):
    """
    Prepare data and label arrays for input to LDA methods given an input list of DataSubset containers
    with formatted data and labels
    :param subset_list: list of DataSubset containers with formatted data and labels
    :type subset_list: list[DataSubset]
    :return: x_data, numeric labels, string labels for direct input into LDA
    """
    x_data = []
    string_labels, numeric_labels = [], []
    # assemble each dataset into a single list by combining all columns of the input matrices
    for subset in subset_list:
        for column in subset.data:
            x_data.append(column)
            string_labels.append(subset.class_label)
            numeric_labels.append(subset.numeric_label)

    # convert to numpy arrays for SKLearn analyses
    x_data = np.asarray(x_data)
    string_labels = np.asarray(string_labels)
    numeric_labels = np.asarray(numeric_labels)
    return x_data, numeric_labels, string_labels


def lda_svc_best_feats(flat_subset_list, selected_features, all_features, output_dir, subclass_labels, max_num_gaussians=0):
    """
    Generate a Scheme container by performing final LDA/SVM analysis on the provided data.
    :param flat_subset_list: list of DataSubset containers
    :type flat_subset_list: list[DataSubset]
    :param selected_features: selected (best) features to use in scheme construction
    :type selected_features: list[CFeature]
    :param all_features: list of all features input for reference
    :type all_features: list[CFeature]
    :param output_dir: directory in which to save output
    :param subclass_labels: list of strings for scheme output naming
    :param max_num_gaussians: for Gaussian mode, the max number of Gaussians to record in the scheme
    :return: Scheme container
    :rtype: ClassificationScheme
    """
    train_data, train_numeric_labels, train_string_labels = arrange_lda_new(flat_subset_list)
    svc, lda = run_lda_svc(train_data, train_numeric_labels)

    x_lda = lda.transform(train_data)
    expl_var_r = lda.explained_variance_ratio_

    # build classification scheme
    clf = SVC(kernel='linear', C=1, probability=True, max_iter=1000)
    clf.fit(x_lda, train_numeric_labels)
    y_pred = clf.predict(x_lda)
    prec_score = precision_score(train_numeric_labels, y_pred, pos_label=1, average='weighted')
    probs = clf.predict_proba(x_lda)

    # initialize classification scheme object and return it
    scheme = ClassificationScheme()
    scheme.all_features = all_features
    scheme.selected_features = selected_features
    scheme.classifier = clf
    scheme.classifier_type = 'SVC'
    scheme.classif_prec_score = prec_score
    scheme.lda = lda
    scheme.explained_variance_ratio = expl_var_r
    scheme.numeric_labels = train_numeric_labels
    scheme.class_labels = train_string_labels
    scheme.unique_labels = get_unique_labels(train_string_labels)
    scheme.transformed_test_data = x_lda
    scheme.params = clf.get_params()
    scheme.input_feats = [feature.cv for feature in selected_features]
    scheme.name = generate_scheme_name(train_string_labels, subclass_labels)
    scheme.num_gaussians = max_num_gaussians

    # Organize outputs by input file for easier viewing, then save
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, combined_filenames = prep_outputs_by_file_new(x_lda, y_pred, probs, flat_subset_list)
    save_lda_and_predictions(scheme, x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, combined_filenames, output_dir, unknowns_bool=False)
    return scheme


def run_lda_svc(x_data, label_data):
    """
    Run LDA and SVC analysis of a set of data and return the resulting classifier
    :param x_data: input x data
    :param label_data: input label data
    :return: sklearn.svm.SVC classifier object and LDA object
    :rtype: SVC, LinearDiscriminantAnalysis
    """
    lda = LinearDiscriminantAnalysis(solver='svd', n_components=5)
    lda.fit(x_data, label_data)
    train_lda = lda.transform(x_data)

    # max_iter=1000 needed to prevent occasional (and unpredictable) freezes with ridiculous iteration numbers
    svm = SVC(kernel='linear', C=1, probability=True, max_iter=1000, cache_size=200)
    try:
        svm.fit(train_lda, label_data)
    except ValueError:
        logger.error('Error in SVM fitting. This should not be reached - check your input data for duplicates (same input used in multiple classes)')

    return svm, lda


def get_classif_data(analysis_obj, params_obj, ufs_mode=False):
    """
    Initialize a classification data matrix in each analysis object in the lists according to the
    classification mode specified in the parameters object. In All_Data mode, this is simply the
    ciu_data matrix. In Gaussian mode, it will be Gaussian information from the object's fitted Gaussian
    lists.
    :param analysis_obj: analysis objects
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :param ufs_mode: boolean, True if using for UFS (feature selection), which requires only centroids from gaussian fitting
    :return: classification data matrix
    """
    if params_obj.classif_1_input_mode == 'All_Data':
        classif_data = analysis_obj.classif_input_std

    else:
        if not ufs_mode:
            # for non-UFS, use full input (standardized) Gaussian dataset.
            classif_data = analysis_obj.classif_input_std
        else:
            # for UFS, only use centroids - remove
            centroids_by_cv = []
            for cv_index, gaussian_data_at_cv in enumerate(analysis_obj.classif_input_std.T):
                current_centroids = []
                for feat_index, gaussian_attribute in enumerate(gaussian_data_at_cv):
                    if feat_index % 3 == 0:
                        current_centroids.append(gaussian_attribute)
                centroids_by_cv.append(current_centroids)
            classif_data = np.asarray(centroids_by_cv).T    # transpose because we transposed at the start (in the loop) to access by CV

    return classif_data


def prep_gaussfeats_for_classif(features_list, analysis_obj):
    """
    Assemble a Gaussian-list-by-CV list from input features data. Fills any gaps between and within
    features with Gaussian data from the filtered_gaussians list and assembles a complete list of
    Gaussians by CV.
    :param features_list: list of Features
    :type features_list: list[Feature]
    :param analysis_obj: CIUAnalysisObj with gaussian fitting and gaussian feature detect performed
    :type analysis_obj: CIUAnalysisObj
    :return: List of (Gaussian lists) sorted by CV
    :rtype: list[list[Gaussian]]
    """
    # make an empty list for Gaussians at each CV
    final_gaussian_lists = [[] for _ in analysis_obj.axes[1]]

    features_list = close_feature_gaps(features_list, analysis_obj.axes[1])

    # iterate over features, filling any gaps within the feature and entering Gaussians into the final list
    for feature in features_list:
        # determine if the feature contains gaps
        gaussian_cvs = [gaussian.cv for gaussian in feature.gaussians]

        for cv in feature.cvs:
            # append Gaussian(s) at this CV to the final list
            try:
                cv_index = np.where(analysis_obj.axes[1] == cv)[0][0]
            except IndexError:
                # A gaussian had a CV that was not in the analysis object's CV axis! This should be caught elsewhere
                logger.warning('Gaussian had CV {}, but that CV is not in the CV axis of this file (after axes were equalized across all files). It will be ignored.'.format(cv))
                continue

            this_cv_gaussian = [x for x in feature.gaussians if x.cv == cv]
            final_gaussian_lists[cv_index].extend(this_cv_gaussian)

            if cv not in gaussian_cvs:
                # a gap is present within this feature- create a Gaussian at median centroid/values to fill it
                new_gaussian = Gaussian(amplitude=np.median([x.amplitude for x in feature.gaussians]),
                                        centroid=feature.gauss_median_centroid,
                                        width=np.median([x.width for x in feature.gaussians]),
                                        collision_voltage=cv,
                                        pcov=None,
                                        protein_bool=True)
                final_gaussian_lists[cv_index].append(new_gaussian)

    for cv_index, cv in enumerate(analysis_obj.axes[1]):
        if len(final_gaussian_lists[cv_index]) == 0:
            # no Gaussians have been added here yet, so we need to add one. Add an empty Gaussian at mean DT (not completely empty so as to avoid LAPACK crashes)
            mean_dt = np.mean(analysis_obj.axes[0])
            final_gaussian_lists[cv_index].append(Gaussian(centroid=mean_dt, amplitude=0, width=0, collision_voltage=cv, pcov=None, protein_bool=False))

    analysis_obj.classif_gaussians_by_cv = final_gaussian_lists
    return final_gaussian_lists


def close_feature_gaps(features_list, cv_axis):
    """
    Check all features for gaps in their CV lists, and fill in the gaps if any exist by inserting
    appropriate CV values
    :param features_list: list of Features
    :type features_list list[Feature]
    :param cv_axis: analysis_obj.axes[1]
    :return: updated features list with gaps closed (in feature.cvs ONLY)
    :rtype: list[Feature]
    """
    cv_step = cv_axis[1] - cv_axis[0]

    for feature in features_list:
        for index, current_cv in enumerate(feature.cvs):
            try:
                next_cv = feature.cvs[index + 1]
            except IndexError:
                # reached the end, ignore
                continue
            while not (next_cv - current_cv) == cv_step:
                # a gap is present - insert the next value to fill it
                correct_next = current_cv + cv_step
                feature.cvs.insert(index + 1, correct_next)
                next_cv = feature.cvs[index + 1]
    return features_list


def peak_crossval_score_detect(test_score_means, diff_from_max):
    """
    Determine the best set of features based on crossvalidation testing scores generated in
    crossval_main. Chooses the first 'peak' (point after which score decreases) that is within
    tolerance of the overall maximum score in the data. This is to choose the minimum number of
    features while still achieving a high score.
    :param test_score_means: list of scores for cross validation test data, in increasing order of number of features
    :param diff_from_max: maximum distance below the max value of test_score_means that a peak is allowed for selection. Default 0.05
    :return: best number of features (index of test_score_means) and score (value)
    """
    max_score = np.max(test_score_means)

    for index, value in enumerate(test_score_means):
        try:
            if test_score_means[index + 1] < value:
                # stop here (reached a peak) if within tolerance of max
                if max_score - value <= diff_from_max:
                    return index + 1, value     # index + 1 because we're determining the NUM of features, indexed from 1 (not 0)
        except IndexError:
            # reached the end of the list - return final index + 1 (because indexing from 1 for num feats) and value
            return index + 1, value


def prep_data_2d(cl_inputs_by_label, params_obj):
    """
    Wrapper method for 2D input list of ClInputs (for use in scheme construction) to ensure
    all CIU analyses ultimately have the same size Gaussian data matrices for classification.
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :return: max number of Gaussians in the input data (also sets it to params_obj)
    """
    max_num_gaussians = 0
    for input_list in cl_inputs_by_label:
        for cl_input in input_list:
            for subclass_label, analysis_obj in cl_input.subclass_dict.items():
                if not params_obj.classif_1_input_mode == 'All_Data':
                    if params_obj.classif_1_input_mode == 'Gaussian_Feat':
                        # prepare gaussian features if using feature mode for classification (saves to container)
                        gaussians_by_cv = prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
                    else:
                        # Gaussian raw mode, so Gaussians by CV comes directly from the container
                        gaussians_by_cv = analysis_obj.raw_protein_gaussians
                        analysis_obj.classif_gaussians_by_cv = gaussians_by_cv

                    # update the max number of gaussians if necessary
                    for gaussian_list in gaussians_by_cv:
                        if len(gaussian_list) > max_num_gaussians:
                            max_num_gaussians = len(gaussian_list)
                    # save num Gaussians to ensure all matrices same size
                    params_obj.silent_clf_4_num_gauss = max_num_gaussians
                else:
                    # all data mode - initialize raw data for classification
                    analysis_obj.classif_input_raw = analysis_obj.ciu_data

    if not params_obj.classif_1_input_mode == 'All_Data':
        # second pass required for Gaussians to format the final input data matrix (now that we know that max num of Gaussians)
        for input_list in cl_inputs_by_label:
            for cl_input in input_list:
                for subclass_label, analysis_obj in cl_input.subclass_dict.items():
                    input_classif_raw = prep_gaussian_input_raw(analysis_obj.classif_gaussians_by_cv, max_num_gaussians)
                    analysis_obj.classif_input_raw = input_classif_raw
    return max_num_gaussians


def prep_gaussian_input_raw(gaussians_by_cv, max_num_gaussians, selected_cvs=None):
    """
    Assemble a 2D numpy array of correct final dimensions from a list of Gaussians at
    each collision voltage. Selected CVs can be provided (e.g. for unknown data)
    :param gaussians_by_cv:
    :type gaussians_by_cv: list[list[Gaussian]
    :param max_num_gaussians: maximum number of Gaussians in the classifying scheme
    :param selected_cvs: list of CVs to consider for unknown data
    :return: 2D numpy array of formatted Gaussian information for input to standardization/classification
    """
    classif_data = []
    for gaussian_list in gaussians_by_cv:
        # skip any non-selected CVs if requested (i.e. in unknown analysis mode)
        if selected_cvs is not None:
            if not gaussian_list[0].cv in selected_cvs:
                continue
        attributes = ['cent', 'width', 'amp']
        attribute_list = np.zeros(max_num_gaussians * len(attributes))

        attribute_index = 0
        if len(gaussian_list) == 0:
            continue
        for gaussian in gaussian_list:
            attribute_list[attribute_index] = gaussian.centroid
            attribute_index += 1
            attribute_list[attribute_index] = gaussian.width
            attribute_index += 1
            attribute_list[attribute_index] = gaussian.amplitude
            attribute_index += 1

        classif_data.append(attribute_list)
    # classif_data = np.asarray(classif_data).T
    classif_data = np.asarray(classif_data)

    return classif_data


def standardize_all_2d(cl_inputs_by_label, params_obj):
    """
    Standardization wrapper to standardize across the complete input dataset. Intended to be
    called prior to any UFS or crossval. Saves standardized data into a field in the CIUAnalysisObj
    contained within all classification inputs for later reference. Returns the standardization
    information (mean, stdev for each input feature) to be saved into the scheme to allow for
    unknown standardization. Data is saved in array by feature, then CV.
    NOTE: assumes axis equalization has already been performed
    NOTE: assumes Gaussian feature prep has already been performed in using Gaussian mode
    :param cl_inputs_by_label: lists of ClInput containers sorted by class label
    :type cl_inputs_by_label: list[list[ClInput]]
    :param params_obj: Parameters object with classification parameter information
    :type params_obj: Parameters
    :return: input 2D cl_input list, mean/stdev matrices used for standardization
    """
    # Read input dimensions and prepare mean/stdev dataframes
    example_input = cl_inputs_by_label[0][0]
    example_obj = list(example_input.subclass_dict.values())[0]
    subclass_labels = list(example_input.subclass_dict.keys())
    cv_axis = example_obj.axes[1]
    feature_axis = get_feature_axis(example_obj, params_obj.classif_1_input_mode, params_obj.classif_93_std_all_gsns_bool)

    means = {x: pandas.DataFrame(np.zeros((len(feature_axis), len(cv_axis))), index=feature_axis, columns=cv_axis) for x in subclass_labels}
    stdevs = {x: pandas.DataFrame(np.zeros((len(feature_axis), len(cv_axis))), index=feature_axis, columns=cv_axis) for x in subclass_labels}

    # Assemble the data across all classes to standardize by feature/cv/subclass
    for cv_index, cv in enumerate(cv_axis):
        for feature_index, feature in enumerate(feature_axis):
            output_data_by_subcl = {x: [] for x in subclass_labels}

            for class_clinput_list in cl_inputs_by_label:
                for cl_input in class_clinput_list:
                    for subclass_label, analysis_obj in cl_input.subclass_dict.items():
                        if not params_obj.classif_1_input_mode == 'All_Data':
                            if params_obj.classif_93_std_all_gsns_bool:
                                # combining all Gaussians to standardize together - get every third feature (all centroids or widths or amplitudes, depending on feat_index)
                                features = analysis_obj.classif_input_raw[cv_index][feature_index::3]
                                output_data_by_subcl[subclass_label].extend(features)
                            else:
                                # Individual Gaussian mode, append data as normal
                                output_data_by_subcl[subclass_label].append(analysis_obj.classif_input_raw[cv_index][feature_index])
                        else:
                            # In raw data mode, average across whole ATD rather than including each point
                            cv_col_matrix = np.swapaxes(analysis_obj.classif_input_raw, 0, 1)
                            cv_col_atd = cv_col_matrix[cv_index]
                            output_data_by_subcl[subclass_label].extend(cv_col_atd)

            # compute mean/std
            for subclass_label, data_array in output_data_by_subcl.items():
                means[subclass_label].loc[feature, cv] = np.mean(data_array)
                stdevs[subclass_label].loc[feature, cv] = np.std(data_array)

    # now that means/stdevs are known, loop over the data again to standardize using that information
    for class_clinput_list in cl_inputs_by_label:
        for cl_input in class_clinput_list:
            cl_input = standardize_ciu_obj(cl_input, means, stdevs, params_obj)

    return cl_inputs_by_label, means, stdevs


def get_feature_axis(ciu_obj, gaussian_mode, combine_gaussians_bool):
    """
    Helper method to return a "feature" axis for standardization. In raw data mode, simply returns a
    list of 1 entry, as each raw DT profile gets standardized as a single feature. In Gaussian mode,
    generates a list of attributes (c=centroid, a=amplitude, w=width) for each Gaussian up to the max
    number as the feature axis
    :param ciu_obj: CIUAnalysisObj with classif_input_raw initialized
    :type ciu_obj: CIUAnalysisObj
    :param gaussian_mode: 'Gaussian_Feat', 'Gaussian_Raw', or 'All_Data'
    :param combine_gaussians_bool: whether standardizing Gaussians together or separately
    :return: list of feature axis values (strings)
    """
    if not gaussian_mode == 'All_Data':
        if combine_gaussians_bool:
            feature_axis = ['c', 'w', 'a']
        else:
            # Individual Gaussian mode: num features per CV = 3 * max_num_gaussians (centroid, width, amplitude for each)
            feature_axis_len = len(ciu_obj.classif_input_raw[0])
            attributes = ['c', 'w', 'a']
            feature_axis = []
            for feat_index in range(feature_axis_len):
                att_type = feat_index % 3
                gaussian_index = feat_index // 3 + 1
                feat_string = '{}{}'.format(attributes[att_type], gaussian_index)
                feature_axis.append(feat_string)
    else:
        feature_axis = ['raw']
    return feature_axis


def standardize_ciu_obj(cl_input, means_dict, stdevs_dict, params_obj):
    """
    Helper method for final standardization of a single ClInput replicate container. Uses
    provided means/stdevs to standardize the input data and saves it into the CIUAnalysisObj(s)
    in the classif_input_std field.
    :param cl_input:
    :type cl_input: ClInput
    :param means_dict: 3D xarray of subclass, feature, CV organized means
    :type means_dict: dict[str: pandas.DataFrame]
    :param stdevs_dict: 3D xarray of subclass, feature, CV organized standard deviations
    :type stdevs_dict: dict[str: pandas.DataFrame]
    :param params_obj: parameters container
    :type params_obj: Parameters
    :return: cl_input with updated information
    :rtype: ClInput
    """
    for subclass_label, analysis_obj in cl_input.subclass_dict.items():
        # can return just the feat/CV 2D array from the input dataframes and use the original method
        means = means_dict[subclass_label].values
        stdevs = stdevs_dict[subclass_label].values
        cv_axis_len = len(means[0])

        standardized_data = np.zeros(np.shape(analysis_obj.classif_input_raw))
        raw_std_data = []
        for cv_index in range(cv_axis_len):
            # get correct feature axis shape
            if params_obj.classif_1_input_mode == 'All_Data':
                feature_axis_len = len(means)
            else:
                feature_axis_len = len(analysis_obj.classif_input_raw[cv_index])

            for feature_index in range(feature_axis_len):
                if not params_obj.classif_1_input_mode == 'All_Data':
                    if params_obj.classif_93_std_all_gsns_bool:
                        # combined Gaussians mode - use single mean/standard for all Gaussians
                        this_feat_mean = means[feature_index % 3][cv_index]
                        this_feat_stdev = stdevs[feature_index % 3][cv_index]
                    else:
                        # separate Gaussians mode - use unique mean/standard for each Gaussian
                        this_feat_mean = means[feature_index][cv_index]
                        this_feat_stdev = stdevs[feature_index][cv_index]
                    raw_datapoint = analysis_obj.classif_input_raw[cv_index][feature_index]
                    standardized_data[cv_index][feature_index] = standardize_data(raw_datapoint, this_feat_mean, this_feat_stdev, params_obj.classif_92_standardize)
                else:
                    # raw data mode - need to average across whole CV column
                    this_feat_mean = means[feature_index][cv_index]
                    this_feat_stdev = stdevs[feature_index][cv_index]
                    raw_data_col = np.swapaxes(analysis_obj.classif_input_raw, 0, 1)[cv_index]
                    std_data_col = standardize_data(raw_data_col, this_feat_mean, this_feat_stdev, params_obj.classif_92_standardize)
                    raw_std_data.append(std_data_col)
        if params_obj.classif_1_input_mode == 'All_Data':
            standardized_data = np.swapaxes(raw_std_data, 0, 1)
        else:
            standardized_data = standardized_data.T
        analysis_obj.classif_input_std = standardized_data
    return cl_input


def standardize_data(datapoint, mean, stdev, standardize_bool):
    """
    Standardize the input CIU data using the common (xi - x_mean) / stdev approach. Datapoint
    can be an input value or array.
    :param datapoint: 2D numpy array of CIU data OR single point (feature/CV combination for Gaussian data)
    :param mean: feature mean for standardization
    :param stdev: feature standard deviation
    :param standardize_bool: whether to standardize (method does nothing if False)
    :return: standardized data in same format as datapoint (input)
    """
    # escape method for testing/comparison
    if not standardize_bool:
        return datapoint

    if stdev == 0:
        # should only occur in cases where data was empty (value of 0) to begin with, as otherwise all replicates would have to have same value to numerical precision
        std_data = 0
        # std_data = mean
    else:
        std_data = (datapoint - mean) / stdev
    return std_data


def reduce_means_for_unk_standardize(unk_cl_input, input_means, input_stds):
    """
    Cuts the mean/stdev DataArrays down to only the size provided in the unknown to allow for standardization
    of reduced data using same methods as for training data.
    :param unk_cl_input: ClInput with unknown data
    :type unk_cl_input: ClInput
    :param input_means: matrix of means from scheme
    :type input_means: dict[str: pandas.DataFrame]
    :param input_stds: matrix of stdevs from scheme
    :type input_stds: dict[str: pandas.DataFrame]
    :return: reduced size mean/stdev arrays
    """
    for subclass_label, analysis_obj in unk_cl_input.subclass_dict.items():
        # include only CVs found in the unknown in case of reduced dataset
        cvs = analysis_obj.axes[1]
        input_means[subclass_label] = input_means[subclass_label].loc[:, cvs]
        input_stds[subclass_label] = input_stds[subclass_label].loc[:, cvs]

    return input_means, input_stds


def subclass_inputs_from_class_inputs(cl_inputs_by_label, subclass_label_list, class_labels):
    """
    Generate and return a list of ClassifInput containers (organized by subclass) from the primary
    input of replicates organized by class label.
    :param cl_inputs_by_label: list of lists of ClInput containers with any subclasses present
    :type cl_inputs_by_label: list[list[ClInput]]
    :param subclass_label_list: list of subclass label strings to search for (strings)
    :param class_labels: list of class labels (strings)
    :return: list of ClassifInputs
    :rtype: list[ClassifInput]
    """
    # Generate subclass oriented files for UFS from the primary input
    classif_inputs = []
    for subclass_label in subclass_label_list:
        objs_by_label = []
        for class_list in cl_inputs_by_label:
            obj_list_this_label = []
            for cl_input in class_list:
                subclass_obj = cl_input.subclass_dict[subclass_label]
                obj_list_this_label.append(subclass_obj)
            objs_by_label.append(obj_list_this_label)
        classif_input = ClassifInput(class_labels, objs_by_label, subclass_label)
        classif_inputs.append(classif_input)
    return classif_inputs


def get_unique_labels(label_list):
    """
    Return a list of unique labels (i.e. without duplicates) from the provided label list
    :param label_list: list of labels (strings)
    :return: list of unique labels (strings)
    """
    unique_labels = []
    for label in label_list:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


def prep_outputs_by_file_new(x_lda, y_pred, probs, flat_subset_list):
    """
    Organize outputs from LDA into groups sorted by input raw file for easy output formatting.
    All 4 input lists are same shape and would be made into a nice container object except that
    they come out all together from classification.
    :param x_lda: transformed LDA result data
    :param y_pred: predicted class outputs
    :param probs: predicted class probabilities
    :param flat_subset_list: list input data from DataSubset containers
    :type flat_subset_list: list[DataSubset]
    :return: sublist of each input list that is all the entries corresponding to the specific file
    """
    x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, combined_filenames = [], [], [], [], []

    num_features = len(flat_subset_list[0].features)

    # loop over each input file (subset) and organize data accordingly
    output_column_index = 0
    for subset in flat_subset_list:
        # each file has num_features columns of data in the output matrix - get those
        file_xlda, file_ypred, file_probs = [], [], []
        for feat_index in range(num_features):
            file_xlda.append(x_lda[output_column_index])
            file_ypred.append(y_pred[output_column_index])
            file_probs.append(probs[output_column_index])
            output_column_index += 1
        x_lda_by_file.append(file_xlda)
        y_pred_by_file.append(file_ypred)
        probs_by_file.append(file_probs)
        filenames_by_file.append(subset.file_id)
        combined_filenames.append(subset.all_filenames)

    return x_lda_by_file, y_pred_by_file, probs_by_file, filenames_by_file, combined_filenames


def generate_scheme_name(class_labels, subclass_labels):
    """
    Generate the name of a classifying scheme to be a combination of the provided class labels
    :param class_labels: list of strings
    :param subclass_labels: list of strings
    :return: string name
    """
    # Get class labels
    unique_labels = get_unique_labels(class_labels)

    # add number of subclasses if applicable
    len_sublabels = len(subclass_labels)
    if len_sublabels > 1:
        name = '_'.join(unique_labels) + '_{}SubCl'.format(len_sublabels)
    else:
        name = '_'.join(unique_labels)
    return name


def plot_feature_scores_subclass(features_by_subclass, params_obj, scheme_name, output_path):
    """
    Plot feature score by collision voltage
    :param features_by_subclass: list of CFeatures
    :type features_by_subclass: list[list[CFeature]]
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param output_path: directory in which to save output
    :return: void
    """
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    colors = ['black', 'blue', 'green', 'orange', 'red', 'purple', 'yellow', 'pink', 'lightblue']
    for index, feature_list in enumerate(features_by_subclass):
        mean_scores = [x.mean_score for x in feature_list]
        std_scores = [x.std_dev_score for x in feature_list]
        cv_axis = [x.cv for x in feature_list]

        try:
            color = colors[index]
        except IndexError:
            color = 'black'
        plt.errorbar(x=cv_axis, y=mean_scores, yerr=std_scores, ls='none', marker='o', color=color, markersize=params_obj.plot_14_dot_size, markeredgecolor='black', alpha=0.8, label=feature_list[0].subclass_label)
        plt.axhline(y=0.0, color='black', ls='--')

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
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('-Log10(p-value)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(output_path, scheme_name + '_UFS' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_crossval_scores(crossval_data, scheme_name, params_obj, outputdir):
    """
    Make plots of mean and std dev scores for training and test data from cross validation.
    :param crossval_data: tuple of (training means, training stds, test means, test stds) lists
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param outputdir: directory in which to save output
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]
    auc_means = np.asarray(crossval_data[4])
    auc_stds = np.asarray(crossval_data[5])

    xax = np.arange(1, len(train_score_means) + 1)
    plt.plot(xax, train_score_means, color='blue', marker='s', label='train_score', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.fill_between(xax, train_score_means-train_score_stds, train_score_means+train_score_stds, color='blue', alpha=0.2)
    plt.plot(xax, test_score_means, color='green', marker='o', label='test_score', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.fill_between(xax, test_score_means-test_score_stds, test_score_means+test_score_stds, color='green', alpha=0.2)
    # add AUC if requested
    if params_obj.classif_5_show_auc_crossval:
        plt.plot(xax, auc_means, color='red', marker='o', label='AUC', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
        plt.fill_between(xax, auc_means - auc_stds, auc_means + auc_stds, color='red', alpha=0.2)

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
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel('Number of Features (Collision Voltages)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Accuracy Ratio', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(outputdir, scheme_name + '_crossval' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_crossval_auc(auc_means, auc_stds, scheme_name, params_obj, outputdir):
    """
    Make plot of AUC (mean +/- std) for the ROC curves for each number of features considered. Same
    general shape as crossvalidation accuracy plot, but with AUC instead of accuracy.
    :param auc_means: list of AUC mean by number of features
    :param auc_stds: list of AUC std deviation by number of features
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param scheme_name: (string) name of scheme for labeling purposes
    :param outputdir: directory in which to save output
    :return: void
    """
    auc_means = np.asarray(auc_means)
    auc_stds = np.asarray(auc_stds)

    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    x_axis = np.arange(1, len(auc_means) + 1)
    plt.plot(x_axis, auc_means, color='red', marker='o', label='AUC', markersize=params_obj.plot_14_dot_size, markeredgecolor='black')
    plt.fill_between(x_axis, auc_means - auc_stds, auc_means + auc_stds, color='red', alpha=0.2)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = scheme_name
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel('Number of Features (Collision Voltages)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Area Under ROC Curve (AUC)', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

    output_name = os.path.join(outputdir, scheme_name + '_AUC' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_probabilities(params_obj, scheme, class_probs_by_file, output_path, unknown_bool):
    """
    Generate a stacked bar graph of classification output probabilities for crossval data or unknowns.
    Organized such that the most likely class is on top of each stacked bar, and that the probabilities
    sum to 100%.
    :param params_obj: parameter container with plot options
    :type params_obj: Parameters
    :param scheme: Classification scheme with label information
    :type scheme: ClassificationScheme
    :param class_probs_by_file: list of lists of probabilities for each class (i.e. [file 1=[cv1[class1, class2, ...], cv2, ...], class 2, ...], file2, ...])
    :param output_path: directory in which to save the plot
    :param unknown_bool: if saving unknown data or not (to label output)
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # prepare colors and labels
    initial_colors = ['dodgerblue', 'orange', 'fuchsia', 'forestgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']
    colors, labels = [], []
    for index, label in enumerate(scheme.unique_labels):
        colors.append(initial_colors[index])
        labels.append(label)

    # Stack each class probability in the bar graph
    for file_index in range(len(class_probs_by_file)):
        avg_probs = np.average(class_probs_by_file[file_index], axis=0)
        prob_class_tups = [(ind, val) for ind, val in enumerate(avg_probs)]
        sorted_probs = sorted(prob_class_tups, key=lambda x: x[1])
        y_offset = 0
        bar_index = 0
        for class_index, class_prob in sorted_probs:
            if bar_index > 0:
                y_offset += sorted_probs[bar_index - 1][1]
            plt.bar(file_index, class_prob, bottom=y_offset, color=colors[class_index], width=0.6)
            bar_index += 1

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(labels, loc='center left', bbox_to_anchor=(1.2, 0.5), fontsize=params_obj.plot_13_font_size)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = 'Probabilities by Class for {}'.format(scheme.name)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel('Sample Number', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Probability', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_07_show_legend:
        handles = []
        for index, color in enumerate(colors):
            patch_artist = matplotlib.patches.Patch(color=color, label=labels[index])
            handles.append(patch_artist)
        plt.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=params_obj.plot_13_font_size)
    num_files = len(class_probs_by_file)
    plt.xticks(np.arange(num_files), np.arange(1, num_files + 1), fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)

    if unknown_bool:
        output_name = os.path.join(output_path, scheme.name + '_probs-unknown' + params_obj.plot_02_extension)
    else:
        output_name = os.path.join(output_path, scheme.name + '_probs' + params_obj.plot_02_extension)

    try:
        plt.savefig(output_name, bbox_inches='tight')
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def plot_classification_decision_regions(class_scheme, params_obj, output_path, unknown_tups=None):
    """
    Make a plot of decision regions determined for this classification scheme and the locations
    of the input test (validation) data against the constructed scheme.
    :param class_scheme: Classification object
    :param output_path: directory in which to save plot
    :param params_obj: Parameters object with plot information
    :type params_obj: Parameters
    :param unknown_tups: tuples of unknown (data, label). data = transformed lda data for unknown
    :return: void
    """
    shape_lda = np.shape(class_scheme.transformed_test_data)
    if shape_lda[1] > 3:
        # do not plot images with more than 3 dimensions
        return
    fig = plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # markers = ('s', 'o')
    markers = ('s', 'o', '^', 'v', 'D', '<', '>', '4', '8', 'h', 'H', '1', '2', '3', '+', '*', 'p', 'P', 'x')
    # colors = ['deepskyblue', 'orange', 'fuchsia', 'mediumspringgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']
    colors = ['dodgerblue', 'orange', 'fuchsia', 'forestgreen', 'gray', 'cyan', 'lightgreen', 'magenta', 'yellow']

    cmap = ListedColormap(colors[:len(class_scheme.unique_labels)])
    # decide whether the data has 1d or nds
    ax = plt.subplot(111)

    # plot 1D or 2D decision regions
    if shape_lda[1] == 1:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data.min()), np.ceil(class_scheme.transformed_test_data.max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                min_unk = unknown_tup[0].min()
                max_unk = unknown_tup[0].max()
                x1_min = min([min_unk, x1_min])
                x1_max = max([max_unk, x1_max])

        # create a grid to evaluate model
        x2_min, x2_max = -1, 1
        x_grid, y_grid = np.mgrid[x1_min:x1_max:100j, x2_min:x2_max:100j]
        z = class_scheme.classifier.predict(x_grid.ravel().reshape(-1, 1))
        z = z.reshape(x_grid.shape)

        plt.contourf(x_grid, y_grid, z, alpha=0.4, cmap=cmap)
        plot_sklearn_lda_1ld(class_scheme, markers, colors, params_obj.plot_14_dot_size)
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel('LD1', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks([])
        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                plt.scatter(unknown_tup[0], np.zeros(np.shape(unknown_tup[0])), marker=marker, color='black',
                            alpha=0.5, label=unknown_tup[1], s=params_obj.plot_14_dot_size ** 2 * 2, edgecolors='black')

    if shape_lda[1] == 2:
        x1_min, x1_max = np.floor(class_scheme.transformed_test_data[:, 0].min()), np.ceil(class_scheme.transformed_test_data[:, 0].max())
        x2_min, x2_max = np.floor(class_scheme.transformed_test_data[:, 1].min()), np.ceil(class_scheme.transformed_test_data[:, 1].max())
        if unknown_tups is not None:
            for unknown_tup in unknown_tups:
                x1_min_unk, x1_max_unk = np.floor(unknown_tup[0][:, 0].min()), np.ceil(unknown_tup[0][:, 0].max())
                x2_min_unk, x2_max_unk = np.floor(unknown_tup[0][:, 1].min()), np.ceil(unknown_tup[0][:, 1].max())
                x1_min = min([x1_min_unk, x1_min])
                x1_max = max([x1_max_unk, x1_max])
                x2_min = min([x2_min_unk, x2_min])
                x2_max = max([x2_max_unk, x2_max])

        num_grid_bins = 1000
        x_grid_1, x_grid_2 = np.meshgrid(np.arange(x1_min, x1_max, abs(x1_max - x1_min) / num_grid_bins), np.arange(x2_min, x2_max, abs(x2_max - x2_min) / num_grid_bins))
        z = class_scheme.classifier.predict(np.array([x_grid_1.ravel(), x_grid_2.ravel()]).T)
        z = z.reshape(x_grid_1.shape)

        plt.contourf(x_grid_1, x_grid_2, z, alpha=0.4, cmap=cmap)
        plot_sklearn_lda_2ld(class_scheme, markers, colors, params_obj.plot_08_show_axes_titles, params_obj.plot_14_dot_size)
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel('LD1', fontsize=params_obj.plot_13_font_size, fontweight='bold')
            plt.ylabel('LD2', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks(fontsize=params_obj.plot_13_font_size)
        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                plt.scatter(x=unknown_tup[0][:, 0], y=unknown_tup[0][:, 1], marker=marker, color='black',
                            alpha=0.5, label=unknown_tup[1], s=params_obj.plot_14_dot_size ** 2, edgecolors='black')

    if shape_lda[1] == 3:
        logger.info('NOTE: 3D plots are not fully optimized. Labels and font sizes may not be perfect.')
        ax = Axes3D(fig)
        plot_data = class_scheme.transformed_test_data
        y_values = class_scheme.numeric_labels
        unique_labels = class_scheme.unique_labels

        for label, marker, color in zip(range(0, len(unique_labels)), markers, colors):
            ax.scatter(xs=plot_data[:, 0][y_values == label + 1],
                       ys=plot_data[:, 1][y_values == label + 1],
                       zs=plot_data[:, 2][y_values == label + 1],
                       # marker=marker,
                       c=color, s=params_obj.plot_14_dot_size ** 2, edgecolors='black',
                       label=unique_labels[label])

        if unknown_tups is not None:
            for ind, unknown_tup in enumerate(unknown_tups):
                if ind >= len(markers):
                    marker = 'x'
                else:
                    marker = markers[ind]
                ax.scatter(xs=unknown_tup[0][:, 0],
                           ys=unknown_tup[0][:, 1],
                           zs=unknown_tup[0][:, 2],
                           marker=marker, c='black',
                           s=params_obj.plot_14_dot_size ** 2, edgecolors='black', alpha=0.9, label=unknown_tup[1])

        if params_obj.plot_08_show_axes_titles:
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')
        # EDIT THIS IF YOU WANT TO CHANGE THE ANGLE OF THE PLOT
        # ax.view_init(elev=20., azim=30.)

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        cv_string = ', '.join([str(np.round(x.cv, 1)) for x in class_scheme.selected_features])
        plot_title = 'From CVs: {}'.format(cv_string)
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_07_show_legend:
        ax.legend(loc='best', fontsize=params_obj.plot_13_font_size * 0.75)

    if unknown_tups is not None:
        output_name = os.path.join(output_path, class_scheme.name + '_Results-unknowns' + params_obj.plot_02_extension)
    else:
        output_name = os.path.join(output_path, class_scheme.name + '_Results' + params_obj.plot_02_extension)
    try:
        plt.savefig(output_name)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_name))
        plt.savefig(output_name)
    plt.close()


def plot_sklearn_lda_2ld(class_scheme, marker, color, label_axes, dot_size):
    """
    Scatter plot the class data onto a decision regions plot using provided colors/markers/etc
    :param class_scheme: scheme containing data and label information
    :param marker: list of marker styles to plot for each class
    :param color: list of colors to plot for each class
    :param dot_size: marker size in standard plot units (will be squared for scatter plot)
    :param label_axes: boolean: whether to label axes
    :return: void
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x=x_data[:, 0][y_values == label + 1], y=x_data[:, 1][y_values == label + 1], marker=marker, color=color,
                    alpha=0.9, label=unique_labels[label], s=dot_size ** 2, edgecolors='black')
    if label_axes:
        plt.xlabel('LD1')
        plt.ylabel('LD2')


def plot_sklearn_lda_1ld(class_scheme, marker, color, dot_size):
    """
    Scatter plot the class data onto a decision regions plot using provided colors/markers/etc
    :param class_scheme: scheme containing data and label information
    :param marker: list of marker styles to plot for each class
    :param color: list of colors to plot for each class
    :param dot_size: marker size in standard plot units (will be squared for scatter plot)
    :return: void
    """
    x_data = class_scheme.transformed_test_data
    y_values = class_scheme.numeric_labels
    unique_labels = class_scheme.unique_labels

    for label, marker, color in zip(range(0, len(unique_labels)), marker, color):
        plt.scatter(x=x_data[:, 0][y_values == label + 1], y=np.zeros(np.shape(x_data[:, 0][y_values == label + 1])), marker=marker, color=color,
                    alpha=0.9, label=unique_labels[label], s=dot_size ** 2 * 2, edgecolors='black')


def save_feature_scores(features_by_subclass, scheme_name, output_path):
    """
    Save CSV output of feature scores
    :param features_by_subclass: list of CFeatures
    :type features_by_subclass: list[list[CFeature]]
    :param scheme_name: (string) name of scheme for labeling purposes
    :param output_path: directory in which to save output
    :return: void
    """
    outfilename = os.path.join(output_path, scheme_name + '_UFS.csv')
    output_string = 'UFS (Feature Selection) Results\n'

    if features_by_subclass[0][0].subclass_label is not '0':
        lineheader = 'SubClass,Collision Voltage (V),Mean Score,StDev Score\n'
    else:
        lineheader = 'Collision Voltage (V),Mean Score,StDev Score\n'
    output_string += lineheader
    for feature_list in features_by_subclass:
        # if feature_list[0].subclass_label is not '0':
        #     output_string += 'Subclass: {}\n'.format(feature_list[0].subclass_label)
        for feature in feature_list:
            if feature_list[0].subclass_label is not '0':
                line = '{},{},{:.4f},{:.4f},{:.4f}\n'.format(feature_list[0].subclass_label, feature.cv, feature.mean_score, feature.std_dev_score, feature.mean_score - feature.std_dev_score)
            else:
                line = '{},{:.4f},{:.4f},{:.4f}\n'.format(feature.cv, feature.mean_score, feature.std_dev_score, feature.mean_score - feature.std_dev_score)
            output_string += line

    try:
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving',
                             'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(
                                 outfilename))
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)
    return outfilename


def save_lda_and_predictions(scheme, lda_transform_data_by_file, predicted_class_by_file, probs_by_file, filenames, combined_filenames, output_path, unknowns_bool):
    """
    Unified CSV output method for saving classification results (data transformed along LD axes and predicted
    classes and probabilities from the SVM). Used both when building a classification scheme and when
    classifying unknowns.
    :param scheme: ClassificationScheme container
    :type scheme: ClassificationScheme
    :param lda_transform_data_by_file: list for each input file of transformed data for each LD axis (i.e. [file1=[LD1, LD2], file2=[LD1, LD2]])
    :param predicted_class_by_file: list of lists of predicted classes, same format as lda_transform data (except listed by each class, not each LD axis)
    :param probs_by_file: list of lists of probabilities for each class, same format as predicted class data
    :param filenames: list of input filenames
    :param combined_filenames: list of combined input filenames (for subclass mode). In regular mode, will be same as filenames
    :param output_path: directory in which to save output
    :param unknowns_bool: True if saving unknown data
    :return: void
    """
    subclass_mode = scheme.get_subclass_mode()
    if unknowns_bool:
        outputname = os.path.join(output_path, scheme.name + '_Results-unknowns.csv')
    else:
        outputname = os.path.join(output_path, scheme.name + '_Results.csv')

    # arrange header information
    predict_labels = ','.join(['{} Probability'.format(x) for x in scheme.unique_labels])
    ld_dim_list = np.arange(1, len(lda_transform_data_by_file[0][0]) + 1)
    lda_header = ','.join('LD {} (linear discriminant dimension {})'.format(x, x) for x in ld_dim_list)
    if subclass_mode:
        header = 'File ID,All Subclass Files,Feature,Feature Subclass,{},Class Prediction,{}\n'.format(lda_header, predict_labels)
    else:
        header = 'File,Feature,{},Class Prediction,{}\n'.format(lda_header, predict_labels)
    output_string = header

    # loop over each file and each CV ("feature")
    for file_index in range(len(lda_transform_data_by_file)):
        transforms_by_cv = lda_transform_data_by_file[file_index]
        classes_by_cv = predicted_class_by_file[file_index]
        probs_by_cv = probs_by_file[file_index]
        if subclass_mode:
            file_string = '{},{}'.format(filenames[file_index], combined_filenames[file_index])
        else:
            file_string = '{}'.format(filenames[file_index])

        for feat_index in range(len(transforms_by_cv)):
            if subclass_mode:
                feat_string = '{},{}'.format(scheme.selected_features[feat_index].cv, scheme.selected_features[feat_index].subclass_label)
            else:
                feat_string = '{}'.format(scheme.selected_features[feat_index].cv)
            ld_data = ','.join(['{:.2f}'.format(x) for x in transforms_by_cv[feat_index]])
            class_pred = scheme.unique_labels[classes_by_cv[feat_index] - 1]
            probs_data = ','.join(['{:.3f}'.format(x) for x in probs_by_cv[feat_index]])

            line = '{},{},{},{},{}\n'.format(file_string, feat_string, ld_data, class_pred, probs_data)
            output_string += line

        # Add 'combined' probabilities at end of each file
        class_mode = scheme.unique_labels[np.argmax(np.bincount(classes_by_cv)) - 1]
        avg_probs = ','.join(['{:.3f}'.format(x) for x in np.average(probs_by_file[file_index], axis=0)])
        buffer_string = ','.join(['' for _ in range(len(ld_dim_list))])     # don't write anything into LD columns
        if subclass_mode:
            output_string += '{},{},{},{},{},{}\n'.format(file_string, 'Combined', '', buffer_string, class_mode, avg_probs)
        else:
            output_string += '{},{},{},{},{}\n'.format(file_string, 'Combined', buffer_string, class_mode, avg_probs)

    # Add explained variance ratio at the end of all files
    if scheme.explained_variance_ratio is not None:
        joined_exp_var = ','.join(['{:.3f}'.format(x) for x in scheme.explained_variance_ratio])
        output_string += 'Explained_variance_ratio,{},{}\n'.format(' ', joined_exp_var)

    # save to file
    try:
        with open(outputname, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outputname))
        with open(outputname, 'w') as outfile:
            outfile.write(output_string)


def save_crossval_score(crossval_data, scheme_name, outputpath):
    """
    Save crossvalidation data output to file at path provided
    :param crossval_data: tuple of (training means, training stds, test means, test stds) lists
    :param scheme_name: (string) name of scheme for labeling purposes
    :param outputpath: directory in which to save output
    :return: filepath saved
    """
    train_score_means = crossval_data[0]
    train_score_stds = crossval_data[1]
    test_score_means = crossval_data[2]
    test_score_stds = crossval_data[3]
    outfilename = os.path.join(outputpath, scheme_name + '_crossval.csv')
    output_string = 'Cross Validation Accuracy results\n'

    lineheader = 'Num_Features, train_score_mean, train_score_std, test_score_mean, test_score_std, \n'
    output_string += lineheader
    for ind in range(len(train_score_means)):
        line = '{}, {}, {}, {}, {}, \n'.format(ind+1, train_score_means[ind], train_score_stds[ind],
                                               test_score_means[ind], test_score_stds[ind])
        output_string += line

    try:
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outfilename))
        with open(outfilename, 'w') as outfile:
            outfile.write(output_string)
    return outfilename


def save_roc_data(roc_data, outputname):
    """
    Save ROC information from scheme training to CSV file for reference. Opens previously saved
    crossval.csv file in append mode.
    :param roc_data: dictionary of lists of ROC/AUC data. See CrossValRun container save_results
    method for more information
    :param outputname: full file path to append to
    :return: void
    """
    tpr_class_mean, tpr_class_std, roc_auc_class_mean, roc_auc_class_std = roc_data['tpr_class_mean'], roc_data['tpr_class_std'], roc_data['roc_auc_class_mean'], roc_data['roc_auc_class_std']
    tpr_micro_mean, tpr_micro_std, roc_auc_micro_mean, roc_auc_micro_std = roc_data['tpr_micro_mean'], roc_data['tpr_micro_std'], roc_data['roc_auc_micro_mean'], roc_data['roc_auc_micro_std']
    tpr_macro_mean, tpr_macro_std, roc_auc_macro_mean, roc_auc_macro_std = roc_data['tpr_macro_mean'], roc_data['tpr_macro_std'], roc_data['roc_auc_macro_mean'], roc_data['roc_auc_macro_std']
    num_classes = len(tpr_class_mean[0])

    # outfilename = os.path.join(outputpath, scheme_name + '_auc.csv')
    output_string = '\nAUC (Area Under ROC Curve) Results\n'

    # write header
    if num_classes == 1:
        lineheader = 'Num_Features,AUC Avg,AUC StDev\n'
    else:
        class_header1 = ','.join(['Class {} Avg'.format(x + 1) for x in range(num_classes)])
        class_header2 = ','.join(['Class {} StDev'.format(x + 1) for x in range(num_classes)])
        lineheader = 'Num_Features,{},{},Micro Avg,Micro StDev,Macro Avg,Macro StDev\n'.format(class_header1, class_header2)
    output_string += lineheader

    for ind in range(len(tpr_class_mean)):
        if num_classes == 1:
            line = '{},{:.4f},{:.4f}\n'.format(ind + 1, roc_auc_class_mean[ind][0], roc_auc_class_std[ind][0])
        else:
            class_auc_means = ','.join(['{:.4f}'.format(x) for x in roc_auc_class_mean[ind]])
            class_auc_stds = ','.join(['{:.4f}'.format(x) for x in roc_auc_class_std[ind]])
            line = '{},{},{},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(ind + 1, class_auc_means, class_auc_stds, roc_auc_micro_mean[ind], roc_auc_micro_std[ind], roc_auc_macro_mean[ind], roc_auc_macro_std[ind])
        output_string += line

    try:
        with open(outputname, 'a') as outfile:
            outfile.write(output_string)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outputname))
        with open(outputname, 'a') as outfile:
            outfile.write(output_string)


def save_scheme(scheme, outputdir, subclass_labels):
    """
    Save a ClassificationScheme object into the provided output directory using pickle
    :param scheme: classification object to save
    :type scheme: ClassificationScheme
    :param outputdir: directory in which to save output
    :param subclass_labels: labels for output name
    :return: void
    """
    unique_labels = get_unique_labels(scheme.class_labels)
    scheme_name = generate_scheme_name(unique_labels, subclass_labels)
    save_name = 'Classifier_' + scheme_name
    output_path = os.path.join(outputdir, save_name + '.clf')

    with open(output_path, 'wb') as save_file:
        pickle.dump(scheme, save_file)


def load_scheme(filepath):
    """
    Load a classification scheme object previously saved using pickle
    :param filepath: full system path to file location to load
    :return: loaded scheme object
    :rtype: ClassificationScheme
    """
    with open(filepath, 'rb') as loadfile:
        scheme = pickle.load(loadfile)
    return scheme


class ClassificationScheme(object):
    """
    Container for final classification information, to be saved and used in future for unknown
    analyses, etc.
    """
    def __init__(self):
        self.lda = None     # type: LinearDiscriminantAnalysis
        self.classifier = None      # type: SVC
        self.classifier_type = None     # string - holds classifier type information for diagnostics
        self.name = None    # string - classifier name, usually assembled from input class names
        self.params = None  # return of self.classier.get_params()
        self.final_axis_cropvals = None     # final axes used in the scheme
        self.classif_mode = None    # standard or Gaussian mode, to ensure unknowns are fit in correct mode
        self.classif_prec_score = None  # sklearn.metrics.precision_score for the classification scheme
        self.explained_variance_ratio = None    # list of ratios of variance explained by each LD axis

        self.selected_features = []     # type: List[CFeature]
        self.all_features = None    # type: List[CFeature]

        self.numeric_labels = None  # list of class labels converted to ints
        self.class_labels = None    # list of original string class labels, with replicates
        self.unique_labels = None   # list of original string class labels, but only 1 per class
        self.transformed_test_data = None   # input data transformed by the final LDA
        # self.test_filenames = None  # list of filenames from input data
        self.input_feats = None     # list of CVs used in the scheme. NOT Feature objects, just CVs

        # cross validation information
        self.crossval_test_score = None     # best test accuracy score (score of the feature set chosen for final classification)
        self.all_crossval_data = None   # list of (train_score_means, train_score_stds, test_score_means, test_score_stds) for each feature combination

        self.num_gaussians = None       # stores the max number of Gaussian found in a voltage for assembling matrices for unknowns
        self.standard_means = None      # store for unknown standardization
        self.standard_stdevs = None     # store for unknown standardization

    def __str__(self):
        label_string = ','.join(self.unique_labels)
        return '<Classif_Scheme> type: {}, data: {}'.format(self.classifier_type, label_string)
    __repr__ = __str__

    def get_subclass_labels(self):
        """
        Return a list of (unique) subclass labels from all selected features
        :return: list of subclass labels (strings)
        """
        subclass_labels = []
        for feature in self.selected_features:
            if feature.subclass_label is not None:
                if feature.subclass_label not in subclass_labels:
                    subclass_labels.append(feature.subclass_label)
        return subclass_labels

    def get_subclass_mode(self):
        """
        Return a boolean signifying whether this scheme uses subclasses or not. Searches the scheme's
        all features list for any features with subclass information.
        :return: boolean: True if subclass mode, False if not
        :rtype: bool
        """
        subclass_labels = []
        for feature in self.all_features:
            if feature.subclass_label is not None:
                if feature.subclass_label not in subclass_labels:
                    subclass_labels.append(feature.subclass_label)

        if len(subclass_labels) > 1:
            return True
        else:
            return False

    def prep_unk_input(self, unk_cl_input, params_obj):
        """
        Assemble/prepare input from the unknown data file for classification. Sets the 'classif_input_raw'
        field in CIUAnalysisObj containers. In Gaussian mode, prepares Gaussian lists from features as required.
        :param unk_cl_input: ClInput container with unknown data
        :type unk_cl_input: ClInput
        :param params_obj: parameters container
        :type params_obj: Parameters
        :return: updated ClInput container
        :rtype: ClInput
        """
        for subclass_label, analysis_obj in unk_cl_input.subclass_dict.items():
            if not params_obj.classif_1_input_mode == 'All_Data':
                if params_obj.classif_1_input_mode == 'Gaussian_Feat':
                    # prepare gaussian features for classification (saves to container)
                    gaussians_by_cv = prep_gaussfeats_for_classif(analysis_obj.features_gaussian, analysis_obj)
                elif params_obj.classif_1_input_mode == 'Gaussian_Raw':
                    # use raw protein Gaussians without feature prep
                    gaussians_by_cv = analysis_obj.raw_protein_gaussians
                else:
                    gaussians_by_cv = []

                analysis_obj.classif_gaussians_by_cv = gaussians_by_cv
                input_classif_raw = prep_gaussian_input_raw(gaussians_by_cv, self.num_gaussians)
                analysis_obj.classif_input_raw = input_classif_raw
            else:
                # all data mode - initialize raw data for classification
                analysis_obj.classif_input_raw = analysis_obj.ciu_data
        return unk_cl_input

    def standardize_unk_input(self, unk_cl_input, params_obj):
        """
        Use the scheme's saved mean/stdev to standardize the unknown input the same way as
        the training data.
        :param unk_cl_input: ClInput containing the ciu_data from unknown to be fitted
        :type unk_cl_input: ClInput
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: updated unk_cl_input with standardized data saved to CIUAnalysisObj
        :rtype: ClInput
        """
        # first, need to match unknown data's CV axis to training data indices in case of reduced CV sampling
        reduced_means, reduced_stds = reduce_means_for_unk_standardize(unk_cl_input, self.standard_means, self.standard_stdevs)

        unk_cl_input = standardize_ciu_obj(unk_cl_input, reduced_means, reduced_stds, params_obj)
        return unk_cl_input

    def classify_unknown_clinput(self, unk_cl_input, params_obj):
        """
        Classify a test dataset according to this classification scheme. Selects features from
        the test dataset (ciudata), classifies, and returns output and score metrics.
        :param unk_cl_input: ClInput containing the ciu_data from unknown to be fitted
        :type unk_cl_input: ClInput
        :param params_obj: parameters information
        :type params_obj: Parameters
        :return: updated analysis object with prediction data saved
        :rtype: ClInput
        """
        # Prepare and standardize input data using scheme's saved mean/standard deviation
        unk_cl_input = self.prep_unk_input(unk_cl_input, params_obj)
        unk_cl_input = self.standardize_unk_input(unk_cl_input, params_obj)

        # Assemble feature data for fitting
        unk_subset = rearrange_ciu_by_feats_helper(unk_cl_input, params_obj, self.selected_features, class_numeric_label=0, num_gaussian_override=self.num_gaussians)
        unk_x_data, numeric_labels, string_labels = arrange_lda_new([unk_subset])

        # Fit/classify data according to scheme LDA and classifier
        unknown_transformed_lda = self.lda.transform(unk_x_data)
        pred_class_label = self.classifier.predict(unknown_transformed_lda)
        pred_probs_by_cv = self.classifier.predict_proba(unknown_transformed_lda)
        pred_probs_avg = np.average(pred_probs_by_cv, axis=0)

        # create plots and save information to object
        unk_cl_input.predicted_label = pred_class_label
        unk_cl_input.probs_by_cv = pred_probs_by_cv
        unk_cl_input.probs_avg = pred_probs_avg
        unk_cl_input.transformed_data = unknown_transformed_lda

        return unk_cl_input

    def plot_all_unknowns_clinput(self, unk_cl_inputs, param_obj, output_path):
        """
        Generate a decision regions plot of all unknown data in a provided list using this scheme,
        parameters, and output path. Input data MUST have transformed test data already set (classified)
        in all CIUAnalysisObj containers
        :param unk_cl_inputs: list of inputs already classified
        :type unk_cl_inputs: list[ClInput]
        :param param_obj: parameters container
        :type param_obj: Parameters
        :param output_path: directory in which to save output
        :return: void
        """
        all_plot_tups = [(x.transformed_data, x.name) for x in unk_cl_inputs]
        plot_classification_decision_regions(self, param_obj, output_path, unknown_tups=all_plot_tups)


class ClInput(object):
    """
    Container for all classification inputs. Essentially just a dictionary of {subclass label: analysis object}
    for each subclass with associated name and space to save output/classification predictions/probabilities
    Allows for all methods to handle subclass or non-subclass data. Represents a SINGLE replicate of a
    single class (and all subclasses if using subclass data)
    """

    def __init__(self, label, dict_subclass_objs):
        """
        Initialize new container. NOTE: if no subclasses, set entry key = '0'
        :param label: string used to label outputs from this combined unknown
        :param dict_subclass_objs: dict of key = subclass label, value = CIUAnalysisObj
        :type dict_subclass_objs: dict[str, CIUAnalysisObj]
        """
        self.class_label = label
        self.subclass_dict = dict_subclass_objs
        self.name = self.get_name()
        self.all_filenames = self.get_all_filenames()

        # output storage
        self.predicted_label = None
        self.probs_by_cv = None
        self.probs_avg = None
        self.transformed_data = None

    def get_subclass_obj(self, subclass_label=None):
        """
        Return the list of CIUAnalysisObj's corresponding to the provided subclass label string
        :param subclass_label: string to search for in the dict. If None, there should be only one entry - return it or warn
        :return: list of analysis objs for this subclass
        :rtype: CIUAnalysisObj
        """
        try:
            if subclass_label is None:
                # no label provided - this is for non-subclass mode, where the dict will have only 1 entry. Return it (and warn if >1 entry)
                if len(self.subclass_dict) > 1:
                    logger.warning('WARNING: single subclass requested but there are more than one. Erratic returns possible')
                for key, obj in self.subclass_dict.items():
                    return obj
            else:
                # return the requested object
                return self.subclass_dict[subclass_label]
        except KeyError:
            return None

    def get_name(self):
        """
        Generate a single (short) name from the input data and subclasses for reference (a replicate ID).
        :return: string name
        """
        first_key = sorted([x for x in self.subclass_dict.keys()])[0]
        if len(self.subclass_dict.keys()) > 1:
            name = self.subclass_dict[first_key].short_filename + '_{}SubCl'.format(len(self.subclass_dict.keys()))
        else:
            name = self.subclass_dict[first_key].short_filename
        return name

    def get_all_filenames(self):
        """
        Concatenate filenames for all subclasses into a single output for later reference
        :return: string
        """
        # all_filenames = ''
        # # iterate through subclasses in order of subclass label
        # for key, subclass_obj in sorted(self.subclass_dict.items(), key=lambda x: x[0]):
        #     all_filenames += '{};'.format(subclass_obj.short_filename)
        # all_filenames.rstrip(';')
        all_filenames = ';'.join([x[1].short_filename for x in sorted(self.subclass_dict.items(), key=lambda x: x[0])])
        return all_filenames

    def __str__(self):
        """
        string rep
        :return: string
        """
        return '<ClInput> {}'.format(self.class_label)


class DataSubset(object):
    """
    Container for a subset of CIU data selected from selected features. Similar to a ClInput, but
    does NOT have subclasses because only the data from best features has been selected for the
    classifying scheme.
    """
    def __init__(self, data, class_label, numeric_label, file_id, all_filenames, features):
        """
        Initialize new subset container
        :param data: dataset arranged using rearrange CIU by feats method (2D numpy array)
        :param class_label: class label
        :param numeric_label: (int) numeric label corresponding to the class label of this subset
        :param file_id: (string) file identifier (CIUAnalysisObj.short_filename). In subclass mode, this should be a combination of filenames for each subclass obj.
        :param all_filenames: (string) combined filenames for all CIU files that comprise the subset
        :param features: list of features used in arranging data
        :type features: list[CFeature]
        """
        self.data = data
        self.class_label = class_label
        self.numeric_label = numeric_label
        self.features = features
        self.file_id = file_id
        self.all_filenames = all_filenames


class ClassifInput(object):
    """
    Container for classification input information (labels and CIUAnalysisObj containers)
    to keep track of labels/replicates and subclasses.
    """
    def __init__(self, class_labels, obj_list_by_label, subclass_label=None):
        """
        Initialize a new container with provided information, including optional subclass label
        :param class_labels: list of strings corresponding to labels for each class
        :param obj_list_by_label: list of lists of analysis containers corresponding to all replicates of each class (sorted by class)
        :param subclass_label: (optional) if this is a subclass analysis, the label corresponding to this subclass
        """
        # prepare labels for all replicates in each class
        self.class_labels = class_labels
        self.shaped_label_list = []
        self.shaped_replicates = []
        for index, label in enumerate(class_labels):
            self.shaped_label_list.append([label for _ in range(len(obj_list_by_label[index]))])
            self.shaped_replicates.append([x for x in range(len(obj_list_by_label[index]))])
        self.analysis_objs_by_label = obj_list_by_label
        self.subclass_label = subclass_label

    def get_flat_list(self, get_type):
        """
        Get a flattened list of labels or analysis objects
        :param get_type: whether to get labels ('label') or analysis objects ('obj')
        :return: flattened list
        :rtype: list[CIUAnalysisObj]    #  not always, but this resolves some erroneous type checking in PyCharm
        """
        if get_type == 'label':
            flat_label_list = [x for label_list in self.shaped_label_list for x in label_list]
            return flat_label_list
        elif get_type == 'rep':
            flat_reps = [x for rep_list in self.shaped_replicates for x in rep_list]
            return flat_reps
        elif get_type == 'obj':
            flat_obj_list = [x for analysis_obj_list in self.analysis_objs_by_label for x in analysis_obj_list]
            return flat_obj_list
        else:
            logger.warning('Invalid type request. Valid values are "label" or "obj"')

    def __str__(self):
        """
        Return string representation - subclass label
        :return: string
        """
        if self.subclass_label is not None:
            return '<ClassInput> {}'.format(self.subclass_label)
        else:
            return '<ClassInput>'
    __repr__ = __str__


class CFeature(object):
    """
    Container for classification feature information in feature selection.
    """
    def __init__(self, cv, cv_index, mean_score, std_dev_score, subclass_label=None):
        self.cv = cv
        self.cv_index = cv_index
        # self.score_list = score_list
        self.mean_score = mean_score
        self.std_dev_score = std_dev_score
        self.subclass_label = subclass_label

    def __str__(self):
        if self.subclass_label is not None:
            return '<CFeature> cv: {}, sub: {}, score: {:.2f}'.format(self.cv, self.subclass_label, self.mean_score)
        else:
            return '<CFeature> cv: {}, score: {:.2f}'.format(self.cv, self.mean_score)
    __repr__ = __str__


class CrossValRun(object):
    """
    Method to generate combinations of training and test datasets and run LDA + classification to generate training
    score and test score on selected features
    """
    def __init__(self, data_list_by_label, label_list, training_size, features):
        """
        Initialize a CrossValProduct method with data, label, training size, and features
        :param data_list_by_label: formatted data (sorted by feats, subclasses handled, etc) list for all classes, sorted by class label
        :type data_list_by_label: list[list[DataSubset]]
        :param label_list: label list for data
        :param training_size: num of ciu data in each class to be used as training set
        :param features: cv features
        :type features: list[CFeature]
        """
        self.data_list_by_label = data_list_by_label
        self.label_list = label_list
        self.training_size = training_size
        self.features = features

        # assemble training and test data and label
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None

        # results info
        self.train_scores = []
        self.train_scores_mean = None
        self.train_scores_std = None
        self.test_scores = []
        self.test_scores_mean = None
        self.test_scores_std = None
        self.probs = []
        self.probs_mean = None
        self.probs_std = None

        # for roc metrics
        self.tmp_fpr = []
        self.tpr_class = []
        self.tpr_class_mean = None
        self.tpr_class_std = None
        self.roc_auc_class = []
        self.roc_auc_class_mean = None
        self.roc_auc_class_std = None
        self.tpr_micro = []
        self.tpr_micro_mean = None
        self.tpr_micro_std = None
        self.roc_auc_micro = []
        self.roc_auc_micro_mean = None
        self.roc_auc_micro_std = None
        self.tpr_macro = []
        self.tpr_macro_mean = None
        self.tpr_macro_std = None
        self.roc_auc_macro = []
        self.roc_auc_macro_mean = None
        self.roc_auc_macro_std = None

        self.results = {}

        self.all_classes_combo_lists = []   # assembled data (combinations within one class)
        self.all_products = []      # assembled data (products of all combinations across all classes)

    def random_sample_run_lda(self, num_iterations):
        """
        Alternative method to compute crossval scores for large datasets for which product/combination lists
        are too large to fit in memory. Randomly samples individual combinations of training/test data from each
        class, computes their product, runs lda, and stores the score in self.
        :param num_iterations: How many cross validation runs to perform (int)
        :return: void (saves scores to self)
        """
        roc_outputs = []

        # Randomly crossvalidate num_iterations times
        iteration_times = []
        current_time = time.time()
        for _ in range(num_iterations):
            allclass_traintest_combos = []
            # For each class, randomly select training and test datasets and generate a data combination
            for class_index, class_subset_list in enumerate(self.data_list_by_label):
                num_replicates = len(class_subset_list)
                train_indices = random.sample(range(num_replicates), self.training_size)
                test_indices = [x for x in range(num_replicates) if x not in train_indices]

                training_subset_list = [class_subset_list[i] for i in train_indices]
                test_subset_list = [class_subset_list[i] for i in test_indices]

                # assemble LDA data here for train_subsets, train_labels (and test_subsets, test_labels)
                train_lda_data, train_lda_labels, train_string_labels = arrange_lda_new(training_subset_list)
                test_lda_data, test_lda_labels, test_string_labels = arrange_lda_new(test_subset_list)

                # initialize container to hold this combination of datasets for eventual product against other class(es)
                data_combo = CrossValData(train_lda_data, train_lda_labels, train_string_labels, test_lda_data, test_lda_labels, test_string_labels)
                allclass_traintest_combos.append(data_combo)

            # take the product of the generated random data combinations and save scores
            for class_combo_pair in itertools.combinations(allclass_traintest_combos, len(allclass_traintest_combos)):
                final_train_data, final_train_labels = [], []
                final_test_data, final_test_labels = [], []

                # assemble training and test data into a single matrix (each) with a corresponding label matrix for each
                for crossval_data in class_combo_pair:
                    # crossval_data contains training and test data, each of which may have multiple columns. Combined into single matrix each
                    for index in range(len(crossval_data.train_data)):
                        final_train_data.append(crossval_data.train_data[index])
                        final_train_labels.append(crossval_data.train_labels_num[index])
                    for index in range(len(crossval_data.test_data)):
                        final_test_data.append(crossval_data.test_data[index])
                        final_test_labels.append(crossval_data.test_labels_num[index])

                # Run LDA/SVC crossvalidation using the assembled data
                svc, lda = run_lda_svc(final_train_data, final_train_labels)

                # Generate an accuracy score for the training and test data against the classifier and save
                train_lda = lda.transform(final_train_data)
                test_lda = lda.transform(final_test_data)

                this_roc_output = roc_curve_area_multiclass(train_lda, final_train_labels, test_lda, final_test_labels, svc)
                roc_outputs.append(this_roc_output)

                train_score = svc.score(train_lda, final_train_labels)
                test_score = svc.score(test_lda, final_test_labels)
                self.train_scores.append(train_score)
                self.test_scores.append(test_score)
            iteration_times.append(time.time() - current_time)
            current_time = time.time()

        # Save final scores and means to container
        self.save_results(roc_outputs)
        logger.debug('Avg crossval iteration: {:.3f}s, total: {:.2f}s, {} iterations'.format(np.mean(iteration_times), np.sum(iteration_times), len(iteration_times)))

    def divide_data_and_run_lda(self):
        """
        Divide the data into each possible training and test arrangement and run LDA/SVC classification
        to assess accuracy
        :return: void (set results to self)
        """
        shaped_label_list = []
        for index, label in enumerate(self.label_list):
            shaped_label_list.append([label for _ in range(len(self.data_list_by_label[index]))])

        # generate all combinations of each class input datalist
        for class_index, class_subset_list in enumerate(self.data_list_by_label):
            class_traintest_combo_list = []
            # create all combinations of training and test data
            for training_subset_list in itertools.combinations(class_subset_list, self.training_size):
                # test data is anything not in the training list
                test_subset_list = [x for x in class_subset_list if x not in training_subset_list]

                # assemble LDA data here for train_subsets, train_labels (and test_subsets, test_labels)
                train_lda_data, train_lda_labels, train_string_labels = arrange_lda_new(training_subset_list)
                test_lda_data, test_lda_labels, test_string_labels = arrange_lda_new(test_subset_list)

                # initialize container to hold this combination of datasets for eventual product against other class(es)
                data_combo = CrossValData(train_lda_data, train_lda_labels, train_string_labels, test_lda_data, test_lda_labels, test_string_labels)
                class_traintest_combo_list.append(data_combo)
            self.all_classes_combo_lists.append(class_traintest_combo_list)

    def assemble_class_products(self, max_crossval_iterations):
        """
        Take divided train/test data combinations from self.divide_data and generate all products between
        classes. (e.g. for 2 classes, 3 replicates each, divide data assigns reps 1, 2 to train and 3 to test for
        one class (and all other combinations), and this method takes class 1, train 1 2 * class 2 train 1 2 to generate
        all combinations of training and test subsets for all classes).
        :param max_crossval_iterations: Maximum number of class products to consider for crossval. Intended to prevent
        running many thousands (or more) cross validations per feature when a smaller sample size would suffice.
        :return: void (saves to self.all_products)
        """
        iteration_times = []
        current_time = time.time()
        # Bootstrap (if required) by randomly sampling from all possible products of the class training/test datasets
        if max_crossval_iterations == 0:
            selected_products = itertools.product(*self.all_classes_combo_lists)
        else:
            try:
                selected_products = random.sample(list(itertools.product(*self.all_classes_combo_lists)), max_crossval_iterations)
            except ValueError:
                # The total product population is less than the max number of iterations. Use the entire list
                selected_products = itertools.product(*self.all_classes_combo_lists)

        # arrays to store roc metrics output
        roc_outputs = []

        # for class_combo_pair in itertools.product(*self.all_classes_combo_lists):
        for class_combo_pair in selected_products:
            final_train_data, final_train_labels = [], []
            final_test_data, final_test_labels = [], []

            # assemble training and test data into a single matrix (each) with a corresponding label matrix for each
            for crossval_data in class_combo_pair:
                # crossval_data contains training and test data, each of which may have multiple columns. Combined into single matrix each
                for index in range(len(crossval_data.train_data)):
                    final_train_data.append(crossval_data.train_data[index])
                    final_train_labels.append(crossval_data.train_labels_num[index])
                for index in range(len(crossval_data.test_data)):
                    final_test_data.append(crossval_data.test_data[index])
                    final_test_labels.append(crossval_data.test_labels_num[index])

            # Run LDA/SVC crossvalidation using the assembled data
            svc, lda = run_lda_svc(final_train_data, final_train_labels)

            # Generate an accuracy score for the training and test data against the classifier and save
            train_lda = lda.transform(final_train_data)
            test_lda = lda.transform(final_test_data)

            roc_curve_multiclass_out = roc_curve_area_multiclass(train_lda, final_train_labels, test_lda, final_test_labels, svc)
            roc_outputs.append(roc_curve_multiclass_out)

            train_score = svc.score(train_lda, final_train_labels)
            test_score = svc.score(test_lda, final_test_labels)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)
            iteration_times.append(time.time() - current_time)
            current_time = time.time()

        # compute the mean and stds for roc metrics and crossval accuracy
        self.save_results(roc_outputs)
        logger.debug('Avg crossval iteration: {:.3f}s, total: {:.2f}s, {} iterations'.format(np.mean(iteration_times), np.sum(iteration_times), len(iteration_times)))

    def save_results(self, roc_outputs):
        """
        Save all stats results (confusion matrix, roc, and crossval accuracies) to a dictionary
        for easy passing to various plotting methods in other methods.
        ONLY intended for AFTER crossval has been completed.
        :param roc_outputs: list of roc output dicts (each from roc_curve_area_multiclass)
        :return: dictionary: key = self.<various attributes>, value = value
        """
        self.results['tmp_fpr'] = roc_outputs[0]['tmp_fpr']

        self.results['tpr_class_mean'] = np.mean([x['tpr_class'] for x in roc_outputs], axis=0)
        self.results['tpr_class_std'] = np.std([x['tpr_class'] for x in roc_outputs], axis=0)
        self.results['roc_auc_class_mean'] = np.mean([x['roc_auc_class'] for x in roc_outputs], axis=0)
        self.results['roc_auc_class_std'] = np.std([x['roc_auc_class'] for x in roc_outputs], axis=0)

        self.results['tpr_micro_mean'] = np.mean([x['tpr_micro'] for x in roc_outputs], axis=0)
        self.results['tpr_micro_std'] = np.std([x['tpr_micro'] for x in roc_outputs], axis=0)
        self.results['roc_auc_micro_mean'] = np.mean([x['roc_auc_micro'] for x in roc_outputs])
        self.results['roc_auc_micro_std'] = np.std([x['roc_auc_micro'] for x in roc_outputs])

        self.results['tpr_macro_mean'] = np.mean([x['tpr_macro'] for x in roc_outputs], axis=0)
        self.results['tpr_macro_std'] = np.std([x['tpr_macro'] for x in roc_outputs], axis=0)
        self.results['roc_auc_macro_mean'] = np.mean([x['roc_auc_macro'] for x in roc_outputs])
        self.results['roc_auc_macro_std'] = np.std([x['roc_auc_macro'] for x in roc_outputs])

        # Save final scores and means to container
        self.results['train_scores_mean'] = np.mean(self.train_scores)
        self.results['train_scores_std'] = np.std(self.train_scores)
        self.results['test_scores_mean'] = np.mean(self.test_scores)
        self.results['test_scores_std'] = np.std(self.test_scores)
        self.results['probs_mean'] = np.mean(self.probs)
        self.results['probs_std'] = np.std(self.probs)


class CrossValData(object):
    """
    Container object for holding training and test data for a single way of combining the datasets in a class.
    Training and test datasets can be of arbitrary size. Holds corresponding label arrays of same shape as
    train/test data.
    """
    def __init__(self, train_data, train_num_labels, train_str_labels, test_data, test_num_labels, test_str_labels):
        """
        Initialize new container
        :param train_data: list containing ciu datasets for training
        :param train_num_labels: numeric labels (ints) list, same length as train data
        :param train_str_labels: same length list as train_data with labels (strings)
        :param test_data: list containing ciu datasets for testing
        :param test_num_labels: numeric labels (ints) list, same length as test data
        :param test_str_labels: same length list as test_data with labels (strings)
        """
        self.train_data = train_data
        self.train_labels_num = train_num_labels
        self.train_labels_str = train_str_labels
        self.test_data = test_data
        self.test_labels_num = test_num_labels
        self.test_labels_str = test_str_labels


class UFSResult(object):
    """
    Container for label and product information for data combinations used in feature selection.
    """
    def __init__(self, data_list, label_list):
        """
        Initialize a new UFSResult with associated label and data
        :param data_list: list of ciu_data matrices (normalized)
        :param label_list: list of class labels (same length as data_list)
        """
        self.data_input = data_list
        self.labels_input = label_list

        self.combined_data = None
        self.combined_label_arr = None
        self.numeric_label_arr = None

        # results information
        self.fit_scores = None
        self.fit_pvalues = None
        self.fit_sc = None

        self.train_scores = None
        self.test_scores = None

        # run data preparation
        self.prepare_data()

    def prepare_data(self):
        """
        Manipulate data into concatenated arrays suitable for input into feature selection algorithm.
        :return: void
        """
        self.combined_data = np.concatenate(self.data_input)

        label_array_list = []
        index = 0
        for label in self.labels_input:
            label_array = [label for _ in range(len(self.data_input[index]))]
            label_array_list.append(label_array)
            index += 1
        self.combined_label_arr = np.concatenate(label_array_list)

        # use the label encoder to generate a numeric label list (could just do this manually??)
        # output is just the class numbers in an array the shape of the input
        encoder = LabelEncoder()
        label_list = encoder.fit(self.combined_label_arr)
        self.numeric_label_arr = label_list.transform(self.combined_label_arr) + 1

    def __str__(self):
        label_string = ','.join(self.labels_input)
        return '<UFSResult> labels: {}'.format(label_string)
    __repr__ = __str__


class ManualFeatureUI(tkinter.Toplevel):
    """
    Popup for manual feature selection. Displays a list of features with a checkbox for each
    to include in scheme. Supports subclasses.
    """
    def __init__(self, ordered_features):
        """
        Display a list of features (in order) with checkboxes to include
        :param ordered_features: list of features in descending order of score from UFS
        :type ordered_features: list[CFeature]
        """
        tkinter.Toplevel.__init__(self)
        self.title('Choose which features to include in classification: ("sc" = score)')

        # output features
        self.return_features = []
        self.feat_var_tups = []

        # initialize graphical elements
        main_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        main_frame.grid(column=0, row=0)
        num_rows = 25
        row_index = -1
        col_index = 0
        for index, feature in enumerate(ordered_features):
            row_index += 1
            if row_index > num_rows:
                row_index = 0
                col_index += 1

            # create a checkbutton for this feature
            if feature.subclass_label is not None:
                feat_text = '{}) {}V, {}, sc: {:.2f}'.format(index + 1, feature.cv, feature.subclass_label, feature.mean_score)
            else:
                feat_text = '{}) {}V, sc: {:.2f}'.format(index + 1, feature.cv, feature.mean_score)
            feat_var = tkinter.IntVar()
            ttk.Checkbutton(main_frame, text=feat_text, variable=feat_var).grid(row=row_index, column=col_index)
            self.feat_var_tups.append((feature, feat_var))

        # Finally, add 'okay' and 'cancel' buttons to the bottom of the dialog
        button_frame = ttk.Frame(self, padding='5 5 5 5')
        button_frame.grid(column=0, row=1)
        ttk.Button(button_frame, text='Cancel', command=self.cancel_button_click).grid(row=0, column=0, sticky='w')
        ttk.Button(button_frame, text='OK', command=self.ok_button_click).grid(row=0, column=1, sticky='e')

    def cancel_button_click(self):
        """
        Cancel classification
        :return: void
        """
        self.return_features = []
        self.on_close_window()

    def ok_button_click(self):
        """
        Figure out which checkboxes are checked and set self's list of features to those selected.
        :return: selected features list
        :rtype: list[CFeature]
        """
        for feature, feat_var in self.feat_var_tups:
            if feat_var.get() == 1:
                # select this feature
                self.return_features.append(feature)
        self.on_close_window()

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()


def get_manual_classif_feats(features_list):
    """
    Run classification manual feature selection UI to get selected features for classification.
    :param features_list: list of features
    :type features_list: list[CFeature]
    :return: list[CFeature]
    """
    feat_ui = ManualFeatureUI(features_list)
    feat_ui.lift()
    feat_ui.grab_set()  # prevent users from hitting multiple windows simultaneously
    feat_ui.wait_window()
    feat_ui.grab_release()

    return feat_ui.return_features


# if __name__ == '__main__':
    # Read the data
    # import tkinter
    # from tkinter import filedialog
    # from tkinter import simpledialog
    # import CIU_Params
    # import Raw_Processing

    #
    # root = tkinter.Tk()
    # root.withdraw()

    # num_classes = simpledialog.askinteger('Class Number', 'Into how many classes do you want to group?')
    # num_classes = 4
    # data_labels = []
    # obj_list_by_label = []
    # # main_dir = 'C:\\'
    # main_dir = r"C:\Users\dpolasky\Desktop\CIUSuite2\CIU2 test data\Classify_Sarah\apo_cdl_pi_3way"

    # class_labels = ['Igg1', 'Igg2', 'Igg4', 'Igg4']
    # f1 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_1.ciu'
    # f2 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_2.ciu'
    # f3 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG1_3.ciu'
    # f4 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_1.ciu'
    # f5 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_2.ciu'
    # f6 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG2_3.ciu'
    # f7 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_1.ciu'
    # f8 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_2.ciu'
    # f9 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG3_3.ciu'
    # f10 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_1.ciu'
    # f11 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_2.ciu'
    # f12 = r'C:\Users\dpolasky\Desktop\CIU2 test data\ldaanalysisscripts\IgG4_3.ciu'

    # class_labels = ['cdl', 'pi', 'apo']
    # class1_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[0] in x.lower()]
    # class2_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and class_labels[1] in x.lower()]
    # class3_files = [os.path.join(main_dir, x) for x in os.listdir(main_dir) if x.endswith('.ciu') and os.path.join(main_dir, x) not in class1_files and os.path.join(main_dir, x) not in class2_files]
    # fs = [class1_files, class2_files, class3_files]
    # f_class1 = [f1, f2, f3]
    # f_class2 = [f4, f5, f6]
    # f_class3 = [f7, f8, f9]
    # f_class4 = [f10, f11, f12]
    # fs = [f_class1, f_class2, f_class4, f_class4]
    # fs= [f_class1, f_class2]

    # sarahfile = r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\Iggs_datalist.csv"
    # files = np.genfromtxt(sarahfile, skip_header=1, delimiter=',', dtype='str')
    # class_sarahfiles = np.unique(files[:, 0])
    # # CDL, PA, PC, PE, PG, PI, PPIX, PS [in order]
    # filelist = [[] for i in range(len(class_sarahfiles))]
    # for index, classtype in enumerate(files[:, 0]):
    #     for class_ind, class_unique in enumerate(class_sarahfiles):
    #         if class_unique == classtype:
    #             filelist[class_ind].append((files[:, 1][index]))
    #
    # class_labels = [class_sarahfiles[0], class_sarahfiles[1], class_sarahfiles[3], class_sarahfiles[3]]
    # fs = [filelist[0], filelist[1], filelist[3], filelist[3]]
    #
    # for class_index in range(0, num_classes):
    #     # Read in the .CIU files and labels for each class
    #     # label = simpledialog.askstring('Class Name', 'What is the name of this class?')
    #     class_label = class_labels[class_index]
    #     # files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    #     files = fs[class_index]
    #     main_dir = os.path.dirname(files[0])
    #
    #     obj_list = []
    #     for file in files:
    #         with open(file, 'rb') as analysis_file:
    #             obj = pickle.load(analysis_file)
    #         obj_list.append(obj)
    #     data_labels.append(class_label)
    #     obj_list_by_label.append(obj_list)
    #
    # unkdata_name = open(r"C:\Users\sugyan\Documents\CIUSuite\Classification\IgGdata\IgG1_3.ciu", 'rb')
    # unkdata_ = pickle.load(unkdata_name)
    #
    # params = CIU_Params.Parameters()
    # params.set_params(CIU_Params.parse_params_file(CIU_Params.hard_descripts_file))
    # obj_list_by_label, equalized_axes_list = Raw_Processing.equalize_axes_2d_list(obj_list_by_label)
    #
    # scheme = main_build_classification(data_labels, obj_list_by_label, params, main_dir)
    # scheme.classify_unknown(unkdata_, params, main_dir, unk_label='Unknown')

    # featurescaling_lda(data_labels, obj_list_by_label, main_dir)
    # class_comparison_lda(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_withprediction(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_lda_withprediction(data_labels, obj_list_by_label, main_dir)

    # univariate_feature_selection(data_labels, obj_list_by_label)
    # output_scheme = main_build_classification(data_labels, obj_list_by_label, main_dir)
    # univariate_feature_selection_datacv_runldaonfeats_suggie(data_labels, obj_list_by_label, main_dir)

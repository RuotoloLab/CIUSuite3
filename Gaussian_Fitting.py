"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky and Sugyan Dixit

Gaussian fitting module for CIUSuite 2
author: DP
date: 10/10/2017
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import scipy.integrate
import scipy.interpolate
import os
import math
import peakutils
import pickle
import logging
import tkinter
from tkinter import filedialog
import scipy.signal
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.patches
import lmfit
import time
import multiprocessing
from tkinter import messagebox

import CIU_raw
import Raw_Processing
import CIU_analysis_obj
import CIU_Params

# imports for type checking
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from CIU_analysis_obj import CIUAnalysisObj

protein_prefix = 'p'
nonprotein_prefix = 'np'
baseline_prefix = 'b'
logger = logging.getLogger('main')


class Gaussian(object):
    """
    Container for fitted gaussian information. Holds fit parameters and any associated metadata
    """
    def __init__(self, amplitude, centroid, width, collision_voltage, pcov, protein_bool):
        """
        Initialize a new Gaussian container
        :param amplitude: peak amplitude
        :param centroid: peak centroid
        :param width: peak width
        :param collision_voltage: Activation axis value at which this Gaussian was generated
        :param pcov: (optional, set to None if not needed) covariance matrix from curve fitting
        :param protein_bool: True if this is a protein (signal) peak, False if a non-protein (noise) peak
        """
        self.amplitude = amplitude
        self.centroid = centroid
        self.width = width
        self.cv = collision_voltage
        self.fwhm = 2*(math.sqrt(2*math.log(2)))*self.width
        self.resolution = self.centroid/(self.fwhm + 1e-10)
        self.fit_covariances = pcov
        self.is_protein = protein_bool
        if pcov is not None:
            self.fit_errors = np.sqrt(np.diag(pcov))

    def __str__(self):
        return 'Gaussian: x0={:.2f} A={:.1f} w={:.1f} cv={}'.format(self.centroid,
                                                                    self.amplitude,
                                                                    abs(self.width),
                                                                    self.cv)
    # set repr = str for printing in lists
    __repr__ = __str__

    def print_info(self):
        """
        Method for generating strings to save to output files with all info
        :return: string
        """
        return '{},{:.2f},{:.2f},{:.2f}'.format(self.cv, self.amplitude, self.centroid, self.fwhm)

    def print_single_param(self, param):
        """
        Format a single Gaussian parameter for printing
        :param param: 'amplitude', 'centroid', or 'width'
        :return: formatted string
        """
        try:
            value = self.__getattribute__(param)
        except AttributeError:
            logger.error('No Gaussian attribute {}'.format(param))
            value = 0

        if param == 'width':
            value = sigma_to_fwhm(value)
        return '{:.2f}'.format(value)

    def return_popt(self):
        """
        Re-generate Gaussian function parameter list (e.g. popt style from curve_fit) from
        gaussian object
        :return: [baseline, amplitude, centroid, width]
        """
        return [self.amplitude, self.centroid, self.width]


class SingleFitStats(object):
    """
    Container for holding fit information for a single multi-Gaussian fitting (one collision voltage).
    Includes r2, fit data, error estimates, etc (all output of linregress)
    Intended to use called when initializing a fit.
    *updated to include output from LMFit and original (curve_fit) in same container. Must have
    one of popt OR lmfit_output, and will generate Gaussians and r2 from both for output
    """
    def __init__(self, x_data, y_data, cv, amp_cutoff, lmfit_output=None, popt=None):
        """
        Initialize a new fit between the provided x/y data and optimized Gaussian parameters
        :param x_data: x (DT) raw data being fit by popt
        :param y_data: y (intensity) raw data being fit by popt
        :param popt: optimized parameters returned from curve_fit
        :param amp_cutoff: minimum amplitude for peak to be allowed
        :param cv: collision voltage at which this fit occurred
        :param lmfit_output: output container from LMFit (from model.fit(...))
        :type lmfit_output:
        """
        self.x_data = x_data
        self.y_data = y_data
        self.cv = cv
        self.baseline_val = 0

        if lmfit_output is not None:
            protein_popt, nonprotein_popt = get_popt_from_lmoutput(lmfit_output, amp_cutoff)
            popt = [x for x in protein_popt]
            popt.extend(nonprotein_popt)

            # Check for baseline as well, and add it to the fit if provided
            keys = sorted(lmfit_output.best_values.keys())
            baseline_key = [lmfit_output.best_values[key] for key in keys if key.startswith(baseline_prefix)]
            if len(baseline_key) > 0:
                self.baseline_val = baseline_key[0]

        else:
            protein_popt = popt
            nonprotein_popt = []

        self.y_fit = multi_gauss_func(x_data, *popt) + self.baseline_val
        self.slope, self.intercept, self.rvalue, self.pvalue, self.stderr = linregress(self.y_data, self.y_fit)
        self.adjrsq = adjrsquared(self.rvalue ** 2, len(y_data))

        # Gaussian lists specific to protein and non-protein components fitted by LMFit
        self.gaussians_protein = generate_gaussians_from_popt(protein_popt, protein_bool=True, cv=cv, pcov=None)
        self.gaussians_nonprotein = generate_gaussians_from_popt(nonprotein_popt, protein_bool=False, cv=cv, pcov=None)

        self.gaussians = [x for x in self.gaussians_protein]
        self.gaussians.extend(self.gaussians_nonprotein)

        # additional information that may be present
        self.p0 = None      # initial guess array used to generate this popt
        self.pcov = None    # output covariance matrix

        self.score = None   # score from second round fitting (r2 - penalties)
        self.peak_penalties = None      # list of penalties for each peak in the Gaussian list

    def __str__(self):
        """
        string rep
        :return: string
        """
        return '<Fit> r2: {:.3f}, sc: {:.3f}, gauss: {}, cv: {}'.format(self.adjrsq, self.score, len(self.gaussians), self.cv)
    __repr__ = __str__

    def compute_fit_score(self, params_obj, penalty_scaling):
        """
        Uses a penalty function to attempt to regularize the fitting and score peak fits optimally.
        Penalty function is designed to penalize:
            - peaks whose widths deviate from expected protein peak width
            - peaks that are too close together (sharing too much area)
        :param params_obj: parameter container
        :type params_obj: Parameters
        :param penalty_scaling: how much to scale penalty (to reduce contribution relative to rsq)
        :return: score (float between 0, 1), penalties by individual peaks
        """
        # compute penalties by peak to allow removal of poorly fit peaks
        peak_penalties = []
        for gaussian in self.gaussians_protein:
            current_penalty = compute_width_penalty(gaussian.width,
                                                    expected_width=fwhm_to_sigma(params_obj.gaussian_72_prot_peak_width),
                                                    tolerance=fwhm_to_sigma(params_obj.gaussian_73_prot_width_tol),
                                                    steepness=1)
            if len(self.gaussians_protein) > 1:
                if gaussian.amplitude > params_obj.gaussian_2_int_threshold:
                    current_penalty += compute_area_penalty(gaussian, self.gaussians_protein, self.x_data, params_obj.gaussian_74_shared_area_mode)
            peak_penalties.append(current_penalty)

        # add up penalties and subtract from the fit adjusted r2 to obtain final score
        total_penalty = np.sum(peak_penalties)

        # add penalty for low amplitude protein peak - max protein peak shouldn't be too low
        if len(self.gaussians_nonprotein) > 0:
            if len(self.gaussians_protein) > 0:
                max_protein_amp = max([x.amplitude for x in self.gaussians_protein])
            else:
                max_protein_amp = 0
            if max_protein_amp < params_obj.gaussian_9_nonprot_min_prot_amp:
                total_penalty += (params_obj.gaussian_9_nonprot_min_prot_amp - max_protein_amp)

        scaled_penalty = total_penalty * penalty_scaling
        score = self.adjrsq - scaled_penalty

        # save information to the fit container
        self.score = score
        self.peak_penalties = peak_penalties

    def get_popt(self):
        """
        Return a single parameters list for all Gaussians from this fit in curve_fit compatible
        format
        :return: list of optimized params (popt)
        """
        popt = []
        for gaussian in self.gaussians:
            popt.extend(gaussian.return_popt())
        return popt

    def plot_fit(self):
        """
        plotting method for diagnostics
        :return: void
        """
        plt.clf()
        plt.scatter(self.x_data, self.y_data)
        plt.plot(self.x_data, self.y_fit, ls='--', color='black')
        for gaussian in self.gaussians:
            plt.plot(self.x_data, gaussfunc(self.x_data, *gaussian.return_popt()))
        plt.show()
        plt.close()


def main_gaussian_lmfit_wrapper(analysis_obj_list, params_obj, outputpath):
    """
    Wrapper method for main_gaussian_lmfit that uses multiprocessing.
    :param analysis_obj_list: list of CIU containers to fit Gaussians to
    :type analysis_obj_list: list[CIUAnalysisObj]
    :param params_obj: parameter information container
    :type params_obj: Parameters
    :param outputpath: directory in which to save output
    :return: updated analysis object list
    :rtype: list[CIUAnalysisObj]
    """
    output_objs = []
    all_csv_output = ''
    all_file_gaussians = []

    # don't use more cores than files selected
    if len(analysis_obj_list) < params_obj.gaussian_61_num_cores:
        num_cores = len(analysis_obj_list)
    else:
        num_cores = params_obj.gaussian_61_num_cores

    if num_cores > 1:
        pool = multiprocessing.Pool(processes=num_cores)
        results = []
        for analysis_obj in analysis_obj_list:
            # Run fitting and scoring across the provided range of peak options with multiprocessing
            logger.info('Gaussian fitting process for file {} added to queue'.format(analysis_obj.short_filename))
            new_params_obj = CIU_Params.Parameters()
            new_params_obj.set_params(params_obj.params_dict)    # copy the params object to prevent simultaneous access
            argslist = [analysis_obj, new_params_obj, outputpath]
            pool_result = pool.apply_async(main_gaussian_lmfit, args=argslist)
            results.append(pool_result)

        pool.close()  # tell the pool we don't need it to process any more data

        for pool_result_container in results:
            # save results
            pool_result = pool_result_container.get()
            analysis_obj, csv_output, cv_gaussians, fit_time = pool_result[0], pool_result[1], pool_result[2], pool_result[3]
            all_csv_output += csv_output
            all_file_gaussians.append(cv_gaussians)
            output_objs.append(analysis_obj)
            logger.info('Fitting for file {} done in {:.2f} s'.format(analysis_obj.short_filename, fit_time))

        pool.join()     # terminate pool processes once finished

    else:
        # User specified one thread, so don't use multiprocessing at all
        for analysis_obj in analysis_obj_list:
            logger.info('Started Gaussian fitting for file {}...'.format(analysis_obj.short_filename))
            analysis_obj, csv_output, cv_gaussians, fit_time = main_gaussian_lmfit(analysis_obj, params_obj, outputpath)
            all_csv_output += csv_output
            all_file_gaussians.append(cv_gaussians)
            output_objs.append(analysis_obj)
            logger.info('Fitting for file {} done in {:.2f} s'.format(analysis_obj.short_filename, fit_time))

    return output_objs, all_csv_output, all_file_gaussians


def main_gaussian_lmfit(analysis_obj, params_obj, outputpath):
    """
    Alternative Gaussian fitting method using LMFit for composite modeling of peaks. Estimates initial peak
    parameters using helper methods, then fits optimized Gaussian distributions and saves results. Intended
    for direct call from buttons in GUI.
    :param analysis_obj: analysis container
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: parameter information container
    :type params_obj: Parameters
    :param outputpath: directory in which to save output
    :return: updated analysis object
    :rtype: CIUAnalysisObj
    """
    start_time = time.time()

    cv_col_data = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    outputfolder = os.path.join(outputpath, analysis_obj.short_filename)
    if params_obj.gaussian_4_save_diagnostics:
        if not os.path.isdir(outputfolder):
            os.makedirs(outputfolder)

    best_fits_by_cv = []
    scores_by_cv = []

    # User specified one thread, so don't use multiprocessing at all
    results = []
    for cv_index, cv_col_intensities in enumerate(cv_col_data):
        cv = analysis_obj.axes[1][cv_index]
        if cv_index > 0:
            best_prev_fit = max(results[cv_index - 1], key=lambda x: x.score)
            gaussian_guess_list = copy_gaussians_from_prevfit(best_prev_fit, cv)
        else:
            # run initial guess method since we have no previous peaks to refer to
            gaussian_guess_list = guess_gauss_init(cv_col_intensities, analysis_obj.axes[0], cv, rsq_cutoff=0.99,
                                                   amp_cutoff=params_obj.gaussian_2_int_threshold)

        all_fits = iterate_lmfitting(analysis_obj.axes[0], cv_col_intensities, gaussian_guess_list, cv, params_obj, outputfolder)
        results.append(all_fits)

    for cv_index, cv_results in enumerate(results):
        all_fits = cv_results

        # save the fit with the highest score out of all fits collected
        best_fit = max(all_fits, key=lambda x: x.score)
        best_fits_by_cv.append(best_fit)
        scores_by_cv.append([fit.score for fit in all_fits])

    # output final results
    fit_time = time.time() - start_time

    prot_gaussians = [fit.gaussians_protein for fit in best_fits_by_cv]
    nonprot_gaussians = [fit.gaussians_nonprotein for fit in best_fits_by_cv]

    # Generate centroid plots
    best_centroids = []
    for gauss_list in prot_gaussians:
        best_centroids.append([x.centroid for x in gauss_list])
    nonprot_centroids = []
    for gauss_list in nonprot_gaussians:
        nonprot_centroids.append([x.centroid for x in gauss_list])
    plot_centroids(best_centroids, analysis_obj, params_obj, outputpath, nonprotein_centroids=nonprot_centroids)

    # save results to analysis obj
    analysis_obj.raw_protein_gaussians = prot_gaussians
    analysis_obj.raw_nonprotein_gaussians = nonprot_gaussians
    analysis_obj.gauss_fits_by_cv = best_fits_by_cv

    # save output
    save_fits_pdf_new(analysis_obj, params_obj, best_fits_by_cv, outputpath)
    combined_output, sorted_gauss_by_cv = save_gauss_params(analysis_obj, outputpath, params_obj.gaussian_51_sort_outputs_by, combine=params_obj.gaussian_5_combine_outputs, protein_only=params_obj.gauss_t1_1_protein_mode)
    return analysis_obj, combined_output, sorted_gauss_by_cv, fit_time


def copy_gaussians_from_prevfit(prev_fit, new_cv):
    """
    Generate new 'guess' Gaussians from a previous fitting result without linking directly to the
    previous objects (so they don't get overwritten). Returns a list of new Gaussian objects with
    the same parameters as before except updated CV
    :param prev_fit: SingleFitStats container
    :type prev_fit: SingleFitStats
    :param new_cv: updated CV
    :return: list of Gaussians
    :rtype: list[Gaussian]
    """
    gaussian_guess_list = []
    for gaussian in prev_fit.gaussians:
        new_gaussian = Gaussian(gaussian.amplitude, gaussian.centroid, gaussian.width, new_cv, pcov=None, protein_bool=gaussian.is_protein)
        gaussian_guess_list.append(new_gaussian)
    return gaussian_guess_list


def guess_next_gaussian(ciu_data_col, dt_axis, width_guess, cv, prev_gaussians):
    """
    Simple algorithm to determine an initial guess for the next peak. Subtracts any Gaussians fit so
    far (if present) from the raw arrival time profile, then finds the max of the remaining data. The
    value of the max becomes the amplitude guess and the location becomes the centroid. The expected
    width of the next component is used for width.
    :param ciu_data_col: 1D array of intensity data for the CIU column (raw)
    :param dt_axis: drift time axis values to determine the centroid of the guess
    :param width_guess: expected width of the next peak
    :param cv: collision voltage for the guess
    :param prev_gaussians: list of previous Gaussians fit to this DT profile to subtract
    :type prev_gaussians: list[Gaussian]
    :return: Gaussian peak guess
    :rtype: Gaussian
    """
    # First, subtract existing peaks from current profile
    all_params = []
    for gaussian in prev_gaussians:
        all_params.extend(gaussian.return_popt())

    # Use the Gaussian function to construct intensity data at each DT
    intensities = multi_gauss_func(dt_axis, *all_params)
    dt_profile_data = np.asarray([x for x in ciu_data_col])
    dt_profile_data -= intensities

    # Determine the max and initialize the guess Gaussian
    max_index = np.argmax(dt_profile_data)
    max_value = np.max(dt_profile_data)
    centroid_guess = dt_axis[max_index]
    guess_gaussian = Gaussian(amplitude=max_value, centroid=centroid_guess, width=width_guess, collision_voltage=cv, pcov=None, protein_bool=None)
    return guess_gaussian


def guess_gauss_init(ciu_col, dt_axis, cv, rsq_cutoff, amp_cutoff):
    """
    Generate initial guesses for Gaussians. Currently using just the estimate_multi_params_all method
    with output formatted as Gaussian objects, but will likely try to include initial first round of
    fitting from curve_fit as well.
    :param ciu_col: intensity (y) data at this CV
    :param dt_axis: DT (x) data
    :param cv: collision voltage to record for Gaussians
    :param rsq_cutoff: r2 convergence criterion for initial fitting (peaks are added until r2 above this value)
    :param amp_cutoff: minimum amplitude for peak to be allowed
    :return: list of Gaussian objects with guess parameters
    """
    gaussians = []

    # estimate a (rather inaccurate) list of possible peaks to use as guesses for fitting
    guess_list = estimate_multi_params_all(ciu_col, dt_axis, width_frac=0.01)

    # run the initial (first round) fitting with curve_fit to generate high quality guesses
    popt, pcov, allfits = sequential_fit_rsq(guess_list, dt_axis, ciu_col, cv=cv, convergence_rsq=rsq_cutoff, amp_cutoff=amp_cutoff)

    # convert all guesses to Gaussians and sort in decreasing quality order to provide for future rounds
    r1_guesses = generate_gaussians_from_popt(opt_params_list=popt, protein_bool=True, cv=cv, pcov=pcov)
    gaussians.extend(sorted(r1_guesses, key=lambda x: x.amplitude, reverse=True))

    for param_guess in guess_list:
        # catch 0 amplitude and just make it very small
        if abs(param_guess[0]) < 1e-5:
            param_guess[0] = 1e-5
        gaussians.extend(generate_gaussians_from_popt(param_guess, protein_bool=True, cv=cv, pcov=None))

    return gaussians


def sequential_fit_rsq(all_peak_guesses, dt_axis, cv_col_intensities, cv, convergence_rsq, amp_cutoff):
    """
    Gaussian fitting 1.0 method - adds peak components from a list of initial guesses (provided)
    until r2 value reaches a user specified convergence criterion. Abstracted for use as a way to
    provide high quality initial guesses for more advanced methods, but can also be used in isolation.
    **This method is for a single CV value** and returns the final popt, pcov, and a list of fits from all rounds.
    :param all_peak_guesses: list of lists of initial guess values for parameters. Typically from estimate_multi_params_all
    :param dt_axis: x-axis for fitting (DT axis values)
    :param cv_col_intensities: y-data for fitting (intensity data along the DT axis)
    :param cv: the collision voltage (CV) at which this fitting takes place
    :param convergence_rsq: the minimum rsq at which to stop adding peak components
    :param amp_cutoff: minimum amplitude for peak to be allowed
    :return: popt, pcov, list of SingleFitStats containers for each round of fitting.
    """
    param_guesses_multiple = []
    all_fit_rounds = []
    popt, pcov = [], []

    # set bounds for fitting: keep baseline and centroid on DT axis, amplitude 0 to 1.5, width 0 to len(dt_axis)
    max_dt = dt_axis[len(dt_axis) - 1]
    min_dt = dt_axis[0]
    fit_bounds_lower, fit_bounds_upper = [], []
    fit_bounds_lower_append = [0, min_dt, 0]
    fit_bounds_upper_append = [1, max_dt, len(dt_axis)]

    i = 0
    iterate_gaussian_flag = True
    # Iterate through peak detection until convergence criterion is met, adding one additional peak each iteration
    while iterate_gaussian_flag:
        # Set up initial guesses
        try:
            param_guesses_multiple.extend(all_peak_guesses[i])
            # ensure bounds arrays maintain same shape as parameter guesses
            fit_bounds_lower.extend(fit_bounds_lower_append)
            fit_bounds_upper.extend(fit_bounds_upper_append)
        except IndexError:
            # No converge with all estimated peaks. Continue with final estimate
            logger.warning('Included all {} peaks found, but r^2 still less than convergence criterion. '
                           'Poor fitting possible'.format(i + 1))
            break

        # Run fitting (round 1)
        try:
            popt, pcov = scipy.optimize.curve_fit(f=multi_gauss_func, xdata=dt_axis, ydata=cv_col_intensities,
                                                  method='trf',
                                                  p0=param_guesses_multiple,
                                                  bounds=(fit_bounds_lower, fit_bounds_upper))
            # perr = np.sqrt(np.diag(pcov))
        except (RuntimeError, ValueError):
            popt, pcov = [], []

        current_fit = SingleFitStats(dt_axis, cv_col_intensities, amp_cutoff=amp_cutoff, popt=popt, cv=cv)
        all_fit_rounds.append(current_fit)

        # stop iterating once convergence criteria have been reached
        if not current_fit.adjrsq < convergence_rsq:
            iterate_gaussian_flag = False
        i += 1

    return popt, pcov, all_fit_rounds


def iterate_lmfitting(x_data, y_data, guesses_list, cv, params_obj, outputpath):
    """
    Primary fitting method. Iterates over combinations of protein and non-protein peaks using
    models generated with LMFit based on the initial peak guesses in the guesses_list. Fits are
    evaulated with r2 and scoring functions to determine which number of components gave the
    best fit, which is returned as a MinimizerResult/ModelFitResult from LMFit
    :param x_data: x_axis data for fitting (DT axis)
    :param y_data: y data to fit (intensity values along the DT axis)
    :param guesses_list: list of Gaussian objects corresponding to guesses from the previous CV to use as starting conditions
    :type guesses_list: list[Gaussian]
    :param cv: collision voltage for Gaussians
    :param params_obj: Parameters container with various parameter information
    :type params_obj: Parameters
    :param outputpath: directory in which to save outputs
    :return: best fit result as a MinimizerResult/ModelFitResult from LMFit
    """
    # determine the number of components over which to iterate fitting
    max_num_prot_pks = params_obj.gaussian_71_max_prot_components
    if not params_obj.gauss_t1_1_protein_mode:
        max_num_nonprot_pks = params_obj.gaussian_82_max_nonprot_comps  # params_obj/advanced for more options?
    else:
        max_num_nonprot_pks = 0

    # cv = guesses_list[0].cv
    output_fits = []
    # iterate over all peak combinations
    for num_prot_pks in range(1, max_num_prot_pks + 1):
        # Mass selected mode (no nonprotein peaks)
        if max_num_nonprot_pks == 0:
            num_nonprot_pks = 0
            current_fit, lmfit_output = perform_fit(x_data, y_data, cv, num_prot_pks, num_nonprot_pks, guesses_list, params_obj)
            output_fits.append(current_fit)

            if params_obj.gaussian_4_save_diagnostics:
                outputname = os.path.join(outputpath, '{}_p{}_fits.png'.format(cv, num_prot_pks))
                plot_fit_result(current_fit, lmfit_output, x_data, outputname)
        else:
            # No selection mode - iterate over nonprotein peaks as well
            for num_nonprot_pks in range(params_obj.gaussian_81_min_nonprot_comps, max_num_nonprot_pks + 1):

                # current_fit, lmfit_output = perform_fit(x_data, y_data, cv, num_prot_pks, num_nonprot_pks, guesses_list, extra_guesses, params_obj)
                current_fit, lmfit_output = perform_fit(x_data, y_data, cv, num_prot_pks, num_nonprot_pks, guesses_list, params_obj)
                output_fits.append(current_fit)

                if params_obj.gaussian_4_save_diagnostics:
                    outputname = os.path.join(outputpath, '{}_p{}_np{}_fits.png'.format(cv, num_prot_pks, num_nonprot_pks))
                    plot_fit_result(current_fit, lmfit_output, x_data, outputname)

    return output_fits


def perform_fit(x_data, y_data, cv, num_prot_pks, num_nonprot_pks, guesses_list, params_obj):
    """
    Helper method to improve code readability. Runs fitting for iterate_lmfit with provided data
    and peak options/guesses.
    :param x_data: DT (x) data to fit (ndarray)
    :param y_data: intensity (y) data to fit (ndarray)
    :param cv: collision voltage (float)
    :param num_prot_pks: (int) number of protein components to fit in this iteration
    :param num_nonprot_pks: (int) number of nonprotein components to fit in this iteration
    :param guesses_list: list of peak initial guesses (list of Gaussian objects)
    :type guesses_list: list[Gaussian]
    :param params_obj: Parameters object
    :type params_obj: Parameters
    :return: SingleFitStats container with fit results and score, LMFit output (ModelResult) from fitting
    """
    # assemble the models and fit parameters for this number of protein/non-protein peaks
    models_list, fit_params = assemble_models(num_prot_pks, num_nonprot_pks, params_obj, guesses_list, cv, dt_axis=x_data, dt_profile=y_data)

    # combine all model parameters and perform the actual fitting
    final_model = models_list[0]
    for model in models_list[1:]:
        final_model += model

    # output = final_model.fit(y_data, fit_params, x=x_data, method='leastsq', nan_policy='omit',
    #                          scale_covar=False, fit_kws={'maxfev': 1000})
    output = final_model.fit(y_data, fit_params, x=x_data, method='leastsq', nan_policy='omit', fit_kws={'maxfev': 1000})

    # compute fits and score
    current_fit = SingleFitStats(x_data=x_data, y_data=y_data, cv=cv, lmfit_output=output,
                                 amp_cutoff=params_obj.gaussian_2_int_threshold)
    # only score protein peaks, as non-protein peaks can overlap and have differing widths (may add different score func eventually if needed)
    current_fit.compute_fit_score(params_obj, penalty_scaling=1)
    return current_fit, output


def plot_fit_result(current_fit, output, x_data, outputname):
    """
    Plotting method for diagnostics only. Creates a fit plot for an iteration.
    :param current_fit: fit stats container
    :type current_fit: SingleFitStats
    :param output: LMFit output (ModelResult) from the fitting. Can't be saved to SingleFitStats without breaking pickle-ability
    :param x_data: (ndarray) x (drift axis) data from fitting to plot
    :param outputname: full output filename and path to save plot
    :return: void
    """
    plt.clf()
    model_components = output.eval_components(x=x_data)
    output.plot_fit()
    for component_name, comp_value in model_components.items():
        try:
            plt.plot(x_data, comp_value, '--', label=component_name)
        except ValueError:
            # baseline component will only have a single value, so plot it at all x-axis points
            y_data = [comp_value for _ in range(len(x_data))]
            plt.plot(x_data, y_data, '--', label=component_name)
    plt.legend(loc='best')
    penalty_string = ['{:.2f}'.format(x) for x in current_fit.peak_penalties]
    plt.title('{}V, r2: {:.3f}, score: {:.4f}, peak pens: {}'.format(current_fit.cv, current_fit.adjrsq, current_fit.score,
                                                                     ','.join(penalty_string)))
    plt.savefig(outputname)
    try:
        plt.savefig(outputname)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(outputname))
        plt.savefig(outputname)


# def assemble_models(num_prot_pks, num_nonprot_pks, params_obj, guesses_list, extra_guesses, dt_axis, dt_profile):
def assemble_models(num_prot_pks, num_nonprot_pks, params_obj, guesses_list, cv, dt_axis, dt_profile):
    """
    Assign the peaks in the list of guesses to protein and non-protein components of the final model.
    Guess list is assumed to be in decreasing order of amplitude. Guesses are assigned to non-protein peaks
    if their width is larger than the expected protein width, and to protein peaks otherwise.
    :param num_prot_pks: number of protein components to be fit in this iteration
    :param num_nonprot_pks: number of nonprotein components to be fit in this iteration
    :param params_obj: parameter container
    :type params_obj: Parameters
    :param guesses_list: list of Gaussian objects containing guess information, in descending order of amplitude
    :type guesses_list: list[Gaussian]
    # :param extra_guesses: list of Gaussian objects in decreasing amplitude order for adding new peaks after exceeding the number of peaks from the previous CV
    # :type extra_guesses: list[Gaussian]
    :param cv: collision voltage for Gaussians
    :param dt_axis: x-axis for the fitting
    :param dt_profile: y data for fitting (intensity values for DT profile)
    :return: list of LMFit Models, LMFit Parameters() dictionary
    """
    fit_params = lmfit.Parameters()

    # assemble models for this number of peaks
    guess_index = 0
    total_num_components = num_nonprot_pks + num_prot_pks
    models_list = []

    # Initialize a common baseline for all Gaussians if requested
    if params_obj.gaussian_75_baseline:
        model, params = make_baseline_model(guess_baseline=0.1)
        models_list.append(model)
        fit_params.update(params)

    # counters for numbers of each peak type left to be fitted
    nonprots_remaining = num_nonprot_pks
    prots_remaining = num_prot_pks

    # initialize width guesses (convert from FWHM [user input] to sigma [fitting input])
    prot_width_center = fwhm_to_sigma(params_obj.gaussian_72_prot_peak_width)
    prot_width_tol = fwhm_to_sigma(params_obj.gaussian_73_prot_width_tol)
    min_nonprot_width = fwhm_to_sigma(params_obj.gaussian_83_nonprot_width_min)

    if num_nonprot_pks > 0:
        # non-protein peak(s) present, assign peaks wider than width max for protein to them
        for comp_index in range(0, total_num_components):
            try:
                next_guess = guesses_list[guess_index]
                guess_index += 1
            except IndexError:
                next_guess = guess_next_gaussian(dt_profile, dt_axis, min_nonprot_width, cv, guesses_list)
                guess_index += 1

            if next_guess.width > (prot_width_center + prot_width_tol):
                # the width of this guess is wider than protein - try fitting a nonprotein peak here
                if nonprots_remaining > 0:
                    model, params = make_nonprotein_model(
                        prefix='{}{}'.format(nonprotein_prefix, guess_index),
                        guess_gaussian=next_guess,
                        nonprot_width_min=min_nonprot_width,
                        dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)

                    nonprots_remaining -= 1
                else:
                    # no more non-protein peaks left, so add a protein peak
                    model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index),
                                                       guess_gaussian=next_guess,
                                                       width_center=prot_width_center,
                                                       width_tol=prot_width_tol,
                                                       dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)
                    prots_remaining -= 1
            else:
                # guess peak width is narrow enough to be protein - guess it first
                if prots_remaining > 0:
                    model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index),
                                                       guess_gaussian=next_guess,
                                                       width_center=prot_width_center,
                                                       width_tol=prot_width_tol,
                                                       dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)
                    prots_remaining -= 1
                else:
                    # no protein peaks left, so guess non-protein
                    model, params = make_nonprotein_model(
                        prefix='{}{}'.format(nonprotein_prefix, guess_index),
                        guess_gaussian=next_guess,
                        nonprot_width_min=min_nonprot_width,
                        dt_axis=dt_axis)
                    models_list.append(model)
                    fit_params.update(params)

                    nonprots_remaining -= 1

    else:
        # protein peaks only - simply go through the guess list (descending order of amplitude)
        for prot_pk_index in range(0, num_prot_pks):
            try:
                next_guess = guesses_list[guess_index]
                guess_index += 1
            except IndexError:
                # out of guesses - make a new protein guess by finding the max of the remaining DT profile after subtracting current peaks
                next_guess = guess_next_gaussian(dt_profile, dt_axis, prot_width_center, cv, guesses_list)
                guess_index += 1

            model, params = make_protein_model(prefix='{}{}'.format(protein_prefix, guess_index),
                                               guess_gaussian=next_guess,
                                               width_center=prot_width_center,
                                               width_tol=prot_width_tol,
                                               dt_axis=dt_axis)
            models_list.append(model)
            fit_params.update(params)
            # guess_index += 1

    return models_list, fit_params


def make_protein_model(prefix, guess_gaussian, width_center, width_tol, dt_axis):
    """
    Generate an LMFit model object from initial parameters in the guess_gaussian container and
    parameters.
    :param prefix: string prefix for this model to prevent params from having same names
    :param guess_gaussian: Gaussian object with initial guess parameters
    :type guess_gaussian: Gaussian
    :param width_center: center of allowed width distribution
    :param width_tol: tolerance (width) of allowed peak width distribution
    :param dt_axis: dt_axis information for determining boundaries
    :return: LMFit model object with initialized parameters, bounds, and constraints
    """
    max_dt = dt_axis[-1]
    min_dt = dt_axis[0]

    model = lmfit.Model(gaussfunc, prefix=prefix)
    model_params = model.make_params()

    min_width = width_center - width_tol
    max_width = width_center + width_tol
    if min_width < 0:
        min_width = 1e-3

    # set initial guesses and boundaries
    model_params[prefix + 'centroid'].set(guess_gaussian.centroid, min=min_dt, max=max_dt)
    model_params[prefix + 'sigma'].set(guess_gaussian.width, min=min_width, max=max_width)
    model_params[prefix + 'amplitude'].set(guess_gaussian.amplitude, min=0, max=1.5)

    # return the model
    return model, model_params


def make_nonprotein_model(prefix, guess_gaussian, nonprot_width_min, dt_axis):
    """
    Generate an LMFit model object from initial parameters in the guess_gaussian container and
    parameters.
    :param prefix: string prefix for this model to prevent params from having same names
    :param guess_gaussian: Gaussian object with initial guess parameters
    :type guess_gaussian: Gaussian
    :param nonprot_width_min: minimum width for non-protein peak
    :param dt_axis: dt_axis information for determining boundaries
    :return: LMFit model object with initialized parameters, bounds, and constraints
    """
    max_dt = dt_axis[-1]
    min_dt = dt_axis[0]
    max_width = (max_dt - min_dt) / 2.0     # should not approach the width of the whole DT axis

    model = lmfit.Model(gaussfunc, prefix=prefix)
    model_params = model.make_params()

    # set initial guesses and boundaries
    model_params[prefix + 'centroid'].set(guess_gaussian.centroid, min=min_dt, max=max_dt)
    model_params[prefix + 'sigma'].set(guess_gaussian.width, min=nonprot_width_min, max=max_width)
    model_params[prefix + 'amplitude'].set(guess_gaussian.amplitude, min=0, max=1.5)

    # return the model
    return model, model_params


def make_baseline_model(guess_baseline):
    """
    Make a model of a flat baseline to add to all Gaussian functions in LMFit fitting.
    :param guess_baseline: Initial baseline guess (provided by user)
    :return: LMFit model, model parameters
    """
    model = lmfit.Model(baseline_func, prefix=baseline_prefix)
    model_params = model.make_params()
    model_params[baseline_prefix + 'baseline'].set(guess_baseline, min=1e-10, max=1.0)

    return model, model_params


def gaussfunc(x, amplitude, centroid, sigma):
    """
    Gaussian function with constraints applied for CIU data
    :param x: x
    :param amplitude: gaussian amplitude (constrained to be positive)
    :param centroid: gaussian centroid
    :param sigma: gaussian width
    :return: y = f(x)
    """
    exponent = ((x - centroid)**2) / (2 * (sigma**2))
    y = amplitude * (np.exp(-exponent))         # using this function since our data is always normalized
    # y = amplitude/(np.sqrt(2*np.pi) * sigma) * (np.exp(-exponent))     # use this for non-normalized data

    return y


def baseline_func(x, baseline):
    """
    Flat baseline function that simply adds a set offset to the y-axis at all values of x. Separated
    from Gaussian function so that all Gaussian components will have the same baseline.
    :param x: input x value (not actually needed, since function does not vary with x, but included for fitting)
    :param baseline: flat baseline offset in y direction.
    :return: (float) baseline
    """
    return baseline


def multi_gauss_func(x, *params):
    """
    Attempt at basic multi-gaussian fitting by passing multiple parameter sets and generating a sum
    of gaussians
    :param x: data to fit
    :param params: List of [amplitdue, centroid x, width] parameters, one set for each Gaussian to fit (in order)
    :return: y = f(x), where f(x) describes a gaussian function
    """
    y = np.zeros_like(x)
    # make a gaussian function for each set of parameters in the input list
    for i in range(0, len(params), 3):
        y = y + gaussfunc(x, params[i], params[i+1], params[i+2])
    return y


def estimate_multi_params(ciu_col, dt_axis, width_frac, peak_int_threshold=0.1, min_spacing_bins=10):
    """
    Make initial guesses for a sum of gaussians fitting
    :param ciu_col: 1D numpy array representing the DT spectrum in a given column (CV)
    :param dt_axis: drift time data (x axis to the fitted gaussian's y) for peak indexing
    :param width_frac: estimation of peak width (DT * fraction), typically 10% has been found to work well
    :param peak_int_threshold: Minimum intensity threshold to detect a peak for fitting
    :param min_spacing_bins: Minimum distance between peaks IN BINS - should be about instrument resolution
    :return: list of [centroid, width, amplitude] initial guesses
    """
    # estimate the number of components by doing a simple peak finding using PeakUtils
    peak_indices = peakutils.indexes(ciu_col, thres=peak_int_threshold, min_dist=min_spacing_bins)

    params_lists = []
    # for each estimated peak/component, compute initial guess parameters for gaussian fitting
    for peak_index in peak_indices:
        centroid_guess = dt_axis[peak_index]    # centroid is the DT at the index of the peak
        amp_guess = ciu_col[peak_index]         # amplitude is the value at the index of the peak
        width_guess = peak_index * width_frac
        params_lists.extend([amp_guess, centroid_guess, width_guess])
        # params_lists.append([0, centroid_guess, amp_guess, width_guess])
    return params_lists


def estimate_multi_params_all(ciu_col, dt_axis, width_frac):
    """
    Make initial guesses for peak locations, but ensure overestimation. SciPy find_peaks_cwt tends to
    way overestimate peaks (at least if a small width range is provided), but the fitting algorithm works
    by passing increasing numbers of peaks until the fit converges, so we need to have a large number of
    peaks to provide.
    :param ciu_col: 1D numpy array representing the DT spectrum in a given column (CV)
    :param dt_axis: drift time data (x axis to the fitted gaussian's y) for peak indexing
    :param width_frac: estimation of peak width (DT * fraction), typically 10% has been found to work well
    :return: list of [centroid, width, amplitude] initial guesses
    """
    # estimate the number of components by doing a simple peak finding using CWT - since it tends to give lots of peaks
    peak_ind_scipy = scipy.signal.find_peaks_cwt(ciu_col, np.arange(1, 5))

    params_lists = []
    # for each estimated peak/component, compute initial guess parameters for gaussian fitting
    for peak_index in peak_ind_scipy:
        centroid_guess = dt_axis[peak_index]    # centroid is the DT at the index of the peak
        amp_guess = ciu_col[peak_index]         # amplitude is the value at the index of the peak
        width_guess = peak_index * width_frac
        params_lists.append([amp_guess, centroid_guess, width_guess])

    # sort guesses by amplitude (index 1 in each sublist) in order from largest to smallest
    params_lists = sorted(params_lists, key=lambda x: x[0], reverse=True)
    return params_lists


def adjrsquared(r2, df):
    """
    Compute adjusted r2 given the number of degrees of freedom in an analysis
    :param r2: original r2 value (float)
    :param df: degrees of freedom (int)
    :return: adjusted r2
    """
    y = 1 - (((1-r2)*(df-1))/(df-4-1))
    return y


def fwhm_to_sigma(fwhm):
    """
    Convert FWHM (full-width at half max) to sigma for Gaussian function
    :param fwhm: (float) peak full-width at half max
    :return: sigma (float)
    """
    return fwhm / (2.0 * (math.sqrt(2 * math.log(2))))


def sigma_to_fwhm(sigma):
    """ Convert sigma to FWHM (full-width at half max) for Gaussian function
    :param sigma: float
    :return: float
    """
    return sigma * 2.0 * math.sqrt(2 * math.log(2))



def filter_fits(params_list, peak_width_cutoff, intensity_cutoff, centroid_bounds=None):
    """
    Simple filter to remove any peaks with a width above a specified cutoff. Intended to separate
    noise 'peaks' from protein peaks as they differ in observed width
    :param params_list: list of optimized parameters from curve fit
    :param peak_width_cutoff: maximum allowed width for a peak to remain in the list
    :param intensity_cutoff: minimum relative intensity to remain in the list
    :param centroid_bounds: list of [lower bound, upper bound] for peak centroid (in ms)
    :return: filtered params_list, with peaks above the width cutoff removed
    """
    index = 0
    filtered_list = []
    while index < len(params_list):
        # test if the peak meets all conditions for inclusion
        include_peak = False

        # ensure peak width (FWHM) is below the cutoff and above 0
        fwhm = 2 * math.sqrt(2 * math.log(2)) * params_list[index + 2]
        if 0 < fwhm < peak_width_cutoff:
            # also remove amplitdues below the intensity cutoff
            if params_list[index] > intensity_cutoff:
                if centroid_bounds is not None:
                    # centroid bounds provided - if matched, include the peak
                    if centroid_bounds[0] < params_list[index + 1] < centroid_bounds[1]:
                        include_peak = True
                elif params_list[index + 1] > 0:
                    # If no bounds provided lso remove centroids < 0
                    include_peak = True

        if include_peak:
            filtered_list.extend(params_list[index:index + 3])
        index += 3
    return filtered_list


def get_popt_from_lmoutput(modelresult, amp_cutoff):
    """
    Generate a list of parameters in the same format as curve_fit (popt) for easy conversion
    to old plotting and result saving methods.
    :param modelresult: ModelResult object from LMFit (returned from model.fit())
    :param amp_cutoff: minimum amplitude for a peak to be included
    :return: list of Gaussian parameters [amp1, cent1, sigma1, amp2, cent2, sigma2, ... ] for protein and non-protein components
    """
    # convert dictionary of key/value parameters into a single list of values
    keys = sorted(modelresult.best_values.keys())
    protein_output_popt = [modelresult.best_values[key] for key in keys if key.startswith(protein_prefix)]
    nonprotein_output_popt = [modelresult.best_values[key] for key in keys if key.startswith(nonprotein_prefix)]

    # remove low amplitude peaks
    protein_output_popt = remove_low_amp(protein_output_popt, amp_cutoff)
    nonprotein_output_popt = remove_low_amp(nonprotein_output_popt, amp_cutoff)

    return protein_output_popt, nonprotein_output_popt


def generate_gaussians_from_popt(opt_params_list, protein_bool, cv=None, pcov=None):
    """
    Convert a list of parameters to a list of Gaussian objects. Initializes Gaussians with a collision voltage
    and covariance matrix if provided.
    :param opt_params_list: list of parameters [amp, centroid, width, amp2, cent2, width2, ... ]
    :param protein_bool: protein Gaussians (True) or non-protein (False)
    :param cv: (optional) collision voltage to associate with all Gaussians
    :param pcov: (optional) covariance matrix from fitting to associate with all Gaussians
    :return: list of Gaussian objects from params list
    :rtype: list[Gaussian]
    """
    index = 0
    gaussian_list = []
    while index < len(opt_params_list):
        gaussian_list.append(Gaussian(amplitude=opt_params_list[index],
                                      centroid=opt_params_list[index + 1],
                                      width=opt_params_list[index + 2],
                                      collision_voltage=cv,
                                      pcov=pcov,
                                      protein_bool=protein_bool))
        index += 3
    return gaussian_list


def remove_low_amp(popt_list, amp_cutoff):
    """
    Helper method to remove low amplitude peaks for both protein and non-protein parameter lists
    Also remove peaks of miniscule width, as these can result in similar behavior.
    NOTE: the width cutoff is calculated against centroid to allow for different magnitude drift axes
    sigma must be > 0.01 * centroid, which corresponds to a resolution of 100 * 2sqrt(2), above any
    typical IM system today.
    :param popt_list: list of Gaussian parameters [amp1, centroid1, sigma1, amp2, centroid2, sigma2, ... ]
    :param amp_cutoff: minimum amplitude to allow
    :return: updated popt_list with low amplitude peaks removed
    """
    values_to_remove = []
    for index, value in enumerate(popt_list):
        # amplitude screen (amplitudes are the first of 3 values)
        if index % 3 == 0:
            current_amplitude = value
            if current_amplitude < amp_cutoff:
                values_to_remove.extend([popt_list[index], popt_list[index + 1], popt_list[index + 2]])

        # check for small width as well
        if index % 3 == 2:
            current_width = value
            if current_width < 1e-2:
                values_to_remove.extend([popt_list[index - 2], popt_list[index - 1], popt_list[index]])

    for value in values_to_remove:
        try:
            popt_list.remove(value)
        except ValueError:
            # this value has already been removed (because it failed both amplitude and width cutoffs) - ignore
            continue
    return popt_list


def compute_width_penalty(input_width, expected_width, tolerance, steepness):
    """
    Penalty scoring method using width. Probably uncessary due to the constraints provided
    to LMFit, but provides a backup for over-wide protein peaks.
    :param input_width: observed peak width
    :param expected_width: expected peak width
    :param tolerance: tolerance on expected width
    :param steepness: degree to which to penalize widths outside tolerance (default 1)
    :return: penalty (float)
    """
    diff = abs(input_width - expected_width)
    if diff < tolerance:
        return 0
    else:
        penalized_width = abs(diff - tolerance)
        return steepness * penalized_width


def compute_area_penalty(gaussian, list_of_gaussians, dt_axis, penalty_mode):
    """
    Shared area penalty intended to penalize peaks that are almost completely overlapped
    by others.
    :param gaussian: Gaussian object to compare agaisnt the rest of the list
    :param list_of_gaussians: all gaussians currently fit at this CV
    :param dt_axis: x axis array over which to compute overlap
    :param penalty_mode: (string) 'strict', 'relaxed', or 'none' - determines which function to use for computing area penalties
    :return: penalty (float)
    """
    if penalty_mode == 'none':
        return 0

    # for this gaussian, compute how much area it shares with the rest of the list
    total_penalty = 0
    # for gaussian in list_of_gaussians:
    my_area = scipy.integrate.trapz(gaussfunc(dt_axis, *gaussian.return_popt()), dt_axis)

    other_gaussians = [x for x in list_of_gaussians if x is not gaussian]
    shared_areas = []
    for other in other_gaussians:
        shared_areas.append(shared_area_gauss(dt_axis, gaussian.return_popt(), other.return_popt()))

    # compute shared area (ratio from 0 to 1) and any penalties if > 0.25 (not much until 0.5)
    max_shared_area = np.max(shared_areas)
    shared_area_ratio = max_shared_area / my_area
    if shared_area_ratio > 0.25:
        if penalty_mode == 'strict':
            my_penalty = (1.25 * shared_area_ratio - 0.25) ** 4
        elif penalty_mode == 'relaxed':
            my_penalty = (shared_area_ratio - 0.4) ** 4
        else:
            logger.error('invalid penalty mode: {}. Penalty set to 0'.format(penalty_mode))
            my_penalty = 0

        total_penalty += my_penalty

    return total_penalty


def shared_area_gauss(x_axis, gauss1_params, gauss2_params):
    """
    Compute a "shared area score" (shared area normalized against the area of the smaller peak being compared)
    and return it.
    :param x_axis: x-axis on which to plot the gaussian functions (doesn't matter as long as it's sufficiently sampled)
    :param gauss1_params: the parameters describing gaussian 1 [amplitude, centroid, width]
    :param gauss2_params: the parameters describing gaussian 2 [amplitude, centroid, width]
    :return: shared area
    """
    # shared area is the area under the lower curve
    gauss1 = gaussfunc(x_axis, *gauss1_params)
    gauss2 = gaussfunc(x_axis, *gauss2_params)
    shared_area_arr = []

    # for each point along the x (DT) axis, determine the amount of shared area
    for index in np.arange(0, len(x_axis)):
        if gauss1[index] > gauss2[index]:
            shared_area_arr.append(gauss2[index])
        else:
            shared_area_arr.append(gauss1[index])

    # return the integrated area over the provided axis
    return scipy.integrate.trapz(shared_area_arr, x_axis)


def save_fits_pdf_new(analysis_obj, params_obj, best_fit_list, outputpath):
    """
    Save a PDF file with the best fit results at each collision voltage as a page of the PDF.
    Fit components are plotted individually along with the raw data and combined fit. Scores
    and r2 value are printed at the top of each plot.
    :param analysis_obj: analysis object with data to plot
    :type analysis_obj: CIUAnalysisObj
    :param best_fit_list: list of SingleFitStats objects from each collision voltage
    :param params_obj: Parameter container
    :type params_obj: Parameters
    :param outputpath: directory in which to save output
    :return: void
    """
    gauss_name = analysis_obj.short_filename + '_GaussFits.pdf'
    gauss_fig = os.path.join(outputpath, gauss_name)
    try:
        pdf_fig = matplotlib.backends.backend_pdf.PdfPages(gauss_fig)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(gauss_fig))
        pdf_fig = matplotlib.backends.backend_pdf.PdfPages(gauss_fig)

    intarray = np.swapaxes(analysis_obj.ciu_data, 0, 1)
    for cv_index in range(len(analysis_obj.axes[1])):
        plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

        best_fit = best_fit_list[cv_index]

        # plot the original raw data as a scatter plot
        plt.scatter(analysis_obj.axes[0], intarray[cv_index])

        # plot the combined 'best fit' data
        plt.plot(best_fit.x_data, best_fit.y_fit, color='black', label='Combined Fit')

        # plot each component individually
        prot_index = 1
        for prot_gauss in best_fit.gaussians_protein:
            gauss_fit = gaussfunc(best_fit.x_data, prot_gauss.amplitude, prot_gauss.centroid, prot_gauss.width)
            plt.plot(best_fit.x_data, gauss_fit, ls='--', label='Signal {}'.format(prot_index))
            prot_index += 1
        nonprot_index = 1
        for nonprot_gauss in best_fit.gaussians_nonprotein:
            gauss_fit = gaussfunc(best_fit.x_data, nonprot_gauss.amplitude, nonprot_gauss.centroid, nonprot_gauss.width)
            plt.plot(best_fit.x_data, gauss_fit, ls='--', label='Noise {}'.format(nonprot_index))
            nonprot_index += 1
        if best_fit.baseline_val > 0:
            plt.plot(best_fit.x_data, [best_fit.baseline_val for _ in range(len(best_fit.x_data))], ls='--', label='baseline: {:.2f}'.format(best_fit.baseline_val))

        # plot titles, labels, and legends
        if params_obj.plot_08_show_axes_titles:
            plt.xlabel(params_obj.plot_10_y_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
            plt.ylabel('Relative Intensity', fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.xticks(fontsize=params_obj.plot_13_font_size)
        plt.yticks(fontsize=params_obj.plot_13_font_size)
        if params_obj.plot_07_show_legend:
            plt.legend(loc='best', fontsize=params_obj.plot_13_font_size)

        # penalty_string = ['{:.2f}'.format(x) for x in best_fit.peak_penalties]
        plt.title('{}V, r2: {:.3f}, score: {:.4f}'.format(analysis_obj.axes[1][cv_index], best_fit.adjrsq, best_fit.score))

        pdf_fig.savefig()
        plt.close()
    pdf_fig.close()


def plot_centroids(centroid_lists_by_cv, analysis_obj, params_obj, outputpath, nonprotein_centroids=None):
    """
    Save a png image of the centroid DTs fit by gaussians
    :param centroid_lists_by_cv: list of [list of centroids]s at each collision voltage
    :param nonprotein_centroids: non-protein components list of [list of centroids]s at each collision voltage
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param params_obj: Parameters container with plot params
    :type params_obj: Parameters
    :param outputpath: directory in which to save output
    :return: void
    """
    plt.clf()
    plt.figure(figsize=(params_obj.plot_03_figwidth, params_obj.plot_04_figheight), dpi=params_obj.plot_05_dpi)

    # plot centroids at each collision voltage
    for x, y in zip(analysis_obj.axes[1], centroid_lists_by_cv):
        plt.scatter([x] * len(y), y, color='b', s=params_obj.plot_14_dot_size ** 2, edgecolors='black')

    # plot non-protein components in red if they are present
    nonprotein_flag = False
    if nonprotein_centroids is not None:
        for x, y in zip(analysis_obj.axes[1], nonprotein_centroids):
            try:
                plt.scatter([x] * len(y), y, color='r', s=params_obj.plot_14_dot_size ** 2, edgecolors='black')
                if y:
                    # only label noise peaks if they are present (if any of the lists is non-empty)
                    nonprotein_flag = True
            except TypeError:
                # empty list - continue to next cv
                continue

    # plot titles, labels, and legends
    if params_obj.plot_12_custom_title is not None:
        plot_title = params_obj.plot_12_custom_title
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    elif params_obj.plot_11_show_title:
        plot_title = analysis_obj.short_filename
        plt.title(plot_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
    if params_obj.plot_08_show_axes_titles:
        plt.xlabel(params_obj.plot_09_x_title, fontsize=params_obj.plot_13_font_size, fontweight='bold')
        plt.ylabel('Peak Centroid', fontsize=params_obj.plot_13_font_size, fontweight='bold')
    plt.xticks(fontsize=params_obj.plot_13_font_size)
    plt.yticks(fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_07_show_legend:
        handles = [matplotlib.patches.Patch(color='b', label='Signal Centroids')]
        if nonprotein_flag:
            handles.append(matplotlib.patches.Patch(color='r', label='Noise Centroids'))
        plt.legend(handles=handles, loc='best', fontsize=params_obj.plot_13_font_size)
    if params_obj.plot_15_grid_bool:
        plt.grid('on')

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

    output_name = analysis_obj.short_filename + '_centroids' + params_obj.plot_02_extension
    output_path = os.path.join(outputpath, output_name)
    try:
        plt.savefig(output_path)
    except PermissionError:
        messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(output_path))
        plt.savefig(output_path)
    plt.close()


def save_gauss_params(analysis_obj, outputpath, sort_type, combine=False, protein_only=False):
    """
    Save all gaussian information to file
    :param analysis_obj: container with gaussian fits to save
    :type analysis_obj: CIUAnalysisObj
    :param outputpath: directory in which to save output
    :param combine: whether to return the string to be combined with other files (True) or save this file without combining (False)
    :param sort_type: string: 'centroid', 'amplitude', or 'width'. How to sort the output Gaussians
    :param protein_only: whether using protein only mode. Saves non-protein (noise) information if not
    :return: if combine is True, returns formatted output string and list of sorted gaussians (protein only)
    """
    output_name = analysis_obj.short_filename + '_gaussians.csv'
    output_string = ''

    # save DT information too to allow for reconstruction
    output_string += '# {}'.format(analysis_obj.short_filename)
    dt_line = ','.join([str(x) for x in analysis_obj.axes[0]])
    output_string += 'Drift axis:,' + dt_line + '\n'
    if not protein_only:
        output_string += '# Protein Gaussians\n'
    output_string += '# CV,Amplitude,Centroid,Peak Width (FWHM)\n'
    index = 0
    sorted_gaussians_by_cv = []
    while index < len(analysis_obj.axes[1]):
        if sort_type == 'amplitude':
            # sort in decreasing amplitude order (rather than increasing, as in centroid/width)
            sorted_gaussians = sorted(analysis_obj.raw_protein_gaussians[index], key=lambda x: x.__getattribute__(sort_type), reverse=True)
        else:
            sorted_gaussians = sorted(analysis_obj.raw_protein_gaussians[index], key=lambda x: x.__getattribute__(sort_type))
        outputline = ','.join([gaussian.print_info() for gaussian in sorted_gaussians])
        output_string += outputline + '\n'
        sorted_gaussians_by_cv.append(sorted_gaussians)
        index += 1

    if not protein_only:
        index = 0
        output_string += '# Non-Protein Gaussians\n'
        output_string += '# CV,Amplitude,Centroid,Peak Width (FWHM)\n'
        while index < len(analysis_obj.axes[1]):
            if len(analysis_obj.raw_nonprotein_gaussians[index]) > 0:
                if sort_type == 'amplitude':
                    sorted_gaussians = sorted(analysis_obj.raw_nonprotein_gaussians[index], key=lambda x: x.__getattribute__(sort_type), reverse=True)
                else:
                    sorted_gaussians = sorted(analysis_obj.raw_nonprotein_gaussians[index], key=lambda x: x.__getattribute__(sort_type))
                gauss_line = ','.join([gaussian.print_info() for gaussian in sorted_gaussians])
                output_string += gauss_line + '\n'
            index += 1

    if combine:
        # output_string += '\n'
        return output_string, sorted_gaussians_by_cv
    else:
        try:
            with open(os.path.join(outputpath, output_name), 'w') as output:
                output.write(output_string)
        except PermissionError:
            messagebox.showerror('Please Close the File Before Saving', 'The file {} is being used by another process! Please close it, THEN press the OK button to retry saving'.format(os.path.join(outputpath, output_name)))
            with open(os.path.join(outputpath, output_name), 'w') as output:
                output.write(output_string)
        return ''


def print_combined_params(all_files_gaussians_by_cv, filenames):
    """
    Takes input sorted Gaussian lists from all files being combined (and all CVs within each file) and
    assembles lists of outputs sorted by parameter type (i.e. a list of centroids, then a list of amplitudes, etc).
    :param all_files_gaussians_by_cv: list of files[list of CVs in each file[list of Gaussians at each CV]]]
    :param filenames: list of filenames to coordinate outputs from each file
    :return: string formatted for output
    """
    output_string = '\n'
    params = ['centroid', 'amplitude', 'width']
    for param in params:
        param_string = 'File,CV,' + param + '\n'
        for index, file_lists in enumerate(all_files_gaussians_by_cv):
            for cv_list in file_lists:
                cv_line = '{},{},'.format(filenames[index], cv_list[0].cv)
                cv_line += ','.join([gaussian.print_single_param(param) for gaussian in cv_list])
                # cv_line += ','.join(['{:.2f}'.format(gaussian.__getattribute__(param)) for gaussian in cv_list])
                param_string += cv_line + '\n'
        output_string += param_string
    return output_string


def reconstruct_from_fits(gaussian_lists_by_cv, axes, new_filename, params_obj):
    """
    Construct a new analysis object using the filtered Gaussian fits of the provided analysis object
    as the raw data. Must have previously performed Gaussian feature detection on the provided analysis_obj
    :param gaussian_lists_by_cv: list of lists of Gaussian objects at each CV
    :param axes: [DT_axis, CV_axis]: two numpy arrays with drift and CV axes to use
    :param new_filename:
    :param params_obj: Parameters container with parameters to save into the new CIUAnalsis obj
    :return: new CIUAnalysisObj with reconstructed raw data
    :rtype: CIUAnalysisObj
    """
    ciu_data_by_cols = []
    dt_axis = axes[0]
    # construct the raw data at each collision voltage to stitch together into a CIU matrix
    for cv_gauss_list in gaussian_lists_by_cv:
        # assemble all the parameters for Gaussians at this CV
        all_params = []
        for gaussian in cv_gauss_list:
            all_params.extend(gaussian.return_popt())

        # Use the Gaussian function to construct intensity data at each DT
        intensities = multi_gauss_func(dt_axis, *all_params)

        ciu_data_by_cols.append(intensities)

    # finally, transpose the CIU data to match the typical format, normalize, and return the object
    final_data = np.asarray(ciu_data_by_cols).T
    final_data = Raw_Processing.normalize_by_col(final_data)

    raw_obj = CIU_raw.CIURaw(final_data, dt_axis, axes[1], new_filename)
    new_analysis_obj = CIU_analysis_obj.CIUAnalysisObj(raw_obj, final_data, axes, params_obj)
    new_analysis_obj.short_filename = new_analysis_obj.short_filename + '_gauss-recon'

    new_analysis_obj.protein_gaussians = gaussian_lists_by_cv
    new_analysis_obj.gaussians = gaussian_lists_by_cv
    return new_analysis_obj


def check_recon_for_crop(gaussian_lists_by_cv, axes):
    """
    Crops any columns from the beginning or end of a fingerprint being reconstructed that
    have no Gaussians fit to them. Gaps with no Gaussians between features are left alone
    and are handled gracefully by Feature Detection and Classification (or by smoothing).
    :param gaussian_lists_by_cv: list of Gaussian lists by CV
    :param axes: axes from the original analysis object, CV axis must be same length as gaussian_lists_by_cv
    :return: cropped gaussian_lists_by_cv, cropped axes
    """
    cv_axis = axes[1]
    true_start_index = 0
    true_end_index = 0
    start_flag = True

    for index, gaussian_list in enumerate(gaussian_lists_by_cv):
        if len(gaussian_list) == 0:
            if start_flag:
                # this list is empty AND we haven't yet found a non-empty list, so crop it
                true_start_index += 1
        else:
            # not empty, so this can't be the true end
            true_end_index = index
            start_flag = False

    # do the cropping
    final_lists = gaussian_lists_by_cv[true_start_index: true_end_index + 1]
    final_cv_axis = cv_axis[true_start_index: true_end_index + 1]
    return final_lists, [axes[0], final_cv_axis]


def parse_gaussian_list_from_file(filepath):
    """
    Read in a list of Gaussians from file and return a list of Gaussian objects.
    ** NOTE: provide widths as FWHM, not sigma **
    File format: (comma delimited text file, headers (#) ignored)
    CV1, gauss1_amp, gauss1_cent, gauss1_FWHM, gauss2 A, c, w, ..., gaussN a, c, w
    CV2, gauss1_amp, (etc)
    CV3
    ...
    :param filepath: full path to file to read
    :return: list of lists of Gaussian objects sorted by collision voltage, list of collision voltages
    """
    gaussian_list_by_cv = []
    cvs = []
    dt_axis = []
    with open(filepath, 'r') as inputfile:
        for line in list(inputfile):
            # skip headers
            if line.startswith('#'):
                continue
            splits = line.rstrip('\n').split(',')
            splits = [x for x in splits if x is not '']

            # read DT axis
            if line.lower().startswith('drift'):
                try:
                    dt_axis = np.asarray([float(x) for x in splits[1:]])
                except ValueError:
                    logger.error('DT axis could not be read. Line was: {}'.format(line))
                    dt_axis = []
                continue

            # get CVs from first column only to avoid duplicates
            try:
                cv = float(splits[0])
                cvs.append(cv)
            except ValueError:
                logger.error('Invalid CV in line: {}; value must be a number. Skipping this line'.format(line))
                continue
            except IndexError:
                # no data on this line, continue
                continue

            # read remaining Gaussian information
            index = 0
            gaussians = []
            while index < len(splits) - 1:
                try:
                    cv = float(splits[index])
                    amp = float(splits[index + 1])
                    cent = float(splits[index + 2])
                    width = fwhm_to_sigma(float(splits[index + 3]))
                    gaussians.append(Gaussian(amp, cent, width, cv, pcov=None, protein_bool=True))
                except (IndexError, ValueError):
                    logger.error('Invalid values for Gaussian. Values were: {}. Gaussian could not be parsed and was skipped'.format(splits[index:index + 3]))
                index += 4
            gaussian_list_by_cv.append(gaussians)

    return gaussian_list_by_cv, [dt_axis, np.asarray(cvs)]


# testing
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    root_dir = os.path.dirname(__file__)
    hard_params_file = os.path.join(root_dir, 'CIU2_param_info.csv')
    myparams = CIU_Params.Parameters()
    myparams.set_params(CIU_Params.parse_params_file(hard_params_file))

    files = filedialog.askopenfilenames(filetypes=[('CIU', '.ciu')])
    for file in files:
        with open(file, 'rb') as analysis_file:
            current_analysis_obj = pickle.load(analysis_file)

        centroids = []
        for mygauss_list in current_analysis_obj.raw_protein_gaussians:
            centroids.append([x.centroid for x in mygauss_list])
        np_centroids = []
        for mygauss_list in current_analysis_obj.raw_nonprotein_gaussians:
            np_centroids.append([x.centroid for x in mygauss_list])
        plot_centroids(centroids, current_analysis_obj, myparams, os.path.dirname(current_analysis_obj.filename), np_centroids)

"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Dan Polasky
10/6/17
"""
import os
import numpy as np
import tkinter
from tkinter import filedialog
import pickle

# typing to allow easier refactoring of custom objects
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import CIU_Params
    import CIU_raw
    import Feature_Detection
    import Gaussian_Fitting


class CIUAnalysisObj(object):
    """
    Container for analysis/processed information from a CIU fingerprint. Requires a CIURaw object
    to start, additional fields added as data is processed.
    """
    def __init__(self, ciu_raw_obj, ciu_data, axes, params_obj, short_filename=None):
        """
        Initialize with raw data and axes. Allows addition of Gaussian fitting data later
        :param ciu_raw_obj: Object containing initial raw data, axes, and filepath of the analysis
        :param ciu_data: pre-processed data (smoothed/interpolated/cropped/etc) - can be modified repeatedly
        :param axes: modified axes corresponding to ciu_data (axes[0] = DT, axes[1] = CV)
        :param params_obj: Parameters object with information about how this object was processed
        :type params_obj: Parameters
        """
        # basic information and objects
        self.raw_obj = ciu_raw_obj  # type: CIU_raw.CIURaw
        self.raw_obj_list = None    # used for replicates (averaged fingerprints) only
        self.ciu_data = ciu_data
        self.axes = axes            # convention: axis 0 = DT, axis 1 = CV
        self.crop_vals = None
        self.params = params_obj  # type: CIU_Params.Parameters
        self.filename = None        # filename of .ciu file saved
        # allow for saving modified short_filenames
        if short_filename is None:
            self.short_filename = os.path.basename(self.raw_obj.filename).rstrip('_raw.csv')
        else:
            self.short_filename = short_filename

        # CIU data manipulations for common use
        self.col_maxes = np.argmax(self.ciu_data, axis=0)       # Index of maximum value in each CV column (in DT bins)
        self.col_max_dts = [self.axes[0][x] for x in self.col_maxes]       # DT of maximum value

        # Feature detection results
        self.transitions = []   # type: List[Feature_Detection.Transition]
        self.features_gaussian = None   # type: List[Feature_Detection.Feature]
        self.features_changept = None   # type: List[Feature_Detection.Feature]

        # Gaussian fitting results - raw and following feature detection included
        self.raw_protein_gaussians = None       # type: List[List[Gaussian_Fitting.Gaussian]]
        self.raw_nonprotein_gaussians = None    # type: List[List[Gaussian_Fitting.Gaussian]]
        self.feat_protein_gaussians = None      # type: List[List[Gaussian_Fitting.Gaussian]]
        self.gauss_fits_by_cv = None            # type: List[Gaussian_Fitting.SingleFitStats]

        # classification (unknown) outputs
        self.classif_predicted_label = None
        self.classif_transformed_data = None
        self.classif_probs_by_cv = None
        self.classif_probs_avg = None

        self.classif_gaussians_by_cv = None   # type: List[List[Gaussian_Fitting.Gaussian]]  # list of Gaussian containers, prior to being prepared into the classif_input_raw matrix
        self.classif_input_raw = None   # for classification, to allow prepared Gaussians and raw data to be used from same field after prep
        self.classif_input_std = None   # for classification, data that has been standardized and ready to classify

    def __str__(self):
        """
        Display the filename of the object as reference
        :return: void
        """
        return '<CIUAnalysisObj> file: {}'.format(os.path.basename(self.filename.rstrip('.ciu')))
    __repr__ = __str__

    def refresh_data(self):
        """
        Recalculate column max values and other basic data attributes. Should be performed after any
        adjustments to the ciu_data (e.g. crop, interpolate, smooth, etc)
        :return:
        """
        self.col_maxes = np.argmax(self.ciu_data, axis=0)  # Index of maximum value in each CV column (in DT bins)
        self.col_max_dts = [self.axes[0][x] for x in self.col_maxes]
        # self.col_max_dts = [self.axes[0][0] + (x - 1) * self.bin_spacing for x in self.col_maxes]  # DT of maximum value

    def get_features(self, gaussian_bool):
        """
        Returns feature list from CIUAnalysisObj for a given mode. Basically just making sure
        keywords are accessed in a single location in case of refactoring/etc
        :param gaussian_bool: (boolean) True if Gaussian mode, False if not
        :return: (boolean) True if features present, False if not
        """
        if gaussian_bool:
            return self.features_gaussian
        else:
            return self.features_changept


# testing
if __name__ == '__main__':
    root = tkinter.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(filetypes=[('pickled gaussian files', '.pkl')])
    files = list(files)
    file_dir = os.path.dirname(files[0])

    for file in files:
        with open(file, 'rb') as first_file:
            ciu1 = pickle.load(first_file)
        ciu1.plot_centroids(file_dir, [10, 20])
        ciu1.save_gauss_params(file_dir)

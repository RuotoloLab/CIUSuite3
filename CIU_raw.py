"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Raw data object for CIU-type analyses.
Author: Dan Polasky
Date: 10/6/2017
"""
import os
import numpy as np


class CIURaw:
    """
    Raw data container for CIU-type data. Analogous to the _raw.csv file in original CIUSuite.
    Holds raw data, axis information, and metadata/parameters as needed
    """
    def __init__(self, raw_data_no_axes, drift_axis, activation_axis, filepath):
        """
        Constructor - requires input data and axis information
        :param raw_data_no_axes: 2D numpy array containing raw data. Rows (axis 0) refer to drift time,
        columns (axis 1) refer to activation steps.
        :param activation_axis: List of activation values corresponding to columns of raw data matrix
        :param drift_axis: List of drift time values corresponding to rows of raw data matrix
        :param filepath: full system path to raw file used
        """
        self.rawdata = raw_data_no_axes
        self.dt_axis = drift_axis
        self.cv_axis = activation_axis
        self.filepath = filepath
        self.filename = os.path.basename(filepath)


def get_data(fname):
    """
    Read _raw.csv file and generate a CIURaw object containing its raw data and filename
    :param fname: string - path to _raw.csv file to read
    :rtype: CIURaw
    :return: CIURaw object with rawdata, axes, and filename initialized
    """
    rawdata = np.genfromtxt(fname, missing_values=[""], filling_values=[0], delimiter=",")
    row_axis = rawdata[1:, 0]
    col_axis = rawdata[0, 1:]
    raw_obj = CIURaw(rawdata[1:, 1:], row_axis, col_axis, fname)
    return raw_obj


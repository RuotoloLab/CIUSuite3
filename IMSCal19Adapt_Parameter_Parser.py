"""
Parsing a .cvs file with calibration infromation to pass to IMSCal19
Author: Carolina Rojas Ramirez
Most code adapted from DP's code.
Date: Dec 6, 2021
"""

import pandas as pd
from tkinter import filedialog
import os

def parse_param(param_file):
    """
    Read template csv file for all parameter information.
    :param param_file: (string) full system path to csv template file
    """

    params_dict = {}


    fingerprint_indx = 0
    with open(param_file, 'r') as pfile:
        for line in list(pfile):
            if line.startswith('#'):
                continue
            elif line.startswith('*'):

                fingerprint_indx += 1
                splits = line.rstrip('\n').split(',')
                #Initilize params object
                params = Parameters()
                params.range_file_params = splits[1:8]
                params.MW = splits[8]
                params.charge = splits[9]

                #IMSCal19 files paths required
                params.IMSCal19_input = splits[10]
                params.IMSCal19_cal = splits[11]
                params.IMSCal19_output = splits[12]

                params.IMSCal19_args = splits[13:]



            else:
                splitsb = line.rstrip('\n').split(',')
                params.rawfile.append(splitsb[0])

            params_dict[fingerprint_indx] = params

    print(f"params = {params_dict}")
    return params_dict

def parse_bool(param_string):
    """
    Parse input strings to boolean
    :param param_string: string
    :return: bool
    """
    if param_string.lower() in ['t', 'true', 'yes', 'y']:
        return True
    elif param_string.lower() in ['f', 'false', 'no', 'n']:
        return False
    else:
        raise ValueError('Invalid boolean: {}'.format(param_string))

class Parameters(object):
    """
    Container to hold all parameters for searches to simplify method calls/etc
    """

    def __init__(self):
        """
        No parameters initialized immediately
        """
        self.params_dict = {}

        # ion prediction parameters
        self.rawfile = []
        self.range_file_params = None
        self.MW = None
        self.charge = None
        self.IMSCal19_input = None
        self.IMSCal19_cal = None
        self.IMSCal19_output = None
        self.IMSCal19_args = None
        self.TWIMExtractOutput = None

    def set_params(self, params_dict):
        """
        Set a series of parameters given a dictionary of (parameter name, value) pairs
        :param params_dict: Dictionary, key=param name, value=param value
        :return: void
        """
        for name, value in params_dict.items():
            try:
                # only set the attribute if it is present in the object - otherwise, raise attribute error
                self.__getattribute__(name)
                self.__setattr__(name, value)
            except AttributeError:
                # no such parameter
                print('No parameter name for param: ' + name)
                continue
        self.update_dict()

    def update_dict(self):
        """
        Build (or rebuild) a dictionary of all attributes contained in this object
        :return: void
        """
        for field in vars(self):
            value = self.__getattribute__(field)
            self.params_dict[field] = value

    def __str__(self):
        """
        string
        :return: string
        """
        return '<Params> File(s) {}'.format(self.rawfile)
    __repr__ = __str__

if __name__ == "__main__":

    batch_file = filedialog.askopenfilename(title='Batch Hits Files', filetypes=[('CSV File', '.csv')])
    dictParams = parse_param(batch_file)

    print(dictParams)

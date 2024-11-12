"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Module for raw (instrument vendor) data import into CIUSuite2. Currently includes Waters
and Agilent handling based on TWIMExtract and MIDAC, respectively.
2/1/2018
DP
"""
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import sys
import os
import shutil
import subprocess
import numpy as np
import tkinter
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import logging
import pygubu
import IMSCal19Adapt_Parameter_Parser
import fromTWIMtoIMSCal
import Breuker_Extractor



logger = logging.getLogger('main')


# File chooser for raw data, created after extensive searching on stack overflow
class FileDialog(QtWidgets.QFileDialog):
    """
    Generate a directory-only filechooser, as both Waters and Agilent raw data comes in folders
    rather than files
    """
    def __init__(self, input_dir, *args):
        QtWidgets.QFileDialog.__init__(self, *args)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)
        self.setDirectory(input_dir)

        self.tree = self.findChild(QtWidgets.QTreeView)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)


def get_data(input_dir):
    """
    Run the QtWidget directory mode filedialog and return filelist generated
    :param input_dir: path to the initial directory for the file chooser
    :return: list of strings of full system folder paths to the folders chosen, updated input_dir
    """
    app = QtWidgets.QApplication(sys.argv)
    ex = FileDialog(input_dir)
    ex.show()
    app.exec_()
    files = ex.selectedFiles()

    # return empty list if user hits 'cancel' button
    if ex.result() == 0:
        files = []

    return files


def check_data(file_list, accepted_endings):
    """
    Run checks to confirm that all files selected are either .raw or .d folders prior to continuing
    :param file_list: list of file path strings
    :param accepted_endings: list of acceptable file ending strings (e.g. '.raw')
    :return: True if data is valid, False if not
    """
    for file in file_list:
        splits = os.path.basename(file).split('.')
        file_ending = splits[len(splits) - 1]
        if file_ending not in accepted_endings:
            return False

    # no problems found, return True
    return True


def twimex_single_range(range_info, raw_files, save_dir, extractor_path):
    """
    Use TWIMExtract to extract _raw.csv files for a single set of range information for all raw files specified.
    Uses DT mode extraction and combines each output from the file to generate _raw.csv files that are ready for
    CIUSuit2 import
    :param range_info: list of [range_name, mz_low, mz_high, rt_low, rt_high, dt_low, dt_high] in m/z, mins, bins, respectively
    :param raw_files: list of paths from which to extract raw data (.raw folders)
    :param save_dir: directory in which to save _raw.csv output
    :param extractor_path: full system path to TWIMExtract.jar
    :return: list of filenames extracted
    """
    # Write rangefile into temp dir
    temp_dir = os.path.join(save_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    range_path = write_rangefile(range_info, temp_dir)
    dt_mode = 1

    logger.info('Starting TWIMExtract. NOTE: Extraction may take some time! CIUSuite 3 will not respond until TWIMExtract is done!')
    raw_files_combined = ','.join(raw_files)
    run_extractor(extractor_path, raw_files_combined, temp_dir, mode_int=dt_mode, range_path=range_path, combine_bool=True)

    # for index, raw_file in enumerate(raw_files):
    #     # Method to write output file given the splits array from a line in the template file
    #     print('Starting TWIMExtract run {} of {}. NOTE: Extraction may take some time!'.format(index + 1, len(raw_files)))
    #     run_extractor(extractor_path, raw_file, temp_dir, mode_int=dt_mode, range_path=range_path, combine_bool=True)

    # Get the extracted files, move them to the new directory, and return a list of filepaths for further analysis
    extracted_files = os.listdir(temp_dir)
    extracted_files = [x for x in extracted_files if x.endswith('_raw.csv')]
    final_filepaths = []
    for ext_file in extracted_files:
        old_path = os.path.join(temp_dir, ext_file)
        new_path = os.path.join(save_dir, os.path.basename(ext_file))
        os.replace(old_path, new_path)
        final_filepaths.append(new_path)

    # empty the temp directory and return final filepaths for further analysis
    shutil.rmtree(temp_dir, ignore_errors=True)
    return final_filepaths


def write_rangefile(info, outputdir):
    """
    Create a range file using the information provided in the info list
    :param info: List of range info. 0 = rangename; 1,2 = m/z start, end; 3,4 = RT start, end; 5,6 = DT start, end
    :param outputdir: directory in which to save output
    :return: void
    """
    output_path = os.path.join(outputdir, info[0] + '.txt')

    # Constants for writing range file index names
    mzstart_text = 'MZ_start_(m/z): '
    mzend_text = 'MZ_end_(m/z): '
    rtstart_text = 'RT_start_(minutes):'
    rtend_text = 'RT_end_(minutes): '
    dtstart_text = 'DT_start_(bins): '
    dtend_text = 'DT_end_(bins): '

    with open(output_path, 'w') as rangefile:
        rangefile.write(mzstart_text + str(info[1]) + '\n')
        rangefile.write(mzend_text + str(info[2]) + '\n')
        rangefile.write(rtstart_text + str(info[3]) + '\n')
        rangefile.write(rtend_text + str(info[4]) + '\n')
        rangefile.write(dtstart_text + str(info[5]) + '\n')
        rangefile.write(dtend_text + str(info[6]))

    return output_path


def run_extractor(ext_path, input_path, output_path, mode_int, func_num=None, range_path=None, rule_bool=None, combine_bool=None):
    """
    Run the tool with specified arguments (after formatting them, combining to 1 string, and adding the tool dir)
    :param ext_path: full system path to TWIMExtract.jar
    :param input_path: Full path to the input .raw folder to extract from
    :param output_path: Full path to directory in which to place output file(s)
    :param mode_int: 0 = RT, 1 = DT, 2 = MZ. Dimension to save upon extraction
    :param func_num: Function number of file to analyze. If not supplied, defaults to extracting all functions
    :param range_path: OPTIONAL. Full path to the range file to use to extract. If not specified, full range used
    :param rule_bool: OPTIONAL. 'true' or 'false'. If true, program will expect range_arg to point to a .rul file
    :param combine_bool: OPTIONAL. 'true' or 'false'. If true, outputs from the same raw file will be combined
    :return: none
    """
    # format arguments that are present. If range/rule/combine are not supplied, do not pass anything
    input_arg = '-i "{}"'.format(input_path)    # use quotes to allow spaces/etc in filenames
    output_arg = '-o "{}"'.format(output_path)
    mode_arg = '-m {}'.format(mode_int)
    if func_num is not None:
        func_arg = '-f {}'.format(func_num)
    else:
        func_arg = ''
    if range_path is not None:
        range_arg = '-r "{}"'.format(range_path)
    else:
        range_arg = ''
    if rule_bool is not None:
        rule_arg = '-rulemode {}'.format(rule_bool)
    else:
        rule_arg = ''
    if combine_bool is not None:
        combine_arg = '-combinemode {}'.format(combine_bool)
    else:
        combine_arg = ''

    tool_arg = 'java -jar "{}"'.format(ext_path)
    arg_list = [tool_arg, input_arg, output_arg, mode_arg, func_arg, range_arg, rule_arg, combine_arg]
    arg_fmt = ['{}'.format(x) for x in arg_list]
    args = ' '.join(arg_fmt)

    completed_proc = subprocess.run(args)
    if not completed_proc.returncode == 0:
        # process finished successfully
        logger.error('Error in extraction for file {} with range file {}. Data NOT extracted. Check that this is a Waters raw data file and that appropriate range values were provided.'.format(input_path, range_path))


def read_agilent_and_correct(filename, cv_axis_to_use, overwrite=True):
    """
    Parse through the file and edit as needed. Saves a copy in the same directory as the input file,
    unless overwrite is True, in which case it replaces the input file with an updated one.
    :param filename: full path to file to edit
    :param cv_axis_to_use: list of collision voltage (or other activation values) to use. Must be same
    length as number of columns in the input file.
    :param overwrite: whether to overwrite the original file or save a copy with '_edit' appended
    :return: void
    """
    edited_lines = []

    # check how many 0 lines to remove
    rawdata = np.genfromtxt(filename, missing_values=[""], filling_values=[0], delimiter=",")
    dt_axis = rawdata[1:, 0]
    num_zeros = np.count_nonzero(dt_axis == 0)
    zeros_skipped_so_far = 0

    with open(filename) as original_file:
        # read through the file, saving all lines (except those we're skipping)
        for index, line in enumerate(original_file):
            splits = line.rstrip('\n').split(',')
            if index == 0:
                # save header information
                if not len(splits) - 1 == len(cv_axis_to_use):
                    logger.error('CV axis provided is NOT the same length as the CV axis in file {}, skipping file'.format(os.path.basename(filename)))
                    return
                new_header_info = ','.join([str(x) for x in cv_axis_to_use])
                new_header = splits[0] + ',' + new_header_info + '\n'
                edited_lines.append(new_header)

            else:
                # all other lines: skip zeros if needed, otherwise, save
                if zeros_skipped_so_far < num_zeros - 1:
                    zeros_skipped_so_far += 1
                else:
                    # save this line
                    edited_lines.append(line)

    # save a copy of the file with the edited information
    if overwrite:
        new_filename = filename
    else:
        new_filename = filename.rstrip('_raw.csv') + '_edit_raw.csv'
    with open(new_filename, 'w') as new_file:
        for line in edited_lines:
            new_file.write(line)


def ask_input_cv_data(original_header):
    """
    Method to prompt the user to enter the CV axis data for an Agilent file into a dialog.
    Displays the original header values in a table for the user to fill out with appropriate CV values
    for each. Returns parsed values as a list.
    :param original_header: list of strings parsed from the original file header line.
    :return: list of CV values, success boolean
    """
    # root = tkinter.Tk()
    # root.withdraw()

    output_cvs = run_header_ui(original_header)
    if output_cvs is not None:
        return output_cvs, True
    else:
        return [], False


def get_header(filename):
    """
    Return the header of a _raw.csv file. SKIPS FIRST ENTRY, as that isn't part of the header for _raw.csv files
    :param filename: full path to file
    :return: list of strings parsed from the first line of the file
    """
    with open(filename, 'r') as myfile:
        line = myfile.readline()
        splits = line.rstrip('\n').split(',')
        return splits[1:]


def run_header_ui(original_header_list):
    """
    Run the HeaderUI graphical menu
    :param original_header_list: list of strings containing original header information
    :return: list of floats - return values from the HeaderUI, or None if the user canceled or something failed
    """
    header_ui = HeaderUI(original_header_list)
    header_ui.lift()

    # prevent users from hitting multiple windows simultaneously
    header_ui.grab_set()
    header_ui.wait_window()
    header_ui.grab_release()

    # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
    if header_ui.return_code == 0:
        return header_ui.return_vals
    else:
        return None


def run_twimex_ui():
    """
    Run the TWIMExRangeUI grpahical menu
    :return: void
    """
    range_ui = TWIMExRangeUI()
    range_ui.lift()

    # prevent users from hitting multiple windows simultaneously
    range_ui.grab_set()
    range_ui.wait_window()
    range_ui.grab_release()

    # Only update parameters if the user clicked 'okay' (didn't click cancel or close the window)
    if range_ui.return_code == 0:
        return range_ui.return_vals
    else:
        return None

#For Winter 2022 patch
class WatersImportTypeUI(object):
    """
    Simple dialog with several fields build with Pygubu for inputting crop values
    """
    def __init__(self, ui_file, twimextracjar):
        # Get crop input from the Crop_vals UI form
        self.builder = pygubu.Builder()
        self.twimxtractexecutable = twimextracjar

        # load the UI file
        self.builder.add_from_file(ui_file)
        # create widget using provided root (Tk) window
        self.mainwindow = self.builder.get_object('watersimport')

        # Connect Delete event to a toplevel window
        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        self.return_code = None


        callbacks = {
            'on_button_waterdt_clicked': self.on_button_driftimextraction_clicked,
            'on_button_watersccs_clicked': self.on_button_ccsextraction_clicked
        }
        self.builder.connect_callbacks(callbacks)





    def run(self):
        """
        Run the UI and return the output values
        :return: List of crop values [dt low, dt high, cv low, cv high]
        """
        self.builder.get_object('watersimport').grab_set()     # prevent users from clicking other stuff while crop is active
        self.mainwindow.mainloop()
        self.builder.get_object('watersimport').grab_release()
        return self.return_code


    def on_close_window(self):
        """
        Quit the mainwindow to stop the mainloop and get it to return
        the crop values, then destroy it to remove it from screen.
        :return: the provided crop values, or None if none were provided
        """
        self.mainwindow.quit()
        self.mainwindow.destroy()

    def on_button_driftimextraction_clicked(self):
        """
        Runs TWIMExtraction using the TWIMExtract included in CIUSuite2
        (supposely the same as the standalone version - done on March, 2021)
        """
        param_file = filedialog.askopenfilename(title='File with Parameters for TWIMExtract', filetypes=[('CSV File', '.csv')])

        paramsdict = IMSCal19Adapt_Parameter_Parser.parse_param(param_file)

        print(paramsdict)

        finishbool = fromTWIMtoIMSCal.twimextraction_forCIUSuitetwo(paramsdict,self.twimxtractexecutable, ccs_mode=False)

        self.return_code = finishbool

        if self.return_code:
            self.mainwindow.destroy()


    def on_button_ccsextraction_clicked(self):
        """
        Runs TWIMExtraction using the TWIMExtract included in CIUSuite2 and then runs IMSCal19
        (supposely the same as the standalone version - done on March, 2021)
        """
        param_file = filedialog.askopenfilename(title='File with Parameters for TWIMExtract and IMSCal19',
                                                filetypes=[('CSV File', '.csv')])

        paramsdict = IMSCal19Adapt_Parameter_Parser.parse_param(param_file)

        print(paramsdict)

        finishbool = fromTWIMtoIMSCal.twimextraction_forCIUSuitetwo(paramsdict, self.twimxtractexecutable, ccs_mode=True)

        self.return_code = finishbool

        if self.return_code:
            self.mainwindow.destroy()

class HeaderUI(tkinter.Toplevel):
    """
    Graphical menu for editing Agilent headers
    """

    def __init__(self, original_header_list):
        """
        Initialize the graphical menu with one line for each value in the original header
        :param original_header_list: list of strings containing original header information
        """
        tkinter.Toplevel.__init__(self)
        self.title('Enter the Activation Values')
        self.return_code = -2

        # return values = list of header values
        self.return_vals = []
        self.entry_vars = []
        self.labels = []

        # display a label (name) and entry (value) for each part of the original header
        row = 1
        labels_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        labels_frame.grid(column=0, row=0)
        ttk.Label(labels_frame, text='Enter the correct activation values for each file/segment in the CIU dataset').grid(row=0, column=0, sticky='e', columnspan=2)
        for header_string in original_header_list:
            entry_var = tkinter.StringVar()
            self.labels.append(header_string)
            # display the original header and an entry box for the new value
            ttk.Label(labels_frame, text=header_string).grid(row=row, column=0, sticky='e')
            ttk.Entry(labels_frame, textvariable=entry_var, width=25).grid(row=row, column=1)

            self.entry_vars.append(entry_var)
            row += 1

        # Finally, add 'okay' and 'cancel' buttons to the bottom of the dialog
        button_frame = ttk.Frame(self, padding='5 5 5 5')
        button_frame.grid(column=0, row=1)
        ttk.Button(button_frame, text='Cancel', command=self.cancel_button_click).grid(row=0, column=0, sticky='w')
        ttk.Label(button_frame, text='                           ').grid(row=0, column=1)
        ttk.Button(button_frame, text='OK', command=self.ok_button_click).grid(row=0, column=2, sticky='e')

    def ok_button_click(self):
        """
        When the user clicks 'OK', confirm that all values are valid and if so, close the window.
        If not, prompt for corrective action.
        All values are saved into self.return_vals to be retrieved in the run_header_ui method
        :return: void
        """
        fail_vals = []
        for index, entry_var in enumerate(self.entry_vars):
            entry_string = entry_var.get()
            try:
                self.return_vals.append(float(entry_string))
            except ValueError:
                fail_vals.append((self.labels[index], entry_string))

        if len(fail_vals) == 0:
            # no params failed, so close window and update parameter values
            self.return_code = 0
            self.on_close_window()
        else:
            # some parameters failed. Tell the user which ones and keep the window open
            error_string = 'The following entry(s) have inappropriate (non-number) values:\n'
            for fail_tup in fail_vals:
                error_string += 'entry {}: {}\n'.format(fail_tup[0], fail_tup[1])
            messagebox.showwarning(title='Entry Error', message=error_string)
            return

    def cancel_button_click(self):
        """
        Close the window and return without updating any parameter values
        :return: void
        """
        self.return_code = -1
        self.on_close_window()

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()


class TWIMExRangeUI(tkinter.Toplevel):
    """
    Graphical menu manually inputting a single range file for TWIMExtract-based CIU data extraction
    """
    range_headers = ['Output File Name', 'm/z Min', 'm/z Max', 'Retention/scan time Min', 'Retention/scan time Max', 'Drift Bin Min', 'Drift Bin Max']
    range_defaults = ['', '', '', 0, 100, 1, 200]

    def __init__(self):
        """
        Initialize the graphical menu with one line for each value in the range file
        """
        tkinter.Toplevel.__init__(self)
        self.title('Enter the Range Information')
        self.return_code = -2

        # return values = list of header values
        self.return_vals = []
        self.entry_vars = []
        self.labels = []

        # display a label (name) and entry (value) for each part of the original header
        row = 1
        labels_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        labels_frame.grid(column=0, row=0)
        ttk.Label(labels_frame, text='Enter the data range to extract. Generally, this describes ONE charge state of one analyte\n\n***NOTE: for batch processing and more options, please run TWIMExtract directly***\n(TWIMExtract can be found in <your CIUSuite 2 install folder>/TWIMExtract)').grid(row=0, column=0, sticky='e', columnspan=2)
        for index, header_string in enumerate(TWIMExRangeUI.range_headers):
            entry_var = tkinter.StringVar()
            entry_var.set(TWIMExRangeUI.range_defaults[index])
            self.labels.append(header_string)
            # display the original header and an entry box for the new value
            ttk.Label(labels_frame, text=header_string).grid(row=row, column=0, sticky='e')
            ttk.Entry(labels_frame, textvariable=entry_var, width=25).grid(row=row, column=1)

            self.entry_vars.append(entry_var)
            row += 1

        # Finally, add 'okay' and 'cancel' buttons to the bottom of the dialog
        button_frame = ttk.Frame(self, padding='5 5 5 5')
        button_frame.grid(column=0, row=1)
        ttk.Button(button_frame, text='Cancel', command=self.cancel_button_click).grid(row=0, column=0, sticky='w')
        ttk.Label(button_frame, text='                           ').grid(row=0, column=1)
        ttk.Button(button_frame, text='OK', command=self.ok_button_click).grid(row=0, column=2, sticky='e')

    def ok_button_click(self):
        """
        When the user clicks 'OK', confirm that all values are valid and if so, close the window.
        If not, prompt for corrective action. First entry (name) can be any string aside from empty, other
        values must be floats.
        All values are saved into self.return_vals to be retrieved in the run_header_ui method
        :return: void
        """
        fail_vals = []

        # Parse the range name
        name_entry = self.entry_vars[0].get()
        if len(name_entry) == 0:
            fail_vals.append((self.labels[0], name_entry))
        else:
            self.return_vals.append(name_entry)

        # Parse the range values
        for index, entry_var in enumerate(self.entry_vars[1:]):
            entry_string = entry_var.get()
            try:
                self.return_vals.append(float(entry_string))
            except ValueError:
                fail_vals.append((self.labels[index], entry_string))

        # Check for any improper values
        if len(fail_vals) == 0:
            # no params failed, so close window and update parameter values
            self.return_code = 0
            self.on_close_window()
        else:
            # some parameters failed. Tell the user which ones and keep the window open
            error_string = 'The following entry(s) have inappropriate (non-number) values:\n'
            for fail_tup in fail_vals:
                error_string += 'entry {}: {}\n'.format(fail_tup[0], fail_tup[1])
            messagebox.showwarning(title='Entry Error', message=error_string)
            return

    def cancel_button_click(self):
        """
        Close the window and return without updating any parameter values
        :return: void
        """
        self.return_code = -1
        self.on_close_window()

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()

# Winter 2024 Update

class BreukerImportTypeUI(object):
    """
    Simple dialog with several fields build with Pygubu for inputting crop values
    """
    def __init__(self, ui_file, mainpythonfile):

        # Get crop input from the Crop_vals UI form
        self.builder = pygubu.Builder()
        self.breukerminapy = mainpythonfile

        # load the UI file
        self.builder.add_from_file(ui_file)
        # create widget using provided root (Tk) window
        self.mainwindow = self.builder.get_object('breukerimport')

        # Connect Delete event to a toplevel window
        self.mainwindow.protocol('WM_DELETE_WINDOW', self.on_close_window)

        self.return_code = None




        callbacks = {
            'on_checkbox_batchmode_clicked': self.on_checkbox_batchmode_clicked,
            'on_checkbox_singlefolder_clicked': self.on_checkbox_singlefolder_clicked,
        }
        self.builder.connect_callbacks(callbacks)







    def run(self):
        """
        Run the UI and return the output values
        :return: List of crop values [dt low, dt high, cv low, cv high]
        """
        self.builder.get_object('breukerimport').grab_set()     # prevent users from clicking other stuff while crop is active
        self.mainwindow.mainloop()
        self.builder.get_object('breukerimport').grab_release()
        return self.return_code

    def on_checkbox_batchmode_clicked(self):
        """
        Runs TWIMExtraction using the TWIMExtract included in CIUSuite2 and then runs IMSCal19
        (supposely the same as the standalone version - done on March, 2021)
        """

        # Obtaining entry values for single folder extractions
        minimunmz = self.builder.get_object("minmz_val").get()
        maxmz = self.builder.get_object("maxmz_val").get()

        #
        # self.return_code = finishbool
        #
        # if self.return_code:
        #     self.mainwindow.destroy()

    def on_checkbox_singlefolder_clicked(self):
        """
        Runs TWIMExtraction using the TWIMExtract included in CIUSuite2 and then runs IMSCal19
        (supposely the same as the standalone version - done on March, 2021)
        """


        # Obtaining entry values for single folder extractions
        minimunmz = self.builder.get_object("minmz_val").get()
        maxmz = self.builder.get_object("maxmz_val").get()
        ionmz = self.builder.get_object("ionmz_val").get()
        ioncharge = self.builder.get_object("ioncharge_val").get()

        if ionmz == 0 or ioncharge == 0:
            finishbool = Breuker_Extractor.main(float(minimunmz), float(maxmz))
        else:
            finishbool = Breuker_Extractor.main(minimunmz, maxmz, [ionmz, ioncharge])

        self.return_code = finishbool

        if self.return_code:
            self.mainwindow.destroy()


    def on_close_window(self):
        """
        Quit the mainwindow to stop the mainloop and get it to return
        the crop values, then destroy it to remove it from screen.
        :return: the provided crop values, or None if none were provided
        """
        self.mainwindow.quit()
        self.mainwindow.destroy()


if __name__ == '__main__':
    # testing
    # raw_dialog = FileDialog()
    # raw_file_dirs = get_data()
    # print(raw_file_dirs)
    #
    # #
    # mypath = r"C:\Users\Dan-7000\Desktop\AgilentCIU_memfixed\release\MIDAC_CIU_Extractor.exe"
    run_twimex_ui()

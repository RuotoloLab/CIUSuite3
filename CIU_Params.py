"""
This file is part of CIUSuite 2
Copyright (C) 2018 Daniel Polasky

Module for Parameter object to hold parameter information in analysis objects and associated
parameter UI, etc
"""
import tkinter
import numpy as np
from tkinter import messagebox
from tkinter import ttk
import logging

logger = logging.getLogger('main')


def parse_param_descriptions(param_file):
    """
    Read in parameter descriptions and requirements from text (csv) file
    :param param_file: file to read (full system path)
    :return: dictionaries of 1) parameter display names, 2) parameter descriptions, 3) parameter
    requirements. All dicts will have keys corresponding to attributes of the Parameters object
    """
    names = {}
    descriptions = {}
    reqs = {}

    with open(param_file) as p_file:
        lines = list(p_file)
        for line in lines:
            # skip header
            if line.startswith('#'):
                continue

            line = line.rstrip('\n')
            # parse a parameter name and description from the line
            splits = line.split(',')
            key = splits[0].strip()
            names[key] = splits[2].strip()
            descriptions[key] = splits[7].strip()

            # parse parameter requirements
            param_type = splits[3].strip()
            if param_type == 'int':
                # parse lower and upper bounds
                if splits[4].strip() == 'ninf':
                    lower_bound = -np.inf
                else:
                    lower_bound = int(splits[4].strip())
                if splits[5].strip() == 'inf':
                    upper_bound = np.inf
                else:
                    upper_bound = int(splits[5].strip())
                reqs[key] = (param_type, [lower_bound, upper_bound])
            elif param_type == 'float':
                # parse lower and upper bounds
                if splits[4].strip() == 'ninf':
                    lower_bound = -np.inf
                else:
                    lower_bound = float(splits[4].strip())
                if splits[5].strip() == 'inf':
                    upper_bound = np.inf
                else:
                    upper_bound = float(splits[5].strip())
                reqs[key] = (param_type, [lower_bound, upper_bound])
            elif param_type == 'string' or param_type == 'bool':
                req_vals = [x.strip() for x in splits[6].strip().split(';')]
                # convert 'none' strings to actual Nonetype
                # for index, value in enumerate(req_vals):
                #     if value == 'none':
                #         req_vals[index] = None
                reqs[key] = (param_type, req_vals)
            elif param_type == 'anystring':
                reqs[key] = (param_type, [])
            else:
                logger.error('invalid type, parsing failed for line: {}'.format(line))

    return names, descriptions, reqs


class Parameters(object):
    """
    Object to hold all parameters used in generation of a CIU_analysis object. Starts with
    nothing initialized and adds parameters over time.
    """

    def __init__(self):
        """
        Initialize an empty parameters object with all params set to None
        """
        self.params_dict = {}

        self.gui_1_size = None

        # Smoothing and processing parameters
        self.smoothing_1_method = None
        self.smoothing_2_window = None
        self.smoothing_3_iterations = None
        self.interpolate_1_axis = None
        self.interpolate_2_scaling = None
        self.interpolate_3_onedim = None

        # Plotting and saving output parameters
        self.ciuplot_cmap_override = None
        self.plot_01_cmap = None
        self.plot_02_extension = None
        self.plot_03_figwidth = None
        self.plot_04_figheight = None
        self.plot_05_dpi = None
        self.plot_06_show_colorbar = None
        self.plot_07_show_legend = None
        self.plot_08_show_axes_titles = None
        self.plot_09_x_title = None
        self.plot_10_y_title = None
        self.plot_11_show_title = None
        self.plot_12_custom_title = None
        self.plot_13_font_size = None
        self.plot_14_dot_size = None
        self.plot_15_grid_bool = None
        self.plot_16_xlim_lower = None
        self.plot_17_xlim_upper = None
        self.plot_18_ylim_lower = None
        self.plot_19_ylim_upper = None

        self.output_1_save_csv = None
        self.compare_batch_1_both_dirs = None
        self.compare_2_custom_red = None
        self.compare_1_custom_blue = None
        self.compare_3_high_contrast = None
        self.compare_4_int_cutoff = None

        # Gaussian fitting
        self.gauss_t1_1_protein_mode = None
        self.gaussian_2_int_threshold = None
        self.gaussian_4_save_diagnostics = None
        self.gaussian_5_combine_outputs = None
        self.gaussian_51_sort_outputs_by = None
        self.gaussian_61_num_cores = None
        self.gaussian_71_max_prot_components = None
        self.gaussian_72_prot_peak_width = None
        self.gaussian_73_prot_width_tol = None
        self.gaussian_74_shared_area_mode = None
        self.gaussian_75_baseline = None
        self.gaussian_81_min_nonprot_comps = None
        self.gaussian_82_max_nonprot_comps = None
        self.gaussian_83_nonprot_width_min = None
        self.gaussian_9_nonprot_min_prot_amp = None

        self.reconstruct_1_mode = None

        self.feature_t1_1_ciu50_mode = None  # gaussian or standard mode
        self.feature_t2_1_min_length = None
        self.feature_t2_2_width_tol = None
        self.feature_t2_3_ciu50_gap_tol = None
        self.feature_t2_4_gauss_fill_gaps = None
        self.feature_t2_5_gauss_allow_nongauss = None
        self.feature_t2_6_ciu50_combine_outputs = None
        self.feature_t2_7_ciu50_concise_outputs = None
        self.feature_t3_1_exclude_feature = None

        self.ciu50_t2_1_centroiding_mode = None
        self.ciu50_t2_2_pad_transitions_cv = None
        self.ciu50_t2_3_gauss_width_adj_tol = None

        self.class_t1_1_load_method = None
        self.class_t1_2_subclass_mode = None
        self.classif_6_ufs_use_error_mode = None
        self.classif_2_score_dif_tol = None
        self.classif_4_score_mode = None
        self.classif_1_input_mode = None
        self.classif_3_auto_featselect = None
        self.classif_5_show_auc_crossval = None
        self.classif_7_max_feats_for_crossval = None
        self.classif_8_max_crossval_iterations = None
        self.classif_91_test_size = None
        self.classif_92_standardize = None
        self.classif_93_std_all_gsns_bool = None
        self.silent_clf_4_num_gauss = None

        # Raw data import parameters
        self.vendor_1_type = None
        self.silent_filechooser_dir = None

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
                logger.warning('No parameter name for param: ' + name)
                continue
        self.update_dict()

    def print_params_to_console(self):
        """
        Method to read all parameters out to the console (alphabetical order)
        :return: void
        """
        for paramkey, value in sorted(self.__dict__.items()):
            print('{}: {}'.format(paramkey, value))

    def compare(self, other_params):
        """
        Compare all parameters in this object to another parameters object. Returns True if
        all parameters are identical, False if not
        :param other_params: Parameters object
        :return: boolean
        """
        for param_key in self.params_dict.keys():
            if param_key == 'params_dict':
                continue
            if not self.params_dict[param_key] == other_params.params_dict[param_key]:
                return False
        return True

    def update_dict(self):
        """
        Build (or rebuild) a dictionary of all attributes contained in this object
        :return: void
        """
        for field in vars(self):
            value = self.__getattribute__(field)
            self.params_dict[field] = value


def parse_params_file_newcsv(params_file):
    """
    Parse a CIU2_param_info.csv file for all parameters. Returns a params_dict that can be used to
    set_params on a Parameters object
    :param params_file: File to parse (.csv), headers = '#'
    :return: params_dict: Dictionary, key=param name, value=param value
    """
    param_dict = {}
    try:
        with open(params_file, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    continue
                splits = line.rstrip('\n').split(',')
                value = splits[1].strip()

                # catch 'None' values and convert to None
                if value == 'None':
                    param_dict[splits[0].strip()] = None
                else:
                    # try parsing numbers
                    try:
                        try:
                            param_dict[splits[0].strip()] = int(value)
                        except ValueError:
                            param_dict[splits[0].strip()] = float(value)
                    except ValueError:
                        # string value - try parsing booleans or leave as a string
                        if value.lower() in ['true', 't', 'yes', 'y']:
                            param_dict[splits[0].strip()] = True
                        elif value.lower() in ['false', 'f', 'no', 'n']:
                            param_dict[splits[0].strip()] = False
                        else:
                            param_dict[splits[0].strip()] = splits[1].strip()
        return param_dict
    except FileNotFoundError:
        logger.error('params file not found!')


def update_param_csv(params_obj, params_filepath):
    """
    Update the existing parameters default values file with new defaults in the provided
    parameters object
    :param params_obj: Parameters to save
    :type params_obj: Parameters
    :param params_filepath: location of the file to edit (format: same as parsed by parse_params_file_newcsv)
    :return: void
    """
    try:
        edited_lines = []
        with open(params_filepath, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    edited_lines.append(line)
                    continue
                splits = line.rstrip('\n').split(',')

                # Update ONLY the value field (splits[1]) and save the new line
                current_key = splits[0].strip()
                try:
                    new_value = str(params_obj.params_dict[current_key])
                except KeyError:
                    new_value = splits[1].strip()
                    logger.error('Parameter {} not found, default unchanged').format(current_key)
                splits[1] = new_value
                new_line = ','.join(splits) + '\n'
                edited_lines.append(new_line)

        # write the updated information back to the file (overwrite old file)
        with open(params_filepath, 'w') as pfile:
            for line in edited_lines:
                pfile.write(line)

    except FileNotFoundError:
        logger.error('Parameters file {} not found! Default values not changed'.format(params_filepath))


def update_specific_param_vals(dict_of_updates, params_filepath):
    """
    Same as updating param csv with method above, but only updates specified key/value pairs (in the
    dict_of_updates) and leaves the rest of the param file unchanged
    :param dict_of_updates: dictionary of key/value pairs to update. Values should be the new value to save
    :param params_filepath: full system path to params csv file to save
    :return: void
    """
    try:
        edited_lines = []
        with open(params_filepath, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    edited_lines.append(line)
                    continue
                splits = line.rstrip('\n').split(',')

                # Update ONLY the value field (splits[1]) and save the new line
                current_key = splits[0].strip()
                try:
                    if current_key in dict_of_updates.keys():
                        new_value = str(dict_of_updates[current_key])
                    else:
                        # not a key to edit, keep value unchanged
                        new_value = splits[1].strip()
                except KeyError:
                    new_value = splits[1].strip()
                    logger.error('Parameter {} not found, default unchanged').format(current_key)
                splits[1] = new_value
                new_line = ','.join(splits) + '\n'
                edited_lines.append(new_line)

        # write the updated information back to the file (overwrite old file)
        with open(params_filepath, 'w') as pfile:
            for line in edited_lines:
                pfile.write(line)

    except FileNotFoundError:
        logger.error('Parameters file {} not found! Values not updated'.format(params_filepath))


def parse_params_file(params_file):
    """
    Parse a CIU_params.txt file for all parameters. Returns a params_dict that can be used to
    set_params on a Parameters object
    :param params_file: File to parse (.txt), headers = '#'
    :return: params_dict: Dictionary, key=param name, value=param value
    """
    param_dict = {}
    try:
        with open(params_file, 'r') as pfile:
            lines = list(pfile)
            for line in lines:
                # skip headers and blank lines
                if line.startswith('#') or line.startswith('\n'):
                    continue
                splits = line.rstrip('\n').split(',')
                value = splits[1].strip()

                # catch 'None' values and convert to None
                if value == 'None':
                    param_dict[splits[0].strip()] = None
                else:
                    # try parsing numbers
                    try:
                        try:
                            param_dict[splits[0].strip()] = int(value)
                        except ValueError:
                            param_dict[splits[0].strip()] = float(value)
                    except ValueError:
                        # string value - try parsing booleans or leave as a string
                        if value.lower() in ['true', 't', 'yes', 'y']:
                            param_dict[splits[0].strip()] = True
                        elif value.lower() in ['false', 'f', 'no', 'n']:
                            param_dict[splits[0].strip()] = False
                        elif value == '':
                            param_dict[splits[0].strip()] = None
                        else:
                            param_dict[splits[0].strip()] = splits[1].strip()

        return param_dict
    except FileNotFoundError:
        logger.error('params file not found!')


def parse_param_value(param_string):
    """
    Parsing hierarchy for strings being passed (or parsed) into parameters.
    :param param_string: stripped line being parsed
    :return:
    """
    if param_string == 'None':
        return None
    else:
        # try parsing numbers
        try:
            try:
                return int(param_string)
            except ValueError:
                return float(param_string)
        except ValueError:
            # string value - try parsing booleans or leave as a string
            if param_string.lower() in ['t', 'true', 'yes', 'y']:
                return True
            elif param_string.lower() in ['f', 'false', 'no', 'n']:
                return False
            else:
                return param_string.strip()


class ParamUI(tkinter.Toplevel):
    """
    Modular parameter editing UI popup class. Designed to take a list of parameters of arbitrary
    length and provide their names, current values, and definitions into a dialog for editing.
    """
    def __init__(self, section_name, params_obj, key_list, param_descripts_file):
        """
        Initialize a graphical menu with the parameters listed by key in the 'key_list' input.
        :param params_obj: Parameters object with parameter value information
        :type params_obj: Parameters
        :param key_list: list of keys (corresponding to keys in params.obj.params_dict and also the
        PARAM_DESCRIPTIONS dict) to display in the menu.
        :param param_descripts_file: full path to the parameter descriptions file ('CIU2_param_info.csv')
        """
        # load parameter descriptions from file
        self.par_names, self.par_descripts, self.par_reqs = parse_param_descriptions(param_descripts_file)

        tkinter.Toplevel.__init__(self)
        self.title(section_name)
        self.return_code = -2

        # return values = dictionary by parameter of the entry
        self.return_vals = {}
        self.keys = key_list
        self.entry_vars = {}

        # display a label (name), entry (value), and label (description) for each parameter in the key list
        row = 0
        labels_frame = ttk.Frame(self, relief='raised', padding='2 2 2 2')
        labels_frame.grid(column=0, row=0)
        for param_key in key_list:
            entry_var = tkinter.StringVar()
            # Get the current parameter value to display, or 'None' if the value is not set
            param_val = params_obj.params_dict[param_key]
            if param_val is not None:
                # change booleans to display as strings rather than 1 or 0
                if isinstance(param_val, bool):
                    if param_val:
                        entry_var.set('True')
                    elif not param_val:
                        entry_var.set('False')
                else:
                    entry_var.set(params_obj.params_dict[param_key])
            else:
                entry_var.set('None')

            # display the parameter name, value, and description
            ttk.Label(labels_frame, text=self.par_names[param_key]).grid(row=row, column=0, sticky='e')

            # display values the user can enter (int/float, strings) as an entry field
            param_type = self.par_reqs[param_key][0]
            if param_type == 'int' or param_type == 'float' or param_type == 'anystring':
                ttk.Entry(labels_frame, textvariable=entry_var, width=25).grid(row=row, column=1)
            else:
                # for fields where the user must choose, display a dropdown menu instead
                option_vals = [x for x in self.par_reqs[param_key][1]]
                # prevent the first option from not appearing in menu. Not sure why this duplication is necessary...
                option_vals.insert(0, entry_var.get())
                menu = ttk.OptionMenu(labels_frame, entry_var, *option_vals)
                menu.grid(row=row, column=1)
                menu['menu'].config(background='white')

            ttk.Label(labels_frame, text=self.par_descripts[param_key], wraplength=500).grid(row=row, column=2, sticky='w')

            self.entry_vars[param_key] = entry_var
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
        :return: void
        """
        fail_params = []
        for param_key in self.keys:
            if not self.check_param_value(param_key):
                fail_params.append(param_key)

        if len(fail_params) == 0:
            # no params failed, so close window and update parameter values
            self.return_code = 0
            self.on_close_window()
        else:
            # some parameters failed. Tell the user which ones and keep the window open
            param_string = 'The following parameter(s) have inappropriate values:\n'
            for param in fail_params:
                if self.par_reqs[param][0] == 'string' or self.par_reqs[param][0] == 'bool':
                    # print acceptable values list for string/bool
                    vals_string = ', '.join(self.par_reqs[param][1])
                    param_string += '{}: value must be one of ({})\n'.format(self.par_names[param], vals_string)
                else:
                    # print type and bounds for float/int
                    lower_bound = self.par_reqs[param][1][0]
                    upper_bound = self.par_reqs[param][1][1]
                    param_string += '{}:\n\t Value Type must be: {}\n\t Value must be within bounds: {} - {}\n'.format(self.par_names[param],
                                                                                                                       self.par_reqs[param][0],
                                                                                                                       lower_bound,
                                                                                                                       upper_bound)
            messagebox.showwarning(title='Parameter Error', message=param_string)
            return

    def check_param_value(self, param_key):
        """
        Check an individual parameter against its requirements
        :param param_key: key to parameter dictionary to be checked
        :return: True if the current value of the corresponding entry is valid, False if not
        """
        param_type = self.par_reqs[param_key][0]
        param_val_list = self.par_reqs[param_key][1]
        if param_type == 'int':
            # If the param is an int, the value must be within the values specified in the requirement tuple
            try:
                entered_val = int(self.entry_vars[param_key].get())
            except ValueError:
                return False
            return param_val_list[0] <= entered_val <= param_val_list[1]

        elif param_type == 'float':
            try:
                entered_val = float(self.entry_vars[param_key].get())
            except ValueError:
                if 'xlim' in param_key or 'ylim' in param_key:
                    if self.entry_vars[param_key].get() == '' or self.entry_vars[param_key].get().strip().lower() == 'none':
                        return True
                return False
            return param_val_list[0] <= entered_val <= param_val_list[1]

        elif param_type == 'string' or param_type == 'bool':
            # check whether the string or "boolean" (really a string) value is in the allowed list
            entered_val = self.entry_vars[param_key].get().strip().lower()
            # if entered_val == 'none':
            #     return 'none' in param_val_list
            # else:
            check_val_list = [x.strip().lower() for x in param_val_list]    # check against lower case/stripped
            return entered_val in check_val_list

        elif param_type == 'anystring':
            # Things like titles can be any string - no checking required
            return True

    def cancel_button_click(self):
        """
        Close the window and return without updating any parameter values
        :return: void
        """
        self.return_code = -1
        self.on_close_window()

    def refresh_values(self):
        """
        Update the values for all parameters
        :return:
        """
        for key in self.entry_vars.keys():
            parsed_val = parse_param_value(self.entry_vars[key].get())
            try:
                if parsed_val.strip().lower() == 'none' or parsed_val.strip() == '':
                    self.return_vals[key] = None
                else:
                    # not a 'none' string, set it as is
                    self.return_vals[key] = parsed_val
            except AttributeError:
                # not a string, return value as is
                self.return_vals[key] = parsed_val
        return self.return_vals

    def on_close_window(self):
        """
        Close the window
        :return: void
        """
        self.quit()
        self.destroy()


def test_param_ui():
    """
    for testing
    """
    myparams = Parameters()
    mydict = parse_params_file(r"C:\CIUSuite2\CIU_params.txt")
    myparams.set_params(mydict)
    key_list = ['smoothing_1_method', 'smoothing_2_window', 'smoothing_3_iterations']

    param_ui = ParamUI('test section', myparams, key_list, r"C:\CIUSuite2\CIU2_param_info.csv")
    test_top = tkinter.Toplevel()
    test_top.title('test top')
    param_ui.wait_window()
    print('finished wait')


# testing
if __name__ == '__main__':
    # myparams = Parameters()
    # mydict = parse_params_file(r"C:\Users\dpolasky\Desktop\CIU_params.txt")
    # myparams.set_params(mydict)
    # myparams.print_params_to_console()
    root = tkinter.Tk()
    root.withdraw()

    test_param_ui()

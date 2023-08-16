"""
Author: Carolina Rojas Ramirez
Date: Dec-2-2021
fromTWIMtoIMS
A script to take a _raw.csv file (TWIMExtract Output) and convert it to an inpur file for IMSCal19
"""
#Python Packages
import tkinter
from tkinter import filedialog
from tkinter import messagebox
import os
from subprocess import Popen
import pandas as pd
from IMSCal19Adapt_Parameter_Parser import parse_param
import subprocess
import Raw_Data_Import

def imscalfile_creator(outputname, protein_label, molecularweight, charge, driftimes):
    """
    Function to create a IMSCal19 input file
    :param protein_label: str
    :param molecularweight: int
    :param charge: int
    :param driftimes: ls
    :return: void
    """
    #Initialize variables
    out_str = ""
    proteinlabel_indx = 0

    #For each drift time create a line with all the info needed to calculate CCS
    for dt in driftimes:
        out_str += f"{protein_label}_{proteinlabel_indx}\t{molecularweight}\t{charge}\t0\t{dt}\n"
        proteinlabel_indx += 1

    #Save results
    output = open(f"{outputname}" + '.dat', 'w')
    output.write(out_str)
    output.close()


def twimxtrctfile_parser(twimfile):
    """
    Read template .txt_raw.csv file to extract drift times
    :param twimfile: (string) full system path to .txt_raw.csv file
    """

    #Initialized output list
    driftimes = []

    df = pd.read_csv(filepath_or_buffer=twimfile, sep=',')


    # print(df)
    # print(df.columns)
    # print(df.index)
    # print(df.index.stop)
    #df.loc[row_indexer,column_indexer]
    voltages = df.loc[1,df.columns[1:]]
    # print(voltages)

    driftime = df.loc[2:,df.columns[0]]
    for time in driftime:
        driftimes.append(time)
    # print(driftimes)

    return driftimes, df

def TWIMCalibrate_batfile_gatherer(inputfile_path,mainoutstr,reference_calfile, outputpath, calibrationparams):
    """
    Creates the .bat file to run IMSCal19
    :param mainoutstr: where to put each line to run in the final .bat file
    :param reference_calfile: path to the file with the ccs values to use for calibration
    :param outputfolder: path, where to save the file with the calculated ccs values
    :param calibrationparams: list, parameters needed for CCS calibration
    :return:
    """

    #Hardcoded for now
    lambda_val = 0.012
    accuracy_val = 2.0

    #Non hard-coded
    twimtube_length = calibrationparams[0]
    wave_vel = calibrationparams[1]
    wave_height = calibrationparams[2]
    pressure = calibrationparams[3]
    temperature = calibrationparams[4]


    mainoutstr += f'bin\TWaveCalibrate -ref "{reference_calfile}" -output "{outputpath}" -input "{inputfile_path}" -length {twimtube_length} -lambda {lambda_val} -velocity {wave_vel} -voltage {wave_height} -pressure {pressure} -temp {temperature} -accuracy {accuracy_val}\n'

    return mainoutstr

def ccs_replacing_dt(dict):
    """
    Funtion to take the resulting calculated ccs values and place them in the fingerprint _raw.csv file
    :param dict: a dictionary with paths to .dat files (keys) and corresponding data frames (values)
    :return: void
    """
    # print(dict)
    for path_file in dict:
        ccsval_ls = []
        # print(f"path_file = {path_file}")
        with open(path_file, 'r') as datfile:
            lines = list(datfile)

            caldataloc = lines.index('[CALIBRATED DATA]\n')
            # print(f"caldataloc = {caldataloc}")
            # print(f"indexed lines = {lines[caldataloc:]}")

            #Added two indeces as that is how the file is organized
            calibratedlines = lines[caldataloc+2:]

            for line in calibratedlines:
                    # ID,Mass,Z,Drift,CCS,CCS Std.Dev
                    line = line.strip("\n")
                    splits = line.split(',')
                    # print(splits)
                    # Index four because corresponds to ccs
                    ccs_val = splits[4]
                    ccsval_ls.append(ccs_val)

        #Find the drif times in the data frame
        ciufingerprint = dict[path_file]
        # print(f"ciufingerprint = {ciufingerprint}")
        dtimes = ciufingerprint.loc[2:, ciufingerprint.columns[0]]

        #Make sure the list of ccs values and drif times at least match in lenght
        if len(ccsval_ls) == len(dtimes):
            #Create a dictironayr of driftimes as keys and ccs as values for replace function
            replacement_dict = {}
            indexnum = 0
            for dtime in dtimes:
                replacement_dict[dtime] = ccsval_ls[indexnum]
                indexnum += 1

            outputfingerprint = ciufingerprint.replace(replacement_dict)

            ccs_fingerprtin_outname = path_file.strip("_output.dat") + "-ccs_raw.csv"

            print(f"ccs_fingerprtin_outname = {ccs_fingerprtin_outname}")

            outputfingerprint.to_csv(ccs_fingerprtin_outname, index=False)

        else:
            print("The calculated ccs values list and driftime lists don't match")
            print(ccsval_ls)
            # print(len(ccsval_ls))
            print(dtimes)


def twim_to_ccs_main():
    """
    Function to run the converter from dt to ccs, TWIMExtract file to CCS calibrated fingeprint
    :param batch: bool, do batch mode
    :return: path
    """
    print("TWIMExtracttoIMSCal")

    batfile_str = ""
    dt_to_ccs_replacedict = {}

    # Which is the batch file > _raw.csv to ccs
    batch_file = filedialog.askopenfilename(title='Batch Hits Files', filetypes=[('CSV File', '.csv')])

    # Read file
    with open(batch_file, 'r') as batch:
        lines = list(batch)
        for line in lines:

            if line.startswith("#"):
                continue

            else:
                line = line.strip("\n")
                splits = line.split(',')
                # for each lin in the batch file extract the needed info
                # print(splits)
                twimfilpath = splits[0]
                proteinlabel = splits[1]
                protein_mass = splits[2]
                chargestate = splits[3]
                outputlocation = splits[4]

                # print(f"{outputlocation}")

                mod_outputlocation = outputlocation.replace("\\", "/")

                # Extrac the drift times
                dtimes, fingerprintinfo = twimxtrctfile_parser(twimfilpath)

                # Create filename
                prefilename = twimfilpath.split('\\')[-1]
                filename = f"{prefilename.split('.')[0]}_z{chargestate}"
                print(filename)

                # Create IMSCal19 input
                os.chdir(mod_outputlocation)
                imscalfile_creator(filename, proteinlabel, protein_mass, chargestate, dtimes)

                # Gather infor for .bat file
                calfile_path = splits[5]
                imscal_ouput_location = splits[6]
                params_forcalibration_calculations = splits[7:]
                ims19_input_file = mod_outputlocation + "\\" + filename + '.dat'
                ims19_output_file = imscal_ouput_location + "\\" + filename + '_output.dat'
                dt_to_ccs_replacedict[ims19_output_file] = fingerprintinfo

                print(f" ims19_input_file = {ims19_input_file}")

                batfile_str = TWIMCalibrate_batfile_gatherer(ims19_input_file, batfile_str, calfile_path,
                                                             ims19_output_file, params_forcalibration_calculations)
                # batfile_str += "pause"



    # TODO: Include IMSCal19 in CIUSuite2
    # ims19loc = filedialog.askdirectory(title='Where is IMSCal19')
    # os.chdir(ims19loc)

    ims19loc = "C:\IMSCal19"

    os.chdir(ims19loc)

    # Creating .bat file
    # print(batfile_str)
    output = open("TWIMCalibrate" + '.bat', 'w')
    output.write(batfile_str)
    output.close()

    p = Popen("TWIMCalibrate.bat")
    stdout, stderr = p.communicate()
    #
    # print(f"stdout = {stdout}")
    # print(f"stderr = {stderr}")

    ccs_replacing_dt(dt_to_ccs_replacedict)

def twimextraction_forCIUSuitetwo(dict_of_params, ccs_mode = False):
    """
    Function to run the converter from dt to ccs (.raw to ccs cal fingerprint)
    :param dict_of_params: dict, a number as key (representing a fingerprint), parameter object as value
    :param ccs_mode: bool, if Ture CCS calibration with IMSCal19 will be done
    :return: path
    """
    print("Inputting raw into TWIMExtract")

    batfile_str = ""
    dt_to_ccs_replacedict = {}

    #So TWIMExtract can be accessed by all users
    # extractor_path = os.path.join(os.environ['ALLUSERSPROFILE'], 'CIUSuite2', 'TWIMExtract', 'jars', 'TWIMExtract.jar')
    extractor_path = os.path.join("C:\\", "TWIMExtract", "jars", "TWIMExtract.jar")
    for fingerprint in dict_of_params:
        # print(dict_of_params[fingerprint])
        raw_files = dict_of_params[fingerprint].rawfile
        range_params = dict_of_params[fingerprint].range_file_params

        # print(f"range_params = {range_params}")
        # print(raw_files)

        output_files_location = dict_of_params[fingerprint].IMSCal19_input

        #twimex_single_range(range_info, raw_files, save_dir, extractor_path)
        outputfilepath = Raw_Data_Import.twimex_single_range(range_params, raw_files, output_files_location, extractor_path)
        dict_of_params[fingerprint].TWIMExtractOutput = outputfilepath


    #Had to go over the dictionary because otherwise IMSCal19 would start reading files without TWIMExtract being able
    # to completely finish.
    for fingerprint in dict_of_params:
        if ccs_mode and dict_of_params[fingerprint].IMSCal19_cal:
            print("CCS Calibrating _raw.csv files")

            chargestate = dict_of_params[fingerprint].charge
            proteinlabel = dict_of_params[fingerprint].range_file_params[0]
            protein_mass = dict_of_params[fingerprint].MW
            twimexoutputpath = dict_of_params[fingerprint].TWIMExtractOutput
            print(twimexoutputpath[0])
            # Extract the drift times
            dtimes, fingerprintinfo = twimxtrctfile_parser(twimexoutputpath[0])
            # Create filename
            filename_path = twimexoutputpath[0].strip('.txt_raw.csv')

            print(filename_path)

            # Create IMSCal19 input
            imscalfile_creator(filename_path, proteinlabel, protein_mass, chargestate, dtimes)

            # Gather infor for .bat file
            calfile_path = dict_of_params[fingerprint].IMSCal19_cal
            imscal_ouput_location = dict_of_params[fingerprint].IMSCal19_output
            params_forcalibration_calculations = dict_of_params[fingerprint].IMSCal19_args
            ims19_input_file = filename_path + '.dat'
            ims19_outputname = filename_path + '_output.dat'
            ims19_output_file = imscal_ouput_location + '\\' + ims19_outputname.split('\\')[-1]


            dt_to_ccs_replacedict[ims19_output_file] = fingerprintinfo

            print(f"ims19_output_file = {ims19_output_file}")



            args = TWIMCalibrate_batfile_gatherer(ims19_input_file, batfile_str, calfile_path,
                                                         ims19_output_file, params_forcalibration_calculations)

            ims19loc = "C:\IMSCal19"

            os.chdir(ims19loc)


            completed_proc = subprocess.run(args)
            print(f'\n{completed_proc}')

    ccs_replacing_dt(dt_to_ccs_replacedict)

    #Returning value to close calibration window and ne able to use main GUI
    return True










if __name__ == '__main__':

    param_file = filedialog.askopenfilename(title='Param File', filetypes=[('CSV File', '.csv')])

    # wdir_spl = param_file.split('/')
    # wdir_ls = wdir_spl[:-1]
    # wdir = '/'.join(wdir_ls)
    # print(wdir)
    #
    # os.chdir(wdir)

    paramsdict = parse_param(param_file)

    print(paramsdict)

    twimextraction_forCIUSuitetwo(paramsdict,ccs_mode=True)



    # twim_to_ccs_main()


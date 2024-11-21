#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Carolina Rojas Ramirez
Date: Nov 4th, 2024
Script to extract Raw Data from Breuker .d files
With some code from tdfextractor by Michael Armbruster
"""

import timsdata
from timsdata import oneOverK0ToCCSforMz
from tkinter import filedialog
import os
import re
import pandas as pd
import numpy as np
import sys

# print(sys.path)
# sys.path.append(os.path.abspath('.'))
# print(sys.path)

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    dll_path = os.path.join(bundle_dir, 'timsdata.dll')
    os.add_dll_directory(os.path.dirname(dll_path))
else:
    os.add_dll_directory(os.getcwd())
    dll_path = os.path.join(os.getcwd(), 'timsdata.dll')

# Function to extract the mz and int dimension from a .d folder
def extractor_koint_fromframes(folder_path, mzmin, mzmax, voltageval, userecalstate = 0):
    """

    @param folder_path:
    @param mzmin:
    @param mzmax:
    @param voltageval:
    @param userecalstate:
    @return:
    """
    print(f"folder_path = {folder_path}")
    # print(f"timsdata.PressureCompensationStrategy.AnalyisGlobalPressureCompensation = {timsdata.PressureCompensationStrategy.AnalyisGlobalPressureCompensation}")
    # Create object to hold the raw data extracted (Attributes for no not editable by user)
    td = timsdata.TimsData(folder_path, use_recalibrated_state=userecalstate,
                  pressure_compensation_strategy=timsdata.PressureCompensationStrategy.AnalyisGlobalPressureCompensation)


    conn = td.conn
    # print(f"conn = {conn}")

    # Extract number of frames
    q = conn.execute("SELECT COUNT(*) FROM Frames")
    row = q.fetchone()
    total_frames = row[0]

    # List to gather all data
    all_data = []

    # Go over all frames
    for frame_id in range(1, total_frames + 1):
        q = conn.execute(f"SELECT NumScans FROM Frames WHERE Id={frame_id}")
        num_scans = q.fetchone()[0]

        # From frames to scans
        scans = td.readScans(frame_id, 0, num_scans)

        for scan_idx, (index_array, intensity_array) in enumerate(scans):
            mz_array = td.indexToMz(frame_id, index_array)

            # print(mz_array)

            # Just select a range of mz
            # TODO: Peak selection like for Agilent data
            filter_mask = (mz_array >= mzmin) & (mz_array <= mzmax)
            mz_filtered = mz_array[filter_mask]
            intensities_filtered = intensity_array[filter_mask]

            # If the data was within the filters and nonzero extract IM data
            if len(mz_filtered) > 0:
                ko_values = td.scanNumToOneOverK0(frame_id, np.array([scan_idx]))
                ko_values_filtered = np.full(len(mz_filtered), ko_values[0])

                data = pd.DataFrame({'ko': ko_values_filtered, f'{voltageval}': intensities_filtered})
                all_data.append(data)
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        grouped = combined_data.groupby(['ko'])[f'{voltageval}'].sum().reset_index()
        return grouped
    else:
        print("No data to process.")


# Function to extract voltage used in .d folder
def extract_voltage_from_method_file(folder_path):
    # print(f"Checking folder: {folder_path}")
    method_folder = None
    for subdir in os.listdir(folder_path):
        if subdir.endswith(".m"):
            method_folder = os.path.join(folder_path, subdir)
            break

    if not method_folder:
        raise FileNotFoundError(f"No subfolder ending with '.m' found in the directory: {folder_path}")

    method_file = None
    for file in os.listdir(method_folder):
        if file.endswith(".method"):
            method_file = os.path.join(method_folder, file)
            break

    if not method_file:
        raise FileNotFoundError(f"No .method file found in the directory: {method_folder}")

    # print(f"Method file found: {method_file}")

    with open(method_file, 'r') as f:
        content = f.read()

    match = re.search(r'<para_double value="([\d.]+)" permname="IMS_TunnelVoltage_Delta_6"/>', content)
    if match:
        return round(float(match.group(1)), 1)
    else:
        print(f"IMS_TunnelVoltage_Delta_6 not found in the method file: {method_file}")
        return None

def BreukerExtractmain(ciudir, mz_min, mz_max, ccs_conversion = None):
    """

    @param ciudir:
    @param mz_min:
    @param mz_max:
    @param ccs_conversion:
    @return:
    """

    # Select folder that contains all the .d files of interest


    masterjoindf = pd.DataFrame()

    # What are the contents of the main folder
    drawdir = [f for f in os.listdir(ciudir)]

    # Make sure each item in the main folder are folders and end in .d
    for diritem in drawdir:
        diritempath = os.path.join(ciudir, diritem)
        if diritem.endswith(".d") and os.path.isdir(diritempath):
            print(f"Extracting data from {diritem}")

            # Extract voltage
            cv_val = extract_voltage_from_method_file(diritempath)
            print(f"Voltage = {cv_val}")

            # Extract IM and int vals
            koint_val = extractor_koint_fromframes(diritempath, mz_min, mz_max, cv_val)

            # print(koint_val)
            kointdftojoin = koint_val.set_index("ko", drop=True)

            masterjoindf = pd.concat([kointdftojoin, masterjoindf], axis=1)

    os.chdir(ciudir)

    # How to call the outputifle
    outputname = os.path.basename(ciudir)
    print(outputname)

    # Extract indeces
    columns_lsofstrs = masterjoindf.columns
    # print(columns_lsofstrs)
    # Turn values to float to be sorted
    columnvals = [float(x) for x in list(columns_lsofstrs)]
    columnvals.sort()
    # Turn sorted values back to strings otherwise they won't get matched in the dataframe
    str_sortedcol_vals = [str(x) for x in columnvals]
    masterjoindf = masterjoindf.reindex(str_sortedcol_vals, axis=1)



    # Must sort mobility values too (They are already float, so less steps required)
    rows = masterjoindf.index
    sortedrows = rows.sort_values()
    masterjoindf = masterjoindf.reindex(sortedrows, axis=0)

    # Rename Mobility Units
    if ccs_conversion:
        masterjoindf.index.names = ['CCS (A^2)']

        # Extract extra parameters for CCS Conversion
        charge = ccs_conversion[1]
        ionmz_value = ccs_conversion[0]

        # Reset index in data frame to perfomr the CCS Conversion only in this last data frame instead of back when in each .d file
        masterjoindf = masterjoindf.reset_index()

        masterjoindf['CCS (A^2)'] = masterjoindf['CCS (A^2)'].apply(
            lambda x: oneOverK0ToCCSforMz(x, charge, ionmz_value))

    else:
        masterjoindf.index.names = ['Mobility (1/K0)']

    # Replace Nan with zeroes
    masterjoindf = masterjoindf.fillna(0)

    # Name file based on mobility status
    if ccs_conversion:
        masterjoindf.to_csv(outputname + f"_mz{mz_min}-{mz_max}_ccscal" + "_raw.csv", index=False)
    else:
        masterjoindf.to_csv(outputname + f"_mz{mz_min}-{mz_max}" + "_raw.csv")


def batchfile_parser(batchfile_path):
    """

    @param batchfile_path:
    @return:
    """

    batchdataframe = pd.read_csv(batchfile_path)

    for row in batchdataframe.index:
        parentfolder_val = batchdataframe.loc[row, "Parent Folder"]
        mzmin_val = batchdataframe.loc[row, "mzmin"]
        mzmax_val = batchdataframe.loc[row, "mzmax"]
        charge_val = batchdataframe.loc[row, "Charge"]
        mz_val = batchdataframe.loc[row, "mz"]

        yield [parentfolder_val, mzmin_val, mzmax_val, charge_val, mz_val]



def main():
    """
    Main function to enable terminal usage
    @return: void
    """

    # BreukerExtractmain(4444, 4455)

    #Checking Python version
    majorversion = sys.version_info.major
    minorversion = sys.version_info.minor

    if majorversion < 3:
        print("Python version must be at least 3.10")
    elif majorversion == 3 and minorversion < 10:
        print("Python version must be at least 3.10")
    else:
        print("Python version is at least greater than or equal to 3.10")

        # Example adding a second attribute
        systemarguments = sys.argv

        minmz_arg = systemarguments[1]
        maxmz_arg = systemarguments[2]
        mz_arg = systemarguments[3]
        charge_arg = systemarguments[4]
        batchvalue = systemarguments[5]

        print(systemarguments)


        if batchvalue.lower() == "false":
            print("Running Single System Folder Mode")
            fingerprintdir = filedialog.askdirectory(title="Choose Folder with .d files to make a fingerprint CIU")

            try:
                mz_arg = float(mz_arg)
                charge_arg = int(charge_arg)

                if mz_arg == 0.0 or charge_arg == 0:
                    BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg))
                else:
                    print("CCS Conversion will be done")
                    BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg), [mz_arg, charge_arg])
            except ValueError:
                BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg))

        elif batchvalue.lower() == "true":
            print("Running Batch Mode")

            breukerExtrac_batchfile = filedialog.askopenfilename(title = "Open Breuker batch mode file", filetypes=[('CSV File', '.csv')])

            paramls = list(batchfile_parser(breukerExtrac_batchfile))

            for analysis in paramls:

                # Because paramls is a generator output...the items are list sof numpt data types (for numerical values only)
                # below I use .item() to extract

                fingerprintdir = analysis[0]
                print(fingerprintdir)
                minmz_arg = analysis[1].item()
                maxmz_arg = analysis[2].item()
                charge_arg = analysis[3].item()
                mz_arg = analysis[4].item()


                try:
                    mz_arg = float(mz_arg)
                    charge_arg = int(charge_arg)

                    if mz_arg == 0.0 or charge_arg == 0:
                        BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg))
                    else:
                        print("CCS Conversion will be done")
                        BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg), [mz_arg, charge_arg])
                except ValueError:
                    BreukerExtractmain(fingerprintdir, float(minmz_arg), float(maxmz_arg))



        else:
            print("Invalid Batch Mode bool")





if __name__ == '__main__':

    #Test with no terminal
    # fingerprintdir = filedialog.askdirectory(title="Choose Folder with .d files to make a fingerprint CIU")
    # BreukerExtractmain(fingerprintdir, 4444, 4455, [16,4450])

    main()


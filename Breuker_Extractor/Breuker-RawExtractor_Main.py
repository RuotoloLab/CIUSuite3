"""
Author: Carolina Rojas Ramirez
Date: Nov 4th, 2024
Script to extract Raw Data from Breuker .d files
With some code from tdfextractor by Michael Armbruster
"""

from timsdata import *
from tkinter import filedialog
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Function to extract the mz and int dimension from a .d folder
def extractor_koint_fromframes(folder_path, mzmin, mzmax, voltageval):

    # Create object to hold the raw data extracted (Attributes for no not editable by user)
    td = TimsData(folder_path, use_recalibrated_state=False,
                  pressure_compensation_strategy=PressureCompensationStrategy.AnalyisGlobalPressureCompensation)
    conn = td.conn

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


def main(mz_min, mz_max, mode):
    """
    Function to run the Breuker Extractor
    @param mode: bool, preparing when all voltages are in one .d file
    @return: void
    """


    if mode == "Batch":
        pass
        # For a mode where several folder need to be read or when a .d file contains all voltages
    else:
        # Select folder that contains all the .d files of interest
        ciudic = filedialog.askdirectory(title="Choose Folder with .d files to make a fingerprint CIU")
        masterjoindf = pd.DataFrame()

        # What are the contents of the main folder
        drawdir = [f for f in os.listdir(ciudic)]


        # Make sure each item in the main folder are folders and end in .d
        for diritem in drawdir:
            diritempath = os.path.join(ciudic, diritem)
            if diritem.endswith(".d") and os.path.isdir(diritempath ):

                print(f"Extracting data from {diritem}")

                # Extract voltage
                cv_val = extract_voltage_from_method_file(diritempath)
                print(f"Voltage = {cv_val}")

                # Extract IM and int vals
                koint_val = extractor_koint_fromframes(diritempath, mz_min, mz_max, cv_val)
                # print(koint_val)
                kointdftojoin = koint_val.set_index("ko", drop=True)
                masterjoindf = pd.concat([kointdftojoin, masterjoindf], axis=1)

        os.chdir(ciudic)

        outputname = os.path.basename(ciudic)
        print(outputname)

        # print(masterjoindf)
        masterjoindf.to_csv(outputname + "_raw.csv")




if __name__ == '__main__':

    main(0, 10000, mode = False)

"""
Author: Chae Kyung Jeon
DTIM_CCS_calibration.py is a script to calibrate Agilent 6560C DTIM-Q-ToF using bovin serum albumin CIU data
as a calibrant. The user must first extract and determine drift time of each feature to use this function
in CIUSuite 3. Users must input the charge state of the BSA CIU fingerprint being used for calibration
along with drift time values of each feature. This script calibrates based on Mason-Schamp equation.
Once calibration is initiated, beta and tfix values will be calculated and given to the user
to input the values when converting drift time to CCS in Agilent Extractor.
"""
import os
import numpy as np
import scipy
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog

# set up Tkinter
root = tk.Tk()
root.withdraw()

# Parameters for Calibration
n2_mass  = 28.0134
bsa_mass = 66430
charge_state = [16, 17, 18]
ccs_byfeature = [[4513.76, 5378.43, 5759.9, 6166.805, 6446.55], [4633.75, 6160.43, 6309.04, 6687.33], [4734.665, 6251.015, 6508.505, 6909.05]]
ccs_byfeature_z16 = [4513.76, 5378.43, 5759.9, 6166.805, 6446.55]
# drift_time = [[36.09, 43.07, 46.145, 49.425, 51.63], [34.86, 46.35, 47.58, 50.45], [33.63, 44.505, 46.35, 49.22]]
# drift_time_z16 = [36.09, 43.07, 46.145, 49.425, 51.63]

# calibrate = LinearRegression()
#
# calibration_model_z16 = calibrate.fit(, ccs_byfeature_z16)
# calibration_model_z17 = calibrate.fit(drift_time[2], ccs_byfeature[2])
# calibration_model_z18 = calibrate.fit(drift_time[3], ccs_byfeature[3])

def perform_single_z_calibration(bsa_zval, dt1, dt2, dt3, dt4, dt5):
# bsa_z = input('BSA charge state: ')
#ask for calibrant drift time
    drift_time = []
    drift_time.append(dt1)
    drift_time.append(dt2)
    drift_time.append(dt3)
    drift_time.append(dt4)
    drift_time.append(dt5)


    if int(bsa_zval) == 16:
        df = pd.DataFrame({'dt': drift_time,
                          'CCS': ccs_byfeature[0]})
        dt_mean = np.mean(drift_time)
        ccs_mean = np.mean(ccs_byfeature[0])

        df['xycov'] = (df['dt'] - dt_mean) * (df['CCS'] - ccs_mean)
        df['xvar'] = (df['dt'] - dt_mean)**2

        coeff = df['xycov'].sum() / df['xvar'].sum()
        tfix = ccs_mean - (coeff * dt_mean)

        beta = (coeff/(16))/ np.sqrt(bsa_mass/(bsa_mass + n2_mass))

        print(coeff)
        print(tfix)
        print(beta)

    if int(bsa_zval) == 17:
        df = pd.DataFrame({'dt': drift_time,
                          'CCS': ccs_byfeature[1]})
        dt_mean = np.mean(drift_time)
        ccs_mean = np.mean(ccs_byfeature[1])

        df['xycov'] = (df['dt'] - dt_mean) * (df['CCS'] - ccs_mean)
        df['xvar'] = (df['dt'] - dt_mean)**2

        coeff = df['xycov'].sum() / df['xvar'].sum()
        tfix = ccs_mean - (coeff * dt_mean)

        beta = (coeff/(17))/ np.sqrt(bsa_mass/(bsa_mass + n2_mass))

        print(coeff)
        print(tfix)
        print(beta)

    if int(bsa_zval) == 18:
        df = pd.DataFrame({'dt': drift_time,
                          'CCS': ccs_byfeature[2]})
        dt_mean = np.mean(drift_time)
        ccs_mean = np.mean(ccs_byfeature[2])

        df['xycov'] = (df['dt'] - dt_mean) * (df['CCS'] - ccs_mean)
        df['xvar'] = (df['dt'] - dt_mean)**2

        coeff = df['xycov'].sum() / df['xvar'].sum()
        tfix = ccs_mean - (coeff * dt_mean)

        beta = (coeff/(18))/ np.sqrt(bsa_mass/(bsa_mass + n2_mass))

        print(coeff)
        print(tfix)
        print(beta)

    return tfix, beta



def calibrant_info_input():
    bsa_z,dt1_input, dt2_input, dt3_input, dt4_input, dt5_input = tk.IntVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar()
    popup = tk.Toplevel()
    popup.title("DTIM CCS Calibration")

    tk.Label(popup, text="BSA charge state for single charge state calibration: \n"
                         "Enter the BSA charge state obtained experimentally (interger only)",
             justify="left", wraplength=500)\
                            .grid(row=0, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=bsa_z).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(popup, text="BSA CIU feature 1 drift time:",
             justify="left", wraplength=500)\
                            .grid(row=1, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=dt1_input).grid(row=1, column=1, padx=5, pady=5)

    tk.Label(popup, text="BSA CIU feature 2 drift time:",
             justify="left", wraplength=500)\
                            .grid(row=2, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=dt2_input).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(popup, text="BSA CIU feature 3 drift time:",
             justify="left", wraplength=500)\
                            .grid(row=3, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=dt3_input).grid(row=3, column=1, padx=5, pady=5)

    tk.Label(popup, text="BSA CIU feature 4 drift time:",
             justify="left", wraplength=500)\
                            .grid(row=4, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=dt4_input).grid(row=4, column=1, padx=5, pady=5)

    tk.Label(popup, text="BSA CIU feature 5 drift time:",
             justify="left", wraplength=500)\
                            .grid(row=5, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=dt5_input).grid(row=5, column=1, padx=5, pady=5)


    submit_button = tk.Button(popup, text="Submit", command=popup.destroy)
    submit_button.grid(row=6, column=1, padx=5, pady=5, sticky="e")
    popup.bind("<Return>", lambda event: submit_button.invoke())

    popup.grab_set()
    popup.wait_window()

    return bsa_z.get(), dt1_input.get(), dt2_input.get(), dt3_input.get(), dt4_input.get(), dt5_input.get()


def DTIM_CCS_calibration_main():

    bsa_z, dt1_input, dt2_input, dt3_input, dt4_input, dt5_input = calibrant_info_input()

    tfix, beta = perform_single_z_calibration(bsa_z, dt1_input, dt2_input, dt3_input, dt4_input, dt5_input)

    new_popup = tk.Toplevel()
    new_popup.title("Beta and tfix")
    tk.Label(new_popup, text="Beta: " + str(beta),
             justify="center", wraplength=100)\
                            .grid(row=0, column=0, padx=5, pady=5, sticky='w')
    tk.Label(new_popup, text="tfix: " + str(tfix),
             justify="center", wraplength=100)\
                            .grid(row=1, column=0, padx=5, pady=5, sticky='w')

    new_popup.grab_set()
    new_popup.wait_window()










if __name__ == '__main__':

    DTIM_CCS_calibration_main()
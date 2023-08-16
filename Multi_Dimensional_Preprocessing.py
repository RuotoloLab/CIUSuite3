import sys
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
import numpy as np
import csv
import scipy


def read_in_csv():
    root = tk.Tk()
    root.withdraw()
    csvfiles = ()
    csv_rawdata = {}
    csv_cv = {}
    csv_dt = {}

    num_classes = simpledialog.askinteger('Folder', 'How many folder(s) is/are the input .csv placed into?')
    if num_classes is None:
        # user hit cancel - return
        logger.info('classification canceled (at number of classes selection)')


    for index in range(0, num_classes):
        # Read in the .CIU files and labels for each class, canceling the process if the user hits 'cancel'
        files = filedialog.askopenfilenames(filetypes=[('csv', '.csv')])
        #print(type(files))
        csvfiles += files
        #print(csvfiles)

    if len(csvfiles) == 0:
        logger.info('multi-dimensional preprocessing canceled'.format(index + 1))

    if len(csvfiles) < 2:
        messagebox.showerror('Not Enough Files', 'At least 2 replicates are required for multi-dimensional preprocessing')

    #output_dir = filedialog.askdirectory(title='Choose Output Folder')
    for file in csvfiles:
        csv_rawdata[file] = np.genfromtxt(os.path.join(file), delimiter=',')
    #print(csv_rawdata)

    for file in csvfiles:
        csv_cv[file] = csv_rawdata[file][0, 1:]
        csv_dt[file] = csv_rawdata[file][1:, 0]
    #print("csv_cv: ", csv_cv)


    return csv_rawdata, csv_cv, csv_dt, csvfiles


def normalize_csv_data(csv_files, csv_rawdata, csv_cv, csv_dt):
    norm_csv_dt = {}
    for file in csv_files:
        norm_csv_dt[file] = (csv_dt[file]-np.min(csv_dt[file]))/(np.max(csv_dt[file])-np.min(csv_dt[file]))

    return norm_csv_dt

def rearrange_csv(raw_data, csv_cv, csv_dt, csvfiles):
    new_csv_cv = {}
    new_csv_rawdata = {}

    for file in csvfiles:
        new_csv_rawdata[file] = raw_data[file][:, 0]
    #print("new_csv_rawdata: ", new_csv_rawdata)
    for file in csvfiles:
        #print("new_csv_rawdata[file]: ", new_csv_rawdata[file])
        for a, dt in enumerate(new_csv_rawdata[file]):
            if a > 0:
                new_csv_rawdata[file][a] = csv_dt[file][a-1]
    for file in csvfiles:
        new_csv_rawdata[file].shape = (201, 1)

    for f1 in csvfiles:
        for i, cv in enumerate(csv_cv[f1]):
            flag = 0
            for f2 in csvfiles:
                if cv not in csv_cv[f2]:
                    flag = 1
            if flag == 0:
                #print("new_csv_cv: ", cv)
                new_csv_cv[f1] = []
                new_csv_cv[f1].append(cv)
                single_col_intensity = raw_data[f1][:, i + 1]
                single_col_intensity.shape = (201, 1)

                new_csv_rawdata[f1] = np.hstack((new_csv_rawdata[f1], single_col_intensity))

    return new_csv_cv, new_csv_rawdata


def write_new_csv(new_csv_rawdata):
    for file in new_csv_rawdata:
        print(file)
        with open(os.path.join(file[0:-12] + "_multidimensional_raw.csv"), "w") as writefile:
            for item in new_csv_rawdata[file]:
                for i in range(len(item)):
                    if i == 0:
                        writefile.write(str(item[i]))
                    else:
                        writefile.write(',' + str(item[i]))
                writefile.write('\n')
            writefile.close()
"""
def multi_dimensional_preprocessing_main():
    # outdir = r"C:\Desktop\Senior\Research\CIUSuite\peakpicking\multi-d-processing"
    csv_rawdata, csv_cv, csv_dt, csv_files = read_in_csv()
    # print("csv_rawdata: ", csv_rawdata)
    # print("csv_cv:", csv_cv)
    # print("csv_dt: ", csv_dt)
    # print("csv_files: ", csv_files)
    norm_csv_dt = normalize_csv_data(csv_files, csv_rawdata, csv_cv, csv_dt)
    new_csv_cv, new_csv_rawdata = rearrange_csv(csv_rawdata, csv_cv, norm_csv_dt, csv_files)
    write_new_csv(new_csv_rawdata)
"""

if __name__ == '__main__':
    #multi_dimensional_preprocessing_main()
    # outdir = r"C:\Desktop\Senior\Research\CIUSuite\peakpicking\multi-d-processing"
    csv_rawdata, csv_cv, csv_dt, csv_files = read_in_csv()
    # print("csv_rawdata: ", csv_rawdata)
    # print("csv_cv:", csv_cv)
    # print("csv_dt: ", csv_dt)
    # print("csv_files: ", csv_files)
    norm_csv_dt = normalize_csv_data(csv_files, csv_rawdata, csv_cv, csv_dt)
    new_csv_cv, new_csv_rawdata = rearrange_csv(csv_rawdata, csv_cv, norm_csv_dt, csv_files)
    write_new_csv(new_csv_rawdata)
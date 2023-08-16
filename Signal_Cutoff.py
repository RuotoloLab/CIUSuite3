"""
Author: Carolina Rojas Ramirez
Date: 11/10/2020
CIUThreshold - cut noise signals from txt_raw.csv file
"""

import tkinter
from tkinter import filedialog
import pandas as pd
from tkinter import simpledialog

# set up Tkinter
root = tkinter.Tk()
root.withdraw()

def thresholder_TWIMExtract(exp_file, thresholdval, outdir):
    """
    :param exp_file: A .csv file produced by TWIM_Extract
    :param thresholdval: Decimal value indicating threshold cutoff (0.2 = 20% intensity)
    :return: A new .csv file with signals below threshold gone
    """
    print(f"Input = {exp_file}")
    df = pd.read_csv(filepath_or_buffer=exp_file, sep=',', header=1)


    # print(df)
    # column names
    columns = df.keys()
    trapCVrows = df.loc[0]
    print(f"columns =  {columns}")
    print(f"trapCVrows =  {trapCVrows}")

    for col in columns:

        if col.startswith("#"):
            continue
        else:
            # print(col)

            # Whole column which includes trap CV values
            # print(df[col])

            # only driftime rows no Trap CV row
            column_noCV = df[col][1:]

            # print(column_noCV)
            maxcol = column_noCV.max()
            # print(f"Max in column = {maxcol}")

            # normalize values
            column_noCV = column_noCV / maxcol
            # for intval in column_noCV:
            #     print(intval)

            #Setting intensity values to zero if they are below threshold
            df[f"{col}"] = column_noCV.apply(lambda x: 0 if x <= thresholdval else x)

            # for intval in column_noCV:
            #     print(intval)

    #Set outputpath
    expfilesplits = exp_file.split('/')
    outfile = expfilesplits[-1]
    outfile = outfile.replace(".txt_raw.csv", f"_cutoff{thresholdval}.txt_raw.csv")
    outputpath = outdir + '/' + outfile
    print(f"output = {outputpath}")

    #adding back the trap cvs
    df.loc[0] = trapCVrows

    #Convert data frame to csv
    df.to_csv(outputpath, index=False)


def thresholder(exp_file, thresholdval, outdir):
    """
    :param exp_file: A .csv file produced by MIDAC_Extractor
    :param thresholdval: Decimal value indicating threshold cutoff (0.2 = 20% intensity)
    :return: A new .csv file with signals below threshold gone and corrected voltages
    """
    print(f"Input = {exp_file}")
    df = pd.read_csv(filepath_or_buffer=exp_file, sep=',')


    # print(df)
    # column names
    columns = df.keys()

    # print(f"columns =  {columns}")

    new_header = {}

    for voltage in columns[1:]:
        # print(f"voltage = {voltage}")
        # print(f"round voltage = {round(float(voltage))}")
        # print(f"voltage%10 = {float(voltage)/10}")       # print(f"round voltage%10 = {round(float(voltage) / 10)}")
        corrected_voltage = (float(voltage) / 10)*10
        corrected_voltage = round(corrected_voltage)
        # print(f"(round voltage%10)*10 = {corrected_voltage}")
        new_header[voltage] = str(corrected_voltage)

    print(new_header)
    print(new_header.values())
    new_col_names = new_header.values()
    # #missing header
    # df = df[1:]

    # #adding new header
    df.rename(columns=new_header, inplace=True)

    #Thresholder
    for col in new_col_names:
        print(df[col][1:])

    for col in new_col_names:

        if col.startswith("#"):
            continue
        else:
            # print(col)

            # Whole column which includes trap CV values
            # print(df[col])

            # only driftime rows no Trap CV row
            column_noCV = df[col]

            # print(column_noCV)
            maxcol = column_noCV.max()
            # print(f"Max in column = {maxcol}")

            # normalize values
            column_noCV = column_noCV / maxcol
            # for intval in column_noCV:
            #     print(intval)

            # Setting intensity values to zero if they are below threshold
            df[f"{col}"] = column_noCV.apply(lambda x: 0 if x <= thresholdval else x)

            # for intval in column_noCV:
            #     print(intval)

    #Set outputpath
    expfilesplits = exp_file.split('/')
    outfile = expfilesplits[-1]
    outfile = outfile.replace(".csv", f"_cutoff{thresholdval}_raw.csv")
    outputpath = outdir + '/' + outfile
    print(f"output = {outputpath}")

    #Convert data frame to csv
    df.to_csv(outputpath, index=False)

def signal_cutoff_main():
    """

    """
    txtraw_files = filedialog.askopenfilenames(title='Choose _raw.csv vendor specific Files',
                                               filetypes=[('TWIM_Extract files', '.txt_raw.csv'),
                                                          ('MIDAC_Extractor files', '.csv')])

    # Where to save results
    outputdir = filedialog.askdirectory(title='Choose Output Folder')

    #
    # Thresholder

    threshold = simpledialog.askfloat('Threshold to use',
                                      'Enter the threshold to use in decimals (e.g. 0.2 for cutting off signals below 20% of maxium intensity)')

    for file_index, file in enumerate(txtraw_files):
        print('Processing file {} of {}...'.format(file_index + 1, len(txtraw_files)))

        if file.endswith(".txt_raw.csv"):
            thresholder_TWIMExtract(file, threshold, outputdir)
        else:
            print("AGILENT")
            thresholder(file, threshold, outputdir)


if __name__ == '__main__':

    signal_cutoff_main()


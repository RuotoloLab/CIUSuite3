import csv
import tkinter
from tkinter import filedialog
from tkinter import simpledialog
import pandas as pd
import numpy as np

# set up Tkinter
root = tkinter.Tk()
root.withdraw()

def CV_modification(exp_file, outdir):
    print(f"Input = {exp_file}")
    df = pd.read_csv(filepath_or_buffer=exp_file, sep=',', header=1)

    # column names
    columns = df.keys()
    trapCVrows = df.loc[0]
    trapCVrows = trapCVrows[1:].astype('int') - 10
    print(f"columns =  {columns}")
    print(f"trapCVrows =  {trapCVrows}")

    # Set outputpath
    expfilesplits = exp_file.split('/')
    outfile = expfilesplits[-1]
    outfile = outfile.replace(".csv", f"_SIU_CVcorrected_raw.csv")
    outputpath = outdir + '/' + outfile
    print(f"output = {outputpath}")

    #adding back the trap cvs
    df.loc[0] = trapCVrows

    #Convert data frame to csv
    df.to_csv(outputpath, index=False)


def CV_correction_main():
    txtraw_files = filedialog.askopenfilenames(title='Choose _raw.csv vendor specific Files',
                                               filetypes=[('TWIM_Extract files', '.txt_raw.csv')])

    # Where to save results
    outputdir = filedialog.askdirectory(title='Choose Output Folder')

    for file_index, file in enumerate(txtraw_files):
        print('Processing file {} of {}...'.format(file_index + 1, len(txtraw_files)))

        if file.endswith(".csv"):
            CV_modification(file, outputdir)


    # filename = filedialog.askopenfilename(title='Choose _raw.csv input file', filetypes=[("CSV Files", "*.csv")])
    # print('filename:', filename)
    # csvpath = r"C:/Desktop/Research/CIUSuite/CIUSuite3dev"

    # Define the input and output file paths
    #input_file = '2022_0204_BSA_CIU_4220mz_500WV_27WH_1.txt_raw.csv'
    # output_file = "output.csv"

    # Read the input file and extract the columns
#     cols = []
#     with open(exp_file, "r") as f:
#         reader = csv.reader(f)
#         #headers = next(reader)  # Extract the header row
#         for column in reader:
#             cols.append(column)
#     return cols, filename
#
#
#
# def cv_subtraction(columns):
#     indices = []
#     for i in range(len(columns)):
#         for j in range(len(columns[i])):
#             if i == 2 and j >= 1:
#                 print(columns[i][j])
#                 columns[i][j] = float(columns[i][j]) - 10
#
#                 if columns[i][j] < 0:
#                     print("column_neg =", i, j)
#                     indices.append(j)
#     print("indices:", indices)
#     for i in range(len(columns)):
#         for j in range(len(indices)):
#             del columns[i][indices[j]]
#     # Write the updated columns to the output file with the same format as the input file
#     return columns
#
# def write_csv(columns, outfile):
#     with open(outfile, "w", newline="") as f:
#         writer = csv.writer(f)
#         #writer.writerow(headers)  # Write the header row
#         writer.writerows(columns)
#
# def CV_correction_main():
#     columns, fpath = read_from_csv()
#     fname = fpath.split("/")[-1]
#     # fname_list = fname.split(".")
#     # fname_list[-2] = "SIU_raw"
#     outfile = fname.replace(".csv", f"_SIU_CVcorrected_raw.csv")
#     print(outfile)
#     columns_corrected = cv_subtraction(columns)
#     write_csv(columns_corrected, outfile)




if __name__ == '__main__':
    CV_correction_main()

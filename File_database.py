"""
Author: Carolina
Date: 02/14/2021
"""
from tkinter import filedialog
import os
import shutil

def open_rawcsv():

    wrkdir = filedialog.askdirectory(title='Choose Input Folder')
    os.chdir(wrkdir)

    files = os.listdir(wrkdir)
    files = [x for x in files if x.endswith('_raw.csv')]
    return files

def create_database(csvfiles):

    outstr = "#Old Filename,New FileName\n"
    for file in csvfiles:
        print(file)
        outstr += f"{file}\n"

    output = open("FileName_Directory.csv", 'w')
    output.write(outstr)
    output.close()

def renamingFiles_fromdatabase(copy=False):
    """

    """

    database_file = filedialog.askopenfilename(title='Database files', filetypes=[('CSV File', '.csv')])

    print(database_file)

    dirname = os.path.dirname(database_file)
    os.chdir(dirname)

    # print(os.path.dirname(database_file))
    # print(os.path.basename(database_file))

    with open(database_file, 'r') as batch:
        lines = list(batch)

        for line in lines:

            if line.startswith("#"):
                continue

            split_ln = line.split(",")
            # print(split_ln)

            old_name = split_ln[0]
            new_name = split_ln[1].strip("\n")

            print(f"{old_name}-->{new_name}")

            if copy:
                shutil.copyfile(old_name, f"{new_name}_raw.csv")
            else:
                os.rename(old_name, f"{new_name}_raw.csv")


if __name__ == '__main__':

    #Creating Databse
    # file_Paths = open_rawcsv()
    # create_database(file_Paths)

    #Read Database
    renamingFiles_fromdatabase(copy = True)


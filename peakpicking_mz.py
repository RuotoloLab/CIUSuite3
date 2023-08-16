import os
import numpy as np
import scipy
from scipy.signal import find_peaks, peak_prominences, savgol_filter, peak_widths
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import csv


def read_csvdata(dirpath, csvfile):
    ciu_data = np.genfromtxt(os.path.join(dirpath, csvfile), delimiter=',')
    # masscharge_value = ciu_data[7:, 0]
    # intensity_bymass = []
    # intensity_bydt = []
    # intensity = ciu_data[7:, 2:]
    # row = intensity.splitlines()
    # for items in row:
    #     row_sum = sum(items)
    #     intensity_bymass.append(row_sum)
    drifttime = ciu_data[0:, 0]
    all_intensity = ciu_data[0:, 1:]
    print(all_intensity)
    intensity_bymass = np.sum(all_intensity, axis=1)
    #print(intensity_bymass)



    return ciu_data, drifttime, intensity_bymass

def smooth_data(intensity_bymass, window_size=1001, smth_iter=5):
    smoothed_intensity = savgol_filter(intensity_bymass, window_size, smth_iter) #window size of 100, polynomial order 5

    return smoothed_intensity


def peak_picking(dirpath, drifttime, intensity, input_file, prominence=3000):
    peaks, properties = find_peaks(intensity, prominence)
    prominences = peak_prominences(intensity, peaks)
    # print(properties)
    width = peak_widths(intensity, peaks)

    filtered_peak_indices = []
    np_width = np.array(width)
    for ind, item in enumerate(np_width[0]):
        if item >= 100:
            # rounded_width = int(round(item))
            filtered_peak_indices.append(ind)

    filtered_peak_indices = np.array(filtered_peak_indices)

    filtered_peaks = []
    for item in filtered_peak_indices:
        filtered_peaks.append(peaks[item])

    filtered_peaks = np.array(filtered_peaks)
    #
    #
    # # #print(prominences)
    # print(peaks)
    # print(filtered_peaks)
    plt.plot(intensity)
    plt.plot(filtered_peaks, intensity[filtered_peaks], "x")
    #pdf
    plt.savefig(os.path.join(dirpath, "".join(input_file.split(".")[0:-1]) + "_plot.pdf"))
    plt.close()

    # with open(os.path.join(dirpath, "detected_peaks.csv"), 'w') as peak_summary:
    #     for item in zip(peaks, width):
    #         peak_summary.write('{}, {}\n'.format(item[0], item[1]))
    #         peak_summary.close()

    return peaks, prominences, width, filtered_peaks

def write_csv_output(dirpath, peaks, width, drifttime, input_file):
    np_width = np.asarray(width)
    half_width = []
    for x in np_width[0]:
        half_width.append(x/2)

    half_width = np.array(half_width)
    print(half_width.dtype)

    peak_mz = []
    for item in peaks:
        peak_mz.append(drifttime[item])
    peak_mz = np.array(peak_mz)
    print(peak_mz.dtype)

    low_peaks = [peak - halfwidth for peak, halfwidth in zip(peaks, half_width)]
    high_peaks = [peak + halfwidth for peak, halfwidth in zip(peaks, half_width)]
    low_peaks = np.array(low_peaks)
    high_peaks = np.array(high_peaks)
    print(low_peaks.dtype)

    low_mz = []
    high_mz = []
    #
    for low in low_peaks:
        ind = int(round(low))
        low_mz.append(drifttime[ind])
    for high in high_peaks:
        ind = int(round(high))
        high_mz.append(drifttime[ind])

    low_mz = np.array(low_mz)
    high_mz = np.array(high_mz)

    with open(os.path.join(dirpath, "".join(input_file.split(".")[0:-1]) + "_summary.csv"), 'w') as peak_summary:
        for item in zip(peak_mz, peaks, low_mz, high_mz, low_peaks, high_peaks):
            peak_summary.write('{}, {}, {}, {}, {}, {}\n'.format(item[0], item[1], item[2], item[3], item[4], item[5]))
        peak_summary.close()



    return half_width, peak_mz, low_peaks, high_peaks, low_mz, high_mz

def write_batch_extraction_input(dirpath, rawdatafile, low_mz, high_mz):
    # np_width = np.asarray(width)
    # half_width = []
    # for x in np_width[0]:
    #     half_width.append(x / 2)
    #
    # half_width = np.array(half_width)
    # print(half_width.dtype)
    #
    # peak_mz = []
    # for item in peaks:
    #     peak_mz.append(drifttime[item])
    # peak_mz = np.array(peak_mz)
    # print(peak_mz.dtype)
    #
    # low_peaks = [peak - halfwidth for peak, halfwidth in zip(peaks, half_width)]
    # high_peaks = [peak + halfwidth for peak, halfwidth in zip(peaks, half_width)]
    # low_peaks = np.array(low_peaks)
    # high_peaks = np.array(high_peaks)
    # print(low_peaks.dtype)
    #
    # low_mz = []
    # high_mz = []
    # #
    # for low in low_peaks:
    #     ind = int(round(low))
    #     low_mz.append(drifttime[ind])
    # for high in high_peaks:
    #     ind = int(round(high))
    #     high_mz.append(drifttime[ind])
    #
    # low_mz = np.array(low_mz)
    # high_mz = np.array(high_mz)
    #path = (dirpath.split("."))[0:-1]
    with open(os.path.join(dirpath, ''.join((rawdatafile.split("."))[0:-1]) + "_batch.csv"),'w') as batch_template:
        batch_template.write('Raw: ' + dirpath + rawdatafile + '\n')
        for item in zip(low_mz, high_mz):
            batch_template.write('{}, {}, , , , , , , 2\n'.format(item[0], item[1]))

        batch_template.close()

def read_csv_and_save_columns():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(title='Choose _autorange.csv input file', filetypes=[("CSV Files", "*.csv")])
    print('filename:', filename)
    input_file = (filename.split('/'))[-1]
    print('inputfile:', input_file)
    ciu_data = np.genfromtxt(filename, delimiter=',')
    drifttime = ciu_data[0:, 0]
    all_intensity = ciu_data[0:, 1:]
    print(all_intensity)
    intensity_bymass = np.sum(all_intensity, axis=1)
    """
    column1 = []
    column2 = []
    with open(filename, "r") as f:
        for line in f:
            cols = line.split()
            if isinstance(cols[0], int) or isinstance(cols[0], float):
                column1.append(cols[0])
                column2.append(cols[1])
    """
    return input_file, ciu_data, drifttime, intensity_bymass


def create_integer_input_popup():
    window, iteration, prominence, width = tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar()
    window.set(1001)
    iteration.set(5)
    prominence.set(3000)
    width.set(100)

    popup = tk.Toplevel()
    popup.title("Integer Input Popup")

    tk.Label(popup, text="Smoothing Window Size: ("
                         "Size of the filter for the applied smoothing method. Increasing this value can "
                         "improve performance in noisy data but can result in smearing if increased too much. "
                         "The input should be an odd number.)",
             justify="left", wraplength=500)\
                          .grid(row=0, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=window).grid(row=0, column=1, padx=5, pady=5)

    tk.Label(popup, text="Smoothing Iteration: (The number of times to sequentially apply smoothing. "
                         "Increasing this value will improve fitting of data with minimal smearing. "
                         "The input should be an odd number.)", justify="left",
             wraplength=500)\
        .grid(row=1, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=iteration).grid(row=1, column=1, padx=5, pady=5)

    tk.Label(popup, text="Peak Prominence: (The parameter that distinguishes the peak from the noise and other fluctuations in the data, "
                         "and determines the peak's significance relative to its neighboring peaks. Increasing this "
                         "value can result in fewer peaks being identified.)", justify="left", wraplength=500)\
        .grid(row=2, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=prominence).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(popup, text="Peak Width: (The width of a peak in samples at a relative distance to the peak’s height and "
                         "prominence. Increasing the peak width will broaden the range of data points considered "
                         "as part of the peak.)", justify="left", wraplength=500).grid(row=3, column=0, padx=5, pady=5, sticky="w")
    tk.Entry(popup, width=10, textvariable=width).grid(row=3, column=1, padx=5, pady=5)

    submit_button = tk.Button(popup, text="Submit", command=popup.destroy)
    submit_button.grid(row=4, column=1, padx=5, pady=5, sticky="e")
    popup.bind("<Return>", lambda event: submit_button.invoke())

    popup.grab_set()
    popup.wait_window()

    return window.get(), iteration.get(), prominence.get(), width.get()

def peakpicking_main():

    input_file, ciu_data, mz, intensity_bymass = read_csv_and_save_columns()
    # Example usage
    window, iteration, prominence, width = create_integer_input_popup()

    smoothed_intensity = smooth_data(intensity_bymass, window, iteration)
    output_dir = filedialog.askdirectory(title='Choose Output Folder')
    peaks, prominences, width, filtered_peaks = peak_picking(output_dir, mz, smoothed_intensity, input_file,prominence)
    peak_picking(output_dir, mz, smoothed_intensity, input_file, prominence)
    half_width, peak_mz, low_peaks, high_peaks, low_mz, high_mz = write_csv_output(output_dir, peaks, width, mz, input_file)
    write_batch_extraction_input(output_dir, input_file, low_mz, high_mz)

if __name__ == '__main__':

    peakpicking_main()

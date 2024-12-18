# -*- coding: utf-8 -*-
"""Test program using Python wrapper for timsdata.dll"""
import sys

if sys.version_info[0] >= 3:
    unicode = str
    
from timsdata import *
import numpy as np, matplotlib.pyplot as plt
import pandas as pd

analysis_dir = r"C:\Users\armbrusm\Documents\tdfExtract\miniset\20240404_fullMS_5uM_BSA_CIU130_000001.d"

#convert 1/K0 to CCS
ccs = oneOverK0ToCCSforMz(1.1846, 1, 946.7764)
print("CCS for 1/K0 1.1846, charge 1, mass 946.7764 : {0}".format(ccs))

td = TimsData(analysis_dir, use_recalibrated_state=False, pressure_compensation_strategy=PressureCompensationStrategy.AnalyisGlobalPressureCompensation)
conn = td.conn

# Get total frame count:
q = conn.execute("SELECT COUNT(*) FROM Frames")
row = q.fetchone()
N = row[0]
print("Analysis has {0} frames.".format(N))


# Get a projected mass spectrum:
frame_id = 30
q = conn.execute("SELECT NumScans FROM Frames WHERE Id={0}".format(frame_id))
num_scans = q.fetchone()[0]

numplotbins = 500;
min_mz = 0
max_mz = 3000
mzbins = np.linspace(min_mz, max_mz, numplotbins)
summed_intensities = np.zeros(numplotbins+1)

for scan in td.readScans(frame_id, 0, num_scans):
    index = np.array(scan[0], dtype=np.float64)
    mz = td.indexToMz(frame_id, index)
    if len(mz) > 0:
        plotbins = np.digitize(mz, mzbins)
        intens = scan[1]
        for i in range(0, len(intens)):
            summed_intensities[plotbins[i]] += intens[i]

# Get list of scanned mobilities
scan_number_axis = np.arange(num_scans, dtype=np.float64)

ook0_axis = td.scanNumToOneOverK0(frame_id, scan_number_axis)
df = pd.DataFrame(ook0_axis)

df.to_csv('b30_130VfalseGlobal.csv')

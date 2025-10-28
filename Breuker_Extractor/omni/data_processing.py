"""
data_processing.py  – run tims_ko_pull2 and post‑process its stdout CSV

Key additions
-------------
• `extraction_method` arg ('method' | 'filename')
• uses file_utils.voltage_from_filename when user picks filename mode
"""
import os
import sys
import contextlib
from io import StringIO
import pandas as pd

from tims_ko_pull2 import main as tims_main
from file_utils import (
    extract_voltage_from_method_file,
    voltage_from_filename
)

# DLL path bootstrap (unchanged from your previous version)
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    dll_path = os.path.join(bundle_dir, 'timsdata.dll')
    os.add_dll_directory(os.path.dirname(dll_path))
else:
    os.add_dll_directory(os.getcwd())
    dll_path = os.path.join(os.getcwd(), 'timsdata.dll')


# ───────────────────────────────────────────────────────────────
def process_folder(d_folder_path,
                   mzmin, mzmax,
                   extraction_method: str = "method",
                   voltage_param_key: str = "transfer",
                   voltage_polarity: str = "positive",
                   use_recalibrated_state=True,
                   pressure_compensation_strategy="AnalysisGlobalPressureCompensation"):
    """
    Run tims_ko_pull2, capture stdout CSV, rename intensity columns
    according to user‑selected voltage extraction method.

    Returns
    -------
    pandas.DataFrame | None
    """
    # 1) ensure we pass a .d folder to timsdata
    data_folder = os.path.dirname(d_folder_path) if d_folder_path.endswith(".m") else d_folder_path

    # 2) run helper CLI
    sys.argv = [
        'tims_ko_pull2.py', data_folder,
        '--mzmin', str(mzmin), '--mzmax', str(mzmax),
        '--use_recalibrated_state', str(use_recalibrated_state),
        '--pressure_compensation_strategy', pressure_compensation_strategy
    ]
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        tims_main()

    df = pd.read_csv(StringIO(buf.getvalue()))
    if 'ko' in df.columns:
        df.rename(columns={'ko': 'Mobility'}, inplace=True)

    intensity_cols = [c for c in df.columns if c.lower().startswith("intensity")]
    if not intensity_cols:
        return df

    # ── A) filename mode ─────────────────────────────────────────
    if extraction_method == "filename":
        v = voltage_from_filename(data_folder) or os.path.basename(data_folder)
        df.rename(columns={intensity_cols[0]: str(v)}, inplace=True)
        return df

    # ── B) method mode ───────────────────────────────────────────
    volts = extract_voltage_from_method_file(
        d_folder_path,
        param_key=voltage_param_key,
        polarity=voltage_polarity
    )
    print(f"Found voltages ({voltage_param_key},{voltage_polarity}): {volts}")

    if isinstance(volts, list):
        # align list length with columns
        if len(volts) != len(intensity_cols):
            print(f"WARNING: {len(volts)} voltages vs {len(intensity_cols)} columns → trunc/pad")
            if len(volts) > len(intensity_cols):
                volts = volts[:len(intensity_cols)]
            else:
                volts += [f"Seg_{i+1}" for i in range(len(intensity_cols)-len(volts))]
        rename = {old: str(v) for old, v in zip(intensity_cols, volts)}
        df.rename(columns=rename, inplace=True)
    else:
        df.rename(columns={intensity_cols[0]: str(volts)}, inplace=True)

    return df

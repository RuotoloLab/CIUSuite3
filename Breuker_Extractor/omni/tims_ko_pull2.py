import sys, os, argparse, sqlite3
import numpy as np
import pandas as pd
from timsdata import *

# ------------------------------------------------------------------------
# Helper: accumulate intensities scan‑by‑scan into a dict {mobility: total}
# ------------------------------------------------------------------------
def _accumulate_scan(td, frame_id, scans, idx_min, idx_max, mob_dict,
                     mzmin, mzmax, sum_mode, mobmin, mobmax):
    """
    td         : TimsData handle
    frame_id   : current frame
    scans      : list[(idx_array, inten_array)]
    idx_min/max: global index bounds for the desired m/z window
    mob_dict   : dict[float] -> running sum  (will be mutated)
    """
    for scan_idx, (idx_arr, inten_arr) in enumerate(scans):
        # Fast boolean slice in **index space**
        mask = (idx_arr >= idx_min) & (idx_arr <= idx_max)
        if not mask.any():
            continue                       # nothing in range – skip

        # Optional mobility‑window filter (sum‑mode only)
        ko_val = td.scanNumToOneOverK0(frame_id, np.array([scan_idx]))[0]
        if sum_mode and not (mobmin <= ko_val <= mobmax):
            continue

        intensity_sum = inten_arr[mask].sum()          # always a scalar

        # Accumulate
        if sum_mode:
            mob_dict[ko_val] = mob_dict.get(ko_val, 0.0) + intensity_sum
        else:
            # store array for later (concatenate once per frame)
            mob_dict[ko_val] = mob_dict.get(ko_val, 0.0) + intensity_sum

# ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Fast TIMS extractor")
    p.add_argument("input_folder")
    p.add_argument("--mzmin", type=float, required=True)
    p.add_argument("--mzmax", type=float, required=True)
    p.add_argument("--use_recalibrated_state", type=bool, default=True)
    p.add_argument("--pressure_compensation_strategy", default="AnalysisGlobalPressureCompensation")
    # unchanged CLI extras
    p.add_argument("--sum_mode", action='store_true')
    p.add_argument("--mobmin", type=float)
    p.add_argument("--mobmax", type=float)
    args = p.parse_args()

    if args.sum_mode and (args.mobmin is None or args.mobmax is None):
        sys.exit("In --sum_mode you must supply --mobmin and --mobmax")

    # --------------------------------------------------------------------
    # Open analysis
    # --------------------------------------------------------------------
    strat = {"No compensation": PressureCompensationStrategy.NoPressureCompensation,
             "Per-frame":       PressureCompensationStrategy.PerFramePressureCompensation,
             "Global":          PressureCompensationStrategy.AnalyisGlobalPressureCompensation}\
            .get(args.pressure_compensation_strategy, PressureCompensationStrategy.AnalyisGlobalPressureCompensation)

    td  = TimsData(args.input_folder,
                   use_recalibrated_state=args.use_recalibrated_state,
                   pressure_compensation_strategy=strat)
    conn = td.conn

    # Does this run contain segments?
    seg_rows = conn.execute(
        "SELECT FirstFrame, LastFrame FROM Segments ORDER BY FirstFrame"
    ).fetchall()
    segments = seg_rows if seg_rows else [(1, conn.execute("SELECT COUNT(*) FROM Frames").fetchone()[0])]

    # --------------------------------------------------------------------
    # Main extraction loop
    # --------------------------------------------------------------------
    dfs_per_segment = []
    for seg_idx, (first_f, last_f) in enumerate(segments):
        mob_dict = {}  # reset per segment

        for frame_id in range(first_f, last_f + 1):
            # ------------------------------------------------------------
            # precalc global index bounds for this frame once
            # ------------------------------------------------------------
            idx_min, idx_max = td.mzToIndex(frame_id, np.array([args.mzmin, args.mzmax]))
            idx_min, idx_max = int(idx_min), int(idx_max)

            scans = td.readScans(frame_id, 0,
                                 conn.execute("SELECT NumScans FROM Frames WHERE Id=?",
                                              (frame_id,)).fetchone()[0])
            _accumulate_scan(td, frame_id, scans,
                             idx_min, idx_max, mob_dict,
                             args.mzmin, args.mzmax,
                             args.sum_mode, args.mobmin, args.mobmax)

        # ----------------------------------------------------------------
        # Segment → DataFrame
        # ----------------------------------------------------------------
        if args.sum_mode:
            # one value per segment → single‑row DF later
            dfs_per_segment.append(sum(mob_dict.values()))
        else:
            # full profile
            ko_vals = np.fromiter(mob_dict.keys(),   dtype=np.float64)
            ints = np.fromiter(mob_dict.values(), dtype=np.float64)
            seg_df  = pd.DataFrame({'ko': ko_vals, 'intensity': ints})
            dfs_per_segment.append(seg_df)

    # --------------------------------------------------------------------
    # Output identical to original script
    # --------------------------------------------------------------------
    if args.sum_mode:
        # try to label columns from method file (unchanged logic)
        try:
            from file_utils import extract_voltage_from_method_file
            volts = extract_voltage_from_method_file(args.input_folder)
        except Exception:
            volts = None
        cols = list(map(str, volts)) if isinstance(volts, list) and len(volts) == len(dfs_per_segment) \
               else [f"Segment_{i+1}" for i in range(len(dfs_per_segment))]
        print(pd.DataFrame([dfs_per_segment], columns=cols).to_csv(index=False))
    else:
        # merge segment profiles
        out_df = dfs_per_segment[0]
        for i, seg_df in enumerate(dfs_per_segment[1:], 1):
            out_df = pd.merge(out_df, seg_df, on='ko', how='outer',
                              suffixes=('', f'_{i}'))
        out_df.sort_values('ko', inplace=True)
        out_df.fillna(0, inplace=True)
        print(out_df.to_csv(index=False))

# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()
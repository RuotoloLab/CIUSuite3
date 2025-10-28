# processing.py

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from tkinter import messagebox

from timsdata import oneOverK0ToCCSforMz
from data_processing import process_folder

# ───────────────────────────────────────────────────────────────
# Helper regex & utilities
# ───────────────────────────────────────────────────────────────
_VOLT_RE = re.compile(r"[_\-](\d+)[vV](?:_|$)")

def _voltage_from_name(path: str) -> str | None:
    m = _VOLT_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def _numeric_key(text: str) -> float:
    m = re.match(r"^-?\d+(?:\.\d+)?", text)
    return float(m.group(0)) if m else float("inf")


def _bin_mobility(df: pd.DataFrame, nbins: int) -> pd.DataFrame:
    """Sum‑bin the mobility axis into *nbins* evenly spaced bins."""
    if nbins < 1:
        return df.copy()

    edges = np.linspace(df["Mobility"].min(), df["Mobility"].max(), nbins + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    df = df.copy()
    df["_bin"] = pd.cut(
        df["Mobility"], bins=edges, labels=mids, include_lowest=True, ordered=True
    )

    icols = [c for c in df.columns if c not in ("Mobility", "_bin")]
    out = (
        df.groupby("_bin", observed=False)[icols]
        .sum()
        .reset_index()
        .rename(columns={"_bin": "Mobility"})
    )
    out["Mobility"] = out["Mobility"].astype(float)
    return out


# ───────────────────────────────────────────────────────────────
#  Extraction for a single parent‑folder / .d file
# ───────────────────────────────────────────────────────────────

def process_data(
    input_folder: str,
    mzmin: float,
    mzmax: float,
    progress_var,
    status_var,
    btn,
    root,
    label_source: str,  # transfer | delta6 | filename
    sort_cols: bool,
    use_recal: bool,
    pcs: str,
    do_bin: bool,
    nbins: int,
    do_ccs: bool,
    charge: str | None,
    mzval: str | None,
    polarity: str = "positive",  # positive | negative | both
) -> None:
    """Core routine called by UI & batch job runner."""

    master = pd.DataFrame()

    # Is *input_folder* already a .d file? If so, treat as single‑file list.
    if input_folder.lower().endswith(".d"):
        folder_list = [""]  # empty string -> operate directly on 'input_folder'
    else:
        subdirs = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
        folder_list = subdirs if subdirs else [""]

    total = len(folder_list)

    for i, sub in enumerate(folder_list, 1):
        folder = input_folder if sub == "" else os.path.join(input_folder, sub)
        status_var.set(f"Processing {os.path.basename(folder)}")
        root.update_idletasks()

        part = process_folder(
            folder,
            mzmin,
            mzmax,
            extraction_method="filename" if label_source == "filename" else "method",
            voltage_param_key=label_source if label_source != "filename" else "transfer",
            voltage_polarity=polarity,
            use_recalibrated_state=use_recal,
            pressure_compensation_strategy=pcs,
        )

        if part is None or part.empty:
            progress_var.set(i / total * 100)
            root.update_idletasks()
            continue

        # When using filename labelling, process_folder already renames the first
        # intensity column.  We *might* still have multiple IntensityX columns if
        # the acquisition used segments – rename them all consistently.
        if label_source == "filename":
            volt = _voltage_from_name(folder) or f"file{i}"
            int_cols = [c for c in part.columns if c != "Mobility"]
            rename_map = {c: volt if c.startswith("Intensity") else c for c in int_cols}
            part.rename(columns=rename_map, inplace=True)

        # Handle duplicate column names when merging multiple files
        if not master.empty:
            dupe = (set(master.columns) & set(part.columns)) - {"Mobility"}
            if dupe:
                part = part.rename(columns={c: f"{c}_{i}" for c in dupe})

        master = part if master.empty else pd.merge(master, part, on="Mobility", how="outer")

        progress_var.set(i / total * 100)
        root.update_idletasks()

    # ────────────────────────────────────────────────────────────
    # Post‑processing & export
    # ────────────────────────────────────────────────────────────
    if master.empty:
        messagebox.showerror("Error", "No data extracted.")
        btn.config(text="Select .d file/folder", state="normal")
        return

    master.fillna(0, inplace=True)

    if sort_cols and "Mobility" in master.columns:
        master.sort_values("Mobility", inplace=True)
        ordered = sorted([c for c in master.columns if c != "Mobility"], key=_numeric_key)
        master = master[["Mobility"] + ordered]

    if do_ccs:
        try:
            master["Mobility"] = master["Mobility"].apply(
                lambda x: oneOverK0ToCCSforMz(x, int(charge), float(mzval))
            )
        except Exception as exc:
            messagebox.showerror("Error", f"CCS conversion failed: {exc}")
            btn.config(text="Select .d file/folder", state="normal")
            return

    if do_bin:
        master = _bin_mobility(master, nbins)

    base = os.path.basename(input_folder.rstrip("/\\"))
    mz_tag = f"{int(mzmin)}-{int(mzmax)}"

    head1 = ["#mz range"] + [mz_tag] * (len(master.columns) - 1)
    head2 = ["#Raw file name"] + [base] * (len(master.columns) - 1)
    head3 = list(master.columns)

    out = pd.concat(
        [pd.DataFrame([head1, head2, head3], columns=master.columns), master],
        ignore_index=True,
    )

    fpath = os.path.join(input_folder, f"{base}_mz{mz_tag}_raw.csv")
    out.to_csv(fpath, index=False, header=False)

    status_var.set("Processing complete")
    btn.config(text="Select .d file/folder", state="normal")
    root.update_idletasks()


# ───────────────────────────────────────────────────────────────
#  Batch CSV runner
# ───────────────────────────────────────────────────────────────

def process_batch_data(df: pd.DataFrame, progress_var, status_var, button, root):
    total = len(df)

    for idx, row in df.iterrows():
        try:
            is_filename_mode = str(row.get("Label Source", "transfer")).lower() == "filename"
            target = str(row["Parent Folder"]).strip()

            # Basic validation
            if not is_filename_mode and not target.lower().endswith(".d"):
                messagebox.showerror(
                    "Error",
                    f"Row {idx+1}: Expected a single .d file for method‑based labelling, got: {target}",
                )
                continue

            process_data(
                input_folder=target,
                mzmin=float(row["mzmin"]),
                mzmax=float(row["mzmax"]),
                progress_var=progress_var,
                status_var=status_var,
                btn=button,
                root=root,
                label_source=row.get("Label Source", "transfer"),
                sort_cols=bool(row.get("Sort Columns", True)),
                use_recal=bool(row.get("Use Recalibrated State", True)),
                pcs=row.get(
                    "Pressure Compensation Strategy", "AnalysisGlobalPressureCompensation"
                ),
                do_bin=bool(row.get("Bin Mobility", False)),
                nbins=int(row.get("Num Bins", 200) if row.get("Bin Mobility", False) else 0),
                do_ccs=bool(row.get("Convert CCS", False)),
                charge=row.get("Charge") if row.get("Convert CCS", False) else None,
                mzval=row.get("mz for CCS") if row.get("Convert CCS", False) else None,
                polarity=row.get("Voltage Polarity", "positive"),
            )

        except Exception as exc:
            messagebox.showerror("Error", f"Row {idx+1} failed: {exc}")

        progress_var.set((idx + 1) / total * 100)
        root.update_idletasks()

    status_var.set("Batch processing complete")
    button.config(text="Batch Extraction", state="normal")

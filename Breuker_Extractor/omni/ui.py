# ui.py  ─  full GUI for tdfExtract
# Frames for bin‑count and CCS parameters remain invisible until requested.

from __future__ import annotations
import os, sys, threading, tkinter as tk
from tkinter import filedialog, messagebox, PhotoImage
import pandas as pd
from ttkbootstrap import Style, ttk
from processing import process_data, process_batch_data

# ── DLL search path setup ───────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    _bundle = sys._MEIPASS                                   # type: ignore[attr-defined]
    os.add_dll_directory(os.path.dirname(os.path.join(_bundle, "timsdata.dll")))
else:
    _bundle = os.path.dirname(os.path.abspath(__file__))
    os.add_dll_directory(os.getcwd())

ICON_PATH = os.path.join(_bundle, "fingerprint.png")


def create_ui() -> None:
    # ── root & theme ───────────────────────────────────────────────────────
    root = tk.Tk()
    Style(theme="flatly")
    root.title("tdfExtract")
    root.geometry("480x770")
    root.minsize(480, 740)
    root.iconphoto(False, PhotoImage(file=ICON_PATH))

    # ── Tk variables ───────────────────────────────────────────────────────
    mzmin_var, mzmax_var    = tk.StringVar(), tk.StringVar()
    label_src_var           = tk.StringVar(value="transfer")  # transfer|delta6|filename
    polarity_var            = tk.StringVar(value="positive")  # positive|negative|both
    sort_var                = tk.BooleanVar(value=True)

    bin_chk_var             = tk.BooleanVar(value=False)
    nbins_var               = tk.StringVar(value="200")

    ccs_chk_var             = tk.BooleanVar(value=False)
    charge_var, mzval_var   = tk.StringVar(), tk.StringVar()

    recalib_var             = tk.BooleanVar(value=True)
    pcs_var                 = tk.StringVar(value="Global")

    progress_var            = tk.DoubleVar(value=0)
    status_var              = tk.StringVar(value="Status: Ready")

    # ── layout frame ───────────────────────────────────────────────────────
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=0)

    ttk.Label(frame, text="tdfExtract", font=("Helvetica", 16))\
        .grid(row=0, column=0, columnspan=2, pady=(0, 18))

    # placeholders for dynamic frames
    bin_frame: tk.Frame | None = None
    ccs_frame: tk.Frame | None = None

    # ── helper creators ────────────────────────────────────────────────────
    def create_bin_frame() -> tk.Frame:
        nonlocal bin_frame
        if bin_frame is None:
            bin_frame = ttk.Frame(frame)
            ttk.Label(bin_frame, text="Number of bins:", font=("Helvetica", 12))\
                .grid(row=0, column=0, sticky="w")
            ttk.Entry(bin_frame, textvariable=nbins_var, font=("Helvetica", 12), width=6)\
                .grid(row=0, column=1, sticky="w")
        return bin_frame

    def create_ccs_frame() -> tk.Frame:
        nonlocal ccs_frame
        if ccs_frame is None:
            ccs_frame = ttk.Frame(frame)
            ttk.Label(ccs_frame, text="Charge:", font=("Helvetica", 12))\
                .grid(row=0, column=0, sticky="w")
            ttk.Entry(ccs_frame, textvariable=charge_var, font=("Helvetica", 12), width=5)\
                .grid(row=0, column=1, sticky="w", padx=(5, 15))
            ttk.Label(ccs_frame, text="m/z:", font=("Helvetica", 12))\
                .grid(row=0, column=2, sticky="w")
            ttk.Entry(ccs_frame, textvariable=mzval_var, font=("Helvetica", 12), width=8)\
                .grid(row=0, column=3, sticky="w")
        return ccs_frame

    # ── batch button placeholder; callback wired later ─────────────────────
    batch_btn = ttk.Button(frame, text="Batch Extraction",
                           bootstyle="primary", padding=(12, 6))
    batch_btn.grid(row=0, column=1, sticky="e", padx=10)

    # ── m/z inputs ─────────────────────────────────────────────────────────
    ttk.Label(frame, text="Enter the m/z range for the ion of interest.",
              font=("Helvetica", 14))\
        .grid(row=1, column=0, columnspan=2, pady=(0, 24))

    ttk.Label(frame, text="Minimum m/z:", font=("Helvetica", 12))\
        .grid(row=2, column=0, sticky="e")
    ttk.Entry(frame, textvariable=mzmin_var, font=("Helvetica", 12))\
        .grid(row=2, column=1, sticky="we")

    ttk.Label(frame, text="Maximum m/z:", font=("Helvetica", 12))\
        .grid(row=3, column=0, sticky="e")
    ttk.Entry(frame, textvariable=mzmax_var, font=("Helvetica", 12))\
        .grid(row=3, column=1, sticky="we")

    # ── voltage labelling choice ───────────────────────────────────────────
    lab_box = ttk.LabelFrame(frame, text="Voltage labelling", padding=(10, 5))
    lab_box.grid(row=4, column=0, columnspan=2, sticky="we", pady=(0, 10))
    ttk.Radiobutton(lab_box, text="Transfer Δ6 (.method)",
                    variable=label_src_var, value="transfer").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(lab_box, text="In‑TIMS Δ6 (.method)",
                    variable=label_src_var, value="delta6").grid(row=1, column=0, sticky="w")
    ttk.Radiobutton(lab_box, text="Parse from folder name (_120V_)",
                    variable=label_src_var, value="filename").grid(row=2, column=0, sticky="w")

    # polarity filter
    pol_box = ttk.LabelFrame(frame, text="Polarity filter (method modes)", padding=(10, 5))
    pol_box.grid(row=5, column=0, columnspan=2, sticky="we", pady=(0, 10))
    for i, txt in enumerate(("Positive", "Negative", "Both")):
        ttk.Radiobutton(pol_box, text=txt, value=txt.lower(),
                        variable=polarity_var).grid(row=0, column=i, sticky="w")

    def _toggle_pol(*_):
        state = "disabled" if label_src_var.get() == "filename" else "!disabled"
        for w in pol_box.winfo_children(): w.state([state])
    label_src_var.trace_add("write", _toggle_pol); _toggle_pol()

    ttk.Checkbutton(frame, text="Sort voltage columns numerically",
                    variable=sort_var).grid(row=6, column=0, columnspan=2, sticky="w")

    # ── binning checkbox ───────────────────────────────────────────────────
    ttk.Checkbutton(frame, text="Re‑bin mobility axis", variable=bin_chk_var)\
        .grid(row=7, column=0, columnspan=2, sticky="w")

    def _bin_toggle(*_):
        bf = create_bin_frame()
        if bin_chk_var.get():
            bf.grid(row=8, column=0, columnspan=2, sticky="w", padx=10)
        else:
            bf.grid_forget()
    bin_chk_var.trace_add("write", _bin_toggle)

    # ── CCS checkbox ───────────────────────────────────────────────────────
    ttk.Checkbutton(frame, text="Convert mobility to CCS", variable=ccs_chk_var)\
        .grid(row=9, column=0, columnspan=2, sticky="w")

    def _ccs_toggle(*_):
        cf = create_ccs_frame()
        if ccs_chk_var.get():
            cf.grid(row=10, column=0, columnspan=2, sticky="w", padx=10)
        else:
            cf.grid_forget()
    ccs_chk_var.trace_add("write", _ccs_toggle)

    # ── advanced settings dialog ───────────────────────────────────────────
    def _advanced():
        dlg = tk.Toplevel(root); dlg.title("Advanced Settings")
        dlg.resizable(False, False); dlg.geometry("300x130")
        rec = tk.BooleanVar(value=recalib_var.get())
        ttk.Checkbutton(dlg, text="Use recalibrated state", variable=rec)\
            .grid(row=0, column=0, sticky="w", padx=10, pady=10)
        ttk.Label(dlg, text="Pressure compensation:")\
            .grid(row=1, column=0, sticky="w", padx=10, pady=5)
        pcs_sel = tk.StringVar(value=pcs_var.get())
        ttk.Combobox(dlg, textvariable=pcs_sel, state="readonly",
                     values=("No compensation", "Per-frame", "Global"))\
            .grid(row=1, column=1, sticky="w", padx=10, pady=5)
        ttk.Button(dlg, text="Save",
                   command=lambda: (recalib_var.set(rec.get()),
                                    pcs_var.set(pcs_sel.get()), dlg.destroy()))\
            .grid(row=2, column=0, columnspan=2, pady=10)

    ttk.Button(frame, text="Advanced settings", bootstyle="primary",
               command=_advanced).grid(row=11, column=0, columnspan=2, pady=15)

    ttk.Label(frame, text='Output saved as "*_raw.csv" in selected folder',
              font=("Helvetica", 10)).grid(row=12, column=0, columnspan=2, pady=(0, 10))

    # process button
    process_btn = ttk.Button(frame, text="Select .d file/folder",
                             bootstyle="primary", padding=(10, 5))
    process_btn.grid(row=13, column=0, columnspan=2, pady=10)

    # progress & status
    pg_frame = ttk.Frame(frame, bootstyle="dark")
    pg_frame.grid(row=14, column=0, columnspan=2, sticky="we", pady=10)
    ttk.Progressbar(pg_frame, variable=progress_var, maximum=100, length=320,
                    bootstyle="info").pack(fill="both", expand=True, padx=2, pady=2)
    ttk.Label(frame, textvariable=status_var, font=("Helvetica", 12))\
        .grid(row=15, column=0, columnspan=2, sticky="w")

    for w in frame.winfo_children():
        w.grid_configure(padx=10, pady=5)

    process_data.update_status = lambda m: (status_var.set(m), root.update_idletasks())

    # ── callbacks ──────────────────────────────────────────────────────────
    def _run_single():
        # validate inputs
        try:
            mz1, mz2 = float(mzmin_var.get()), float(mzmax_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enter numeric m/z values."); return
        target = filedialog.askdirectory(title="Select .d file or parent folder")
        if not target: return
        if ccs_chk_var.get() and (not charge_var.get() or not mzval_var.get()):
            messagebox.showerror("Error", "Charge and m/z required for CCS."); return

        process_btn.config(text="Processing…", state="disabled")
        progress_var.set(0); status_var.set("Starting extraction…"); root.update_idletasks()

        threading.Thread(
            target=process_data,
            args=(target, mz1, mz2,
                  progress_var, status_var, process_btn, root,
                  label_src_var.get(), sort_var.get(),
                  recalib_var.get(), pcs_var.get(),
                  bin_chk_var.get(), int(nbins_var.get() or 0),
                  ccs_chk_var.get(), charge_var.get() or None, mzval_var.get() or None,
                  polarity_var.get()),
            daemon=True
        ).start()

    def _run_batch():
        csv_path = filedialog.askopenfilename(title="Select batch CSV",
                                              filetypes=[("CSV files", "*.csv")])
        if not csv_path: return
        try:
            batch_df = pd.read_csv(csv_path)
        except Exception as exc:
            messagebox.showerror("Error", f"CSV read error:\n{exc}"); return
        batch_btn.config(text="Batch running…", state="disabled")
        threading.Thread(target=process_batch_data,
                         args=(batch_df, progress_var, status_var, batch_btn, root),
                         daemon=True).start()

    process_btn.config(command=_run_single)
    batch_btn.config(command=_run_batch)

    root.mainloop()


if __name__ == "__main__":
    create_ui()

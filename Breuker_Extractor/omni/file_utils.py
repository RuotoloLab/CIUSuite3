"""
file_utils.py  –  misc. helpers
────────────────────────────────
• `voltage_from_filename`         – pull “_120V_” (or “120V”) from folder name
• `extract_voltage_from_method_file` – existing logic, now refactored
• small UI helpers (unchanged)
"""
import os
import re
import sys
import tkinter as tk
from tkinter import simpledialog, filedialog


# ───────────────────────────────────────────────────────────────
# Tiny UI helpers
# ───────────────────────────────────────────────────────────────
def select_folders():
    root = tk.Tk()
    root.withdraw()
    folder_paths = filedialog.askdirectory(
        title="Select one or more folders containing .d files", mustexist=True
    )
    return folder_paths.split()


def get_user_input(prompt, default_value):
    root = tk.Tk()
    root.withdraw()
    user_input = simpledialog.askfloat(
        title="Input", prompt=prompt, initialvalue=default_value
    )
    root.destroy()
    return user_input


# ───────────────────────────────────────────────────────────────
# 1)  Voltage encoded in the **folder name**
#     e.g.  “Native_140V_test.d”  →  140
# ───────────────────────────────────────────────────────────────
def voltage_from_filename(folder_path: str):
    """
    Return the first number that precedes the character “V” (case‑insensitive).
    If no such pattern, return None.
    """
    name = os.path.basename(folder_path)
    m = re.search(r'(\d+(?:\.\d+)?)\s*V', name, re.IGNORECASE)
    return float(m.group(1)) if m else None


# ───────────────────────────────────────────────────────────────
# 2)  Voltage from a *.method file (Transfer_PDU_Delta6 or Δ6)
#     – keeps all earlier functionality, plus polarity filtering.
# ───────────────────────────────────────────────────────────────
def extract_voltage_from_method_file(folder_path,
                                     param_key: str = "transfer",
                                     polarity: str = "positive"):
    """
    Parameters
    ----------
    folder_path : str
        Either the .d folder or its *.m sub‑folder.
    param_key   : 'transfer' | 'delta6'
    polarity    : 'positive' | 'negative' | 'both'
    Returns
    -------
    • float   – single value
    • list[float] – multiple segment voltages
    • None    – nothing found
    """
    print(f"Checking folder: {folder_path}", file=sys.stderr)

    # ── locate *.m folder ────────────────────────────────────────
    if folder_path.endswith(".m"):
        method_folder = folder_path
    else:
        method_folder = next(
            (os.path.join(folder_path, d) for d in os.listdir(folder_path)
             if d.endswith(".m")), None
        )

    if not method_folder:
        raise FileNotFoundError(f"No subfolder ending with '.m' in {folder_path!r}")

    # Prefer canonical filename if present
    preferred = "microtofqimpactemacquisition.method"
    method_file = next(
        (os.path.join(method_folder, f) for f in os.listdir(method_folder)
         if f.lower() == preferred), None
    ) or next(
        (os.path.join(method_folder, f) for f in os.listdir(method_folder)
         if f.lower().endswith(".method")), None
    )

    if not method_file:
        raise FileNotFoundError(f"No .method file found in {method_folder!r}")

    print(f"Method file found: {method_file}", file=sys.stderr)

    with open(method_file, encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    # Which parameter?
    target = ("Transfer_PDU_Delta6"
              if param_key.lower() == "transfer"
              else "IMS_TunnelVoltage_Delta_6")

    tag_pat  = re.compile(r'<\s*para_double\b[^>]*?>', re.I)
    attr_pat = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

    vals = []
    for tag in tag_pat.findall(content):
        attrs = dict(attr_pat.findall(tag))
        if attrs.get("permname") == target and "value" in attrs:
            raw = attrs["value"].strip()
            neg = raw.startswith('-')
            flt = round(float(raw), 1)
            pol = polarity.lower()
            if pol == "positive" and not neg:
                vals.append(flt)
            elif pol == "negative" and neg:
                vals.append(flt)
            elif pol == "both":
                vals.append(flt)

    if not vals:
        print(f"{target} not found (polarity={polarity}) in {method_file}", file=sys.stderr)
        return None

    # Deduplicate while preserving order
    seen, unique = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            unique.append(v)

    return unique[0] if len(unique) == 1 else unique

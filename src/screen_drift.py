r"""
Screen gridded drift tracks against native per-level GPS, and NaN any
profile where the two disagree by more than a threshold distance.

Cause and rationale.  Our horizontal drift track (\S3.1 of the spec) is
the time-integral of the gridded horizontal winds u, v.  When the
provider's ASPEN fall-speed model is accurate, \int u dt matches the
sonde's native GPS track to within a hundred metres.  When it is not,
the two disagree -- sometimes by many km -- and the gridded drift is no
longer trustworthy for that profile.  We detect this by comparing the
gridded surface drift (``x_offset``, ``y_offset`` at the bin nearest the
native surface altitude) to the native GPS drift from the earliest-time
fix to the latest-time fix.  Profiles whose disagreement exceeds
``threshold_m`` have ``x_offset``, ``y_offset``, ``lat``, and ``lon``
set to NaN.

Run after ``src/process.py`` (and alongside ``src/attach_climatology.py``).
See doc/regridding.tex \S3.1 for the discussion.

Usage:
    python screen_drift.py                  # all dropsonde files, 2 km threshold
    python screen_drift.py --threshold 3000 # override threshold
    python screen_drift.py joanne otrec     # specific datasets
"""

import argparse
import glob
import os
import sys

import netCDF4 as nc
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from test_validate_drift import DATASETS, _native_offset_m  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
DEFAULT_THRESHOLD_M = 2000.0


def _native_summary(sonde_id, native_lookup, max_deg_from_anchor=3.0):
    """Return (anchor_lat, anchor_lon, surface_lat, surface_lon, surface_alt)
    from the earliest- and latest-time valid GPS fixes in the native file,
    or None if the native file is unreadable or has no valid fixes.

    Some ASPEN-processed archives carry occasional sentinel GPS rows
    (e.g. ``(-89.58, 179.99)``) that would otherwise be picked as the
    "surface" sample.  We filter out samples whose ``(lat, lon)`` differs
    from the anchor by more than ``max_deg_from_anchor`` degrees, since
    no dropsonde drifts that far.
    """
    native = native_lookup(sonde_id)
    if native is None:
        return None
    if len(native) == 4:
        alt_n, lat_n, lon_n, time_n = native
    else:
        return None
    valid = np.isfinite(alt_n) & np.isfinite(lat_n) & np.isfinite(lon_n) \
        & np.isfinite(time_n)
    if not valid.any():
        return None
    idx = np.where(valid)[0]
    k_anchor = idx[np.argmin(time_n[idx])]
    lat_a = float(lat_n[k_anchor]); lon_a = float(lon_n[k_anchor])

    # Restrict to samples within a generous radius of the anchor
    not_outlier = (np.abs(lat_n - lat_a) <= max_deg_from_anchor) \
        & (np.abs(((lon_n - lon_a) + 180) % 360 - 180) <= max_deg_from_anchor)
    idx = np.where(valid & not_outlier)[0]
    if len(idx) == 0:
        return None
    # Spec (§3.1): compare against the native GPS drift at the lowest-altitude
    # bin with valid GPS.  This also guards against sentinel rows at extreme
    # later times that passed the anchor-radius filter.
    k_surface = idx[np.argmin(alt_n[idx])]
    return (lat_a, lon_a,
            float(lat_n[k_surface]), float(lon_n[k_surface]),
            float(alt_n[k_surface]))


def screen_dataset(path, native_lookup, threshold_m):
    """Screen one NetCDF; return (n_total, n_screened, list_of_sonde_ids)."""
    ds = xr.open_dataset(path)
    z_grid = ds["altitude"].values
    x_off = ds["x_offset"].values
    y_off = ds["y_offset"].values
    sonde_ids = [str(s) for s in ds["sonde_id"].values]
    ds.close()

    bad = []
    for i, sid in enumerate(sonde_ids):
        summary = _native_summary(sid, native_lookup)
        if summary is None:
            continue
        lat_a, lon_a, lat_s, lon_s, alt_s = summary
        # Grid surface drift = at the bin nearest the native surface altitude
        nb = int(np.argmin(np.abs(z_grid - alt_s)))
        xg = x_off[i, nb]
        yg = y_off[i, nb]
        if not (np.isfinite(xg) and np.isfinite(yg)):
            continue
        xn, yn = _native_offset_m(lat_s, lon_s, lat_a, lon_a)
        d = float(np.hypot(xg - xn, yg - yn))
        if d > threshold_m:
            bad.append((i, sid, d))

    if bad:
        with nc.Dataset(path, mode="a") as nfile:
            for i, sid, d in bad:
                for var in ("x_offset", "y_offset", "lat", "lon"):
                    if var in nfile.variables:
                        nfile.variables[var][i, :] = np.nan

    return len(sonde_ids), bad


def _discover(names):
    if not names:
        return [n for n in DATASETS if os.path.exists(
            os.path.join(OUTPUT_DIR, DATASETS[n][1]))]
    return [n for n in names if n in DATASETS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="dataset names, or omit for all")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_M,
                     help="flag profiles whose grid-vs-native surface drift "
                          "disagrees by more than this many metres")
    args = ap.parse_args()

    names = _discover(args.datasets)
    if not names:
        print("no matching dataset output files found")
        sys.exit(1)

    print(f"screening drift at {args.threshold:.0f} m threshold")
    total_screened = 0
    for name in names:
        native_lookup, output_name = DATASETS[name]
        path = os.path.join(OUTPUT_DIR, output_name)
        if not os.path.exists(path):
            print(f"  skip {name} (no output file)")
            continue
        n_total, bad = screen_dataset(path, native_lookup, args.threshold)
        total_screened += len(bad)
        if bad:
            print(f"  {name}: NaN'd {len(bad)}/{n_total}; worst:")
            for _, sid, d in sorted(bad, key=lambda r: -r[2])[:5]:
                print(f"     {d:8.0f} m  {sid}")
        else:
            print(f"  {name}: 0/{n_total} screened")

    print(f"done — {total_screened} profile(s) screened across "
          f"{len(names)} dataset(s)")


if __name__ == "__main__":
    main()

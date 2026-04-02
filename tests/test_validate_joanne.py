"""
Validate the regridded JOANNE output against JOANNE Level 3.

JOANNE L3 provides a provider-gridded product on a 10 m altitude grid
(0--10 000 m).  We load our output NetCDF and compare per-sonde RMSD
and mean bias to the L3 product.

The spec says the goal is "broad consistency" — not exact agreement,
because JOANNE L3 uses interpolation while we use bin averaging.
"""

import os
import sys

import numpy as np
import xarray as xr

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

JOANNE_L3_PATH = os.path.join(DATA_DIR, "joanne", "Level_3",
                               "EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")
OUR_PATH = os.path.join(OUTPUT_DIR, "joanne.nc")


def compute_stats(our_values, l3_values):
    """Per-sonde RMSD and mean bias where both are finite."""
    mask = np.isfinite(our_values) & np.isfinite(l3_values)
    if mask.sum() < 5:
        return np.nan, np.nan
    diff = our_values[mask] - l3_values[mask]
    rmsd = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    return rmsd, bias


def test_joanne_validation():
    """Compare our regridded output to JOANNE L3 for all matched sondes."""
    l3 = xr.open_dataset(JOANNE_L3_PATH)
    ours = xr.open_dataset(OUR_PATH)

    # L3 altitude grid: 0, 10, 20, ..., 10000  (1001 points)
    # Our grid: 5, 15, 25, ..., 9995  (cell centers, 1000 points within 0--10 km)
    l3_alt = l3["alt"].values
    our_alt = ours["altitude"].values
    # Restrict to altitudes within L3 range
    alt_mask = our_alt <= l3_alt[-1]
    our_alt_sub = our_alt[alt_mask]

    l3_sonde_ids = [str(s) for s in l3["sonde_id"].values]
    l3_idx = {sid: i for i, sid in enumerate(l3_sonde_ids)}

    our_sonde_ids = [str(s) for s in ours["sonde_id"].values]

    # Variables to compare (our name → L3 name)
    comparisons = {
        "T":  "ta",    # K
        "p":  "p",     # Pa
        "RH": "rh",    # L3 rh is 0--1, ours is 0--100
        "u":  "u",     # m/s
        "v":  "v",     # m/s
    }

    all_rmsd = {var: [] for var in comparisons}
    all_bias = {var: [] for var in comparisons}
    n_matched = 0

    for i, sonde_id in enumerate(our_sonde_ids):
        if sonde_id not in l3_idx:
            continue
        n_matched += 1
        j = l3_idx[sonde_id]

        for our_var, l3_var in comparisons.items():
            our_vals = ours[our_var].values[i, 0, alt_mask]

            l3_raw = l3[l3_var].values[j, :]
            # Interpolate L3 to our cell centers
            l3_interp = np.interp(our_alt_sub, l3_alt, l3_raw)

            # Handle RH unit mismatch: L3 is 0--1, ours is 0--100
            if our_var == "RH":
                l3_interp = l3_interp * 100.0

            rmsd, bias = compute_stats(our_vals, l3_interp)
            all_rmsd[our_var].append(rmsd)
            all_bias[our_var].append(bias)

    l3.close()
    ours.close()

    print(f"\nJOANNE validation: {n_matched} matched sondes")
    print(f"{'Variable':>8}  {'Median RMSD':>12}  {'Mean bias':>12}  {'Units':>8}")
    print("-" * 50)

    for var in comparisons:
        rmsds = np.array(all_rmsd[var])
        biases = np.array(all_bias[var])
        good = np.isfinite(rmsds)
        units = {"T": "K", "p": "Pa", "RH": "%", "u": "m/s", "v": "m/s"}[var]
        if good.any():
            print(f"{var:>8}  {np.median(rmsds[good]):12.4f}  {np.mean(biases[good]):12.4f}  {units:>8}")

    # Assert consistency — thresholds match the spec (doc/regridding.tex §6).
    thresholds = {
        "T":  (0.5,  "K"),
        "u":  (0.5,  "m/s"),
        "v":  (0.5,  "m/s"),
        "RH": (5.0,  "%"),
        "p":  (100.0, "Pa"),    # 1 hPa = 100 Pa
    }

    print("\nAssertion checks:")
    for var, (threshold, unit) in thresholds.items():
        rmsds = np.array(all_rmsd[var])
        biases = np.array(all_bias[var])
        good = np.isfinite(rmsds)
        if not good.any():
            continue
        med_rmsd = np.nanmedian(rmsds)
        mean_bias = np.nanmean(biases[good])
        assert med_rmsd < threshold, \
            f"{var} median RMSD too large: {med_rmsd:.3f} {unit} (threshold {threshold})"
        assert abs(mean_bias) < 0.5 * threshold, \
            f"{var} mean bias too large: {mean_bias:.3f} {unit} (threshold ±{0.5*threshold})"
        print(f"  {var:>4}: RMSD {med_rmsd:.4f} < {threshold} {unit}, "
              f"bias {mean_bias:+.4f} OK")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    test_joanne_validation()

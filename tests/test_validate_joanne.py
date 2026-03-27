"""
Validate our bin-averaged regridding against JOANNE Level 3.

JOANNE L3 provides a provider-gridded product on a 10 m altitude grid
(0--10 000 m).  We regrid each L2 profile with our algorithm and compare
per-sonde RMSD and mean bias to the L3 product.

The spec says the goal is "broad consistency" — not exact agreement,
because JOANNE L3 uses interpolation while we use bin averaging.
"""

import os
import sys

import numpy as np
import xarray as xr

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from readers import read_joanne
from regrid import regrid_sonde

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JOANNE_DIR = os.path.join(DATA_DIR, "joanne")
L3_PATH = os.path.join(JOANNE_DIR, "Level_3",
                        "EUREC4A_JOANNE_Dropsonde-RD41_Level_3_v2.0.0.nc")


def load_l3():
    """Load JOANNE L3 gridded product, indexed by sonde_id."""
    ds = xr.open_dataset(L3_PATH)
    return ds


def regrid_all_joanne():
    """Regrid all JOANNE L2 profiles to 10 m grid (0--10 000 m)."""
    profiles = read_joanne(JOANNE_DIR)
    results = {}
    for prof in profiles:
        ds = regrid_sonde(
            prof["altitude"],
            {"u": prof["u"], "v": prof["v"], "p": prof["p"],
             "T": prof["T"], "RH": prof["RH"]},
            z_min=0.0, z_max=10000.0, dz=10.0,
        )
        results[prof["sonde_id"]] = ds
    return results


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
    """Compare our regridding to JOANNE L3 for all matched sondes."""
    l3 = load_l3()
    ours = regrid_all_joanne()

    # L3 altitude grid: 0, 10, 20, ..., 10000  (1001 points)
    # Our grid: 5, 15, 25, ..., 9995  (cell centers, 1000 points)
    # They differ by 5 m offset and L3 has one extra point.
    # For comparison, interpolate L3 to our cell centers.
    l3_alt = l3["alt"].values      # 0, 10, 20, ..., 10000
    our_alt = np.arange(5.0, 10000.0, 10.0)  # 5, 15, ..., 9995

    l3_sonde_ids = [str(s) for s in l3["sonde_id"].values]
    l3_idx = {sid: i for i, sid in enumerate(l3_sonde_ids)}

    # Variables to compare (our name → L3 name, unit conversion)
    comparisons = {
        "T":  ("ta", 1.0, 0.0),     # K
        "p":  ("p",  1.0, 0.0),     # Pa
        "RH": ("rh", 1.0, 0.0),     # L3 rh is 0--1, ours is 0--100
        "u":  ("u",  1.0, 0.0),     # m/s
        "v":  ("v",  1.0, 0.0),     # m/s
    }

    all_rmsd = {var: [] for var in comparisons}
    all_bias = {var: [] for var in comparisons}
    n_matched = 0

    for sonde_id, our_ds in ours.items():
        if sonde_id not in l3_idx:
            continue
        n_matched += 1
        idx = l3_idx[sonde_id]

        for our_var, (l3_var, scale, offset) in comparisons.items():
            our_vals = our_ds[our_var].values  # on our_alt grid

            l3_raw = l3[l3_var].values[idx, :]  # on l3_alt grid
            # Interpolate L3 to our cell centers
            l3_interp = np.interp(our_alt, l3_alt, l3_raw)

            # Handle RH unit mismatch: L3 is 0--1, ours is 0--100
            if our_var == "RH":
                l3_interp = l3_interp * 100.0

            rmsd, bias = compute_stats(our_vals, l3_interp)
            all_rmsd[our_var].append(rmsd)
            all_bias[our_var].append(bias)

    l3.close()

    print(f"\nJOANNE validation: {n_matched} matched sondes of {len(ours)} regridded")
    print(f"{'Variable':>8}  {'Median RMSD':>12}  {'Mean bias':>12}  {'Units':>8}")
    print("-" * 50)

    for var in comparisons:
        rmsds = np.array(all_rmsd[var])
        biases = np.array(all_bias[var])
        good = np.isfinite(rmsds)
        units = {"T": "K", "p": "Pa", "RH": "%", "u": "m/s", "v": "m/s"}[var]
        if good.any():
            print(f"{var:>8}  {np.median(rmsds[good]):12.4f}  {np.mean(biases[good]):12.4f}  {units:>8}")

    # Assert broad consistency — these thresholds allow for
    # interpolation-vs-averaging differences.
    median_T_rmsd = np.nanmedian(all_rmsd["T"])
    median_u_rmsd = np.nanmedian(all_rmsd["u"])
    median_v_rmsd = np.nanmedian(all_rmsd["v"])
    median_RH_rmsd = np.nanmedian(all_rmsd["RH"])

    assert median_T_rmsd < 1.0, f"T RMSD too large: {median_T_rmsd:.3f} K"
    assert median_u_rmsd < 1.0, f"u RMSD too large: {median_u_rmsd:.3f} m/s"
    assert median_v_rmsd < 1.0, f"v RMSD too large: {median_v_rmsd:.3f} m/s"
    assert median_RH_rmsd < 5.0, f"RH RMSD too large: {median_RH_rmsd:.3f} %"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    test_joanne_validation()

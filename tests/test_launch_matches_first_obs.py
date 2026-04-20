"""
Sanity-check that the stored launch_lat/launch_lon matches the sonde's
own first per-level GPS fix, for every dataset whose source files report
per-level lat/lon.

Providers sometimes populate the launch position from an aircraft
centroid or mission plan rather than the sonde's first GPS fix, with
offsets of up to ~15 km (e.g. HALO-(AC)3).  Our readers are supposed to
pick up the earliest-time GPS fix instead.  This test exercises a random
sample of output profiles, re-parses the matching source file, and
checks that the stored launch position is within 200 m of the first-GPS
fix.
"""

import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Re-use the native-source parsers already defined for the drift test
from test_validate_drift import (  # noqa: E402
    DATASETS,
    _haversine_m,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

N_PROFILES = 25
TOLERANCE_M = 200.0
RNG_SEED = 0


@pytest.mark.parametrize("dataset_name", list(DATASETS.keys()))
def test_launch_matches_first_obs(dataset_name):
    native_lookup, output_name = DATASETS[dataset_name]
    output_path = os.path.join(OUTPUT_DIR, output_name)
    if not os.path.exists(output_path):
        pytest.skip(f"Output file {output_name} not present — regenerate first")

    ds = xr.open_dataset(output_path)
    launch_lats = ds["launch_lat"].values
    launch_lons = ds["launch_lon"].values
    sonde_ids = [str(s) for s in ds["sonde_id"].values]
    ds.close()

    rng = np.random.default_rng(RNG_SEED)
    profile_order = rng.permutation(len(sonde_ids))

    errors = []
    n_checked = 0
    worst = (0.0, None)
    for i in profile_order:
        if n_checked >= N_PROFILES:
            break
        native = native_lookup(sonde_ids[i])
        if native is None:
            continue
        if len(native) == 4:
            alt_n, lat_n, lon_n, time_n = native
        else:
            alt_n, lat_n, lon_n = native
            time_n = np.arange(len(alt_n), dtype=np.float64)
        valid = np.isfinite(time_n) & np.isfinite(lat_n) & np.isfinite(lon_n)
        if not valid.any():
            continue
        idx = np.where(valid)[0]
        k = idx[np.argmin(time_n[idx])]
        d = _haversine_m(launch_lats[i], launch_lons[i], lat_n[k], lon_n[k])
        errors.append(d)
        if d > worst[0]:
            worst = (d, sonde_ids[i])
        n_checked += 1

    if not errors:
        pytest.skip(f"{dataset_name}: no native-readable profiles found")

    errors = np.array(errors)
    max_err = float(errors.max())
    median_err = float(np.median(errors))

    print(f"\n{dataset_name}: n={len(errors)}; median={median_err:.1f} m, "
          f"max={max_err:.1f} m ({worst[1]})")

    assert max_err < TOLERANCE_M, (
        f"{dataset_name}: launch_lat/lon is {max_err:.1f} m from the "
        f"first-GPS fix (sonde {worst[1]}); expected < {TOLERANCE_M} m. "
        f"Median across {len(errors)} sondes: {median_err:.1f} m."
    )


if __name__ == "__main__":
    for name in DATASETS:
        try:
            test_launch_matches_first_obs(name)
        except AssertionError as e:
            print(f"{name}: FAIL — {e}")

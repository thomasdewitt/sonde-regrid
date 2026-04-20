"""
Validate the integrated horizontal drift tracks against native per-level
GPS coordinates, for every dataset whose source files report them.

For each dataset, a random subset of profiles is re-read through the
existing reader, random (altitude, lat, lon) triples are drawn from the
native profile, and the native position is compared to the gridded
(lat, lon) at the nearest altitude bin.  The haversine distance is
required to stay below 500 m; the RMS distance is reported.

These tests assume the output NetCDFs have been regenerated since the
drift integration was added to the pipeline.
"""

import os
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from readers import (
    read_joanne,
    read_beach,
    read_otrec,
    read_dynamo,
    read_haloac3,
    read_activate,
    read_shout,
    read_hurricane,
    read_hs3,
    read_predict,
    read_arrecon,
    read_enrr,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

MAX_ERROR_M = 500.0
N_PROFILES_PER_DATASET = 20
N_POINTS_PER_PROFILE = 10
RNG_SEED = 0

# Per-dataset source readers.  Each returns profiles carrying per-level
# lat/lon in the `obs_time` branch of the reader; we re-parse the source
# for the raw native coordinates since `readers.py` discards them.
def _native_joanne(sonde_id):
    path = os.path.join(DATA_DIR, "joanne", "Level_2",
                         f"{sonde_id}.nc")
    if not os.path.exists(path):
        # JOANNE filenames are complex; search
        import glob
        matches = glob.glob(os.path.join(DATA_DIR, "joanne", "Level_2",
                                           f"*{sonde_id}*.nc"))
        if not matches:
            return None
        path = matches[0]
    ds = xr.open_dataset(path)
    alt = ds["alt"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64) if "lat" in ds else None
    lon = ds["lon"].values.astype(np.float64) if "lon" in ds else None
    ds.close()
    if lat is None or lon is None:
        return None
    return alt, lat, lon


def _native_beach(sonde_id):
    import glob
    matches = glob.glob(os.path.join(DATA_DIR, "beach", "Level_2", "*",
                                       f"{sonde_id}.zarr"))
    if not matches:
        return None
    try:
        ds = xr.open_dataset(matches[0], engine="zarr", consolidated=False)
    except Exception:
        return None
    if "alt" not in ds.data_vars or "lat" not in ds or "lon" not in ds:
        ds.close()
        return None
    alt = ds["alt"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64)
    lon = ds["lon"].values.astype(np.float64)
    ds.close()
    return alt, lat, lon


def _native_nc_gpsalt(sonde_id, data_subdir):
    import glob
    matches = glob.glob(os.path.join(DATA_DIR, data_subdir, f"{sonde_id}.nc"))
    if not matches:
        matches = glob.glob(os.path.join(DATA_DIR, data_subdir, f"*{sonde_id}*.nc"))
    if not matches:
        return None
    ds = xr.open_dataset(matches[0])
    alt = ds["gpsalt"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64)
    lon = ds["lon"].values.astype(np.float64)
    ds.close()
    MISSING = -999.0
    alt[alt <= MISSING] = np.nan
    lat[lat <= MISSING] = np.nan
    lon[lon <= MISSING] = np.nan
    return alt, lat, lon


def _native_haloac3(sonde_id):
    import glob
    matches = glob.glob(os.path.join(DATA_DIR, "halo-ac3", "Level_2",
                                       f"*{sonde_id}*.nc"))
    if not matches:
        return None
    ds = xr.open_dataset(matches[0])
    alt = ds["gpsalt"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64) if "lat" in ds else None
    lon = ds["lon"].values.astype(np.float64) if "lon" in ds else None
    ds.close()
    if lat is None or lon is None:
        return None
    return alt, lat, lon


def _native_activate(sonde_id):
    import glob
    import io
    matches = glob.glob(os.path.join(DATA_DIR, "activate", f"{sonde_id}.ict"))
    if not matches:
        return None
    with open(matches[0]) as fh:
        lines = fh.readlines()
    n_header = int(lines[0].split(",")[0])
    cols = lines[n_header - 1].strip().split(",")
    col_idx = {name.strip(): i for i, name in enumerate(cols)}
    if "Latitude" not in col_idx or "Longitude" not in col_idx:
        return None
    data = np.genfromtxt(io.StringIO("".join(lines[n_header:])),
                          delimiter=",", filling_values=np.nan)
    if data.ndim < 2:
        return None
    MISSING = -9999.0
    alt = data[:, col_idx["GPS Altitude"]].astype(np.float64)
    lat = data[:, col_idx["Latitude"]].astype(np.float64)
    lon = data[:, col_idx["Longitude"]].astype(np.float64)
    alt[alt <= MISSING] = np.nan
    lat[lat <= MISSING] = np.nan
    lon[lon <= MISSING] = np.nan
    return alt, lat, lon


def _native_eol(path):
    """Return (alt, lat, lon) from an EOL-format file, or None."""
    import io
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return None
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("------"):
            header_end = i + 1
            break
    if header_end == 0:
        return None
    dls = [l.strip() for l in lines[header_end:]
           if l.strip() and not l.strip().startswith("/")]
    if not dls:
        return None
    data = np.genfromtxt(io.StringIO("\n".join(dls)), invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ncols = data.shape[1]
    MISSING = -999.0
    if ncols > 16:
        alt = data[:, 16].astype(np.float64)
        geop = data[:, 13].astype(np.float64)
        alt[alt <= MISSING] = np.nan
        geop[geop <= MISSING] = np.nan
        alt[~np.isfinite(alt)] = geop[~np.isfinite(alt)]
    elif ncols > 13:
        alt = data[:, 13].astype(np.float64)
        alt[alt <= MISSING] = np.nan
    else:
        return None
    lat = data[:, 15].astype(np.float64) if ncols > 15 else np.full_like(alt, np.nan)
    lon = data[:, 14].astype(np.float64) if ncols > 14 else np.full_like(alt, np.nan)
    lat[lat <= MISSING] = np.nan
    lon[lon <= MISSING] = np.nan
    return alt, lat, lon


def _native_frd(path):
    """Return (alt, lat, lon) from an ASPEN FRD file, or None."""
    import io
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return None
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("IX"):
            header_end = i + 1
            break
    if header_end == 0:
        return None
    dls = [l.strip() for l in lines[header_end:] if l.strip()]
    if not dls:
        return None
    data = np.genfromtxt(io.StringIO("\n".join(dls)), invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ncols = data.shape[1]
    MISSING = -999.0
    alt = data[:, 5].astype(np.float64) if ncols > 5 else None
    if alt is None:
        return None
    alt[alt <= MISSING] = np.nan
    lat = data[:, 17].astype(np.float64) if ncols > 17 else np.full_like(alt, np.nan)
    lon = data[:, 18].astype(np.float64) if ncols > 18 else np.full_like(alt, np.nan)
    lat[lat <= MISSING] = np.nan
    lon[lon <= MISSING] = np.nan
    return alt, lat, lon


def _native_by_path_glob(sonde_id, pattern_roots):
    """Look up a sonde_id in a list of (root, suffix) pairs."""
    import glob
    for root, suffix in pattern_roots:
        matches = glob.glob(os.path.join(root, "**", f"{sonde_id}{suffix}"),
                             recursive=True)
        if matches:
            return matches[0]
    return None


def _native_shout(sonde_id):
    import glob
    matches = glob.glob(os.path.join(DATA_DIR, "shout", f"{sonde_id}.eol"))
    # readers preserves basename (incl .eol) — sonde_id already contains it
    if not matches:
        matches = glob.glob(os.path.join(DATA_DIR, "shout", f"{sonde_id}"))
    if not matches:
        return None
    return _native_eol(matches[0])


def _native_hurricane(sonde_id):
    path = _native_by_path_glob(
        sonde_id,
        [(os.path.join(DATA_DIR, "hurricane"), "")],
    )
    return _native_eol(path) if path else None


def _native_hs3(sonde_id):
    path = _native_by_path_glob(
        sonde_id, [(os.path.join(DATA_DIR, "hs3"), "")],
    )
    return _native_eol(path) if path else None


def _native_predict(sonde_id):
    import glob
    matches = glob.glob(os.path.join(DATA_DIR, "predict", f"{sonde_id}"))
    if not matches:
        return None
    return _native_eol(matches[0])


def _native_arrecon(sonde_id):
    path = _native_by_path_glob(
        sonde_id, [(os.path.join(DATA_DIR, "arrecon"), "")],
    )
    return _native_frd(path) if path else None


def _native_enrr(sonde_id):
    # Could be FRD (corrected/, c130_drybiascor/) or EOL (globalhawk_corrected/)
    path = _native_by_path_glob(
        sonde_id,
        [(os.path.join(DATA_DIR, "enrr", "corrected"), ""),
         (os.path.join(DATA_DIR, "enrr", "c130_drybiascor"), "")],
    )
    if path:
        return _native_frd(path)
    path = _native_by_path_glob(
        sonde_id,
        [(os.path.join(DATA_DIR, "enrr", "globalhawk_corrected"), "")],
    )
    return _native_eol(path) if path else None


DATASETS = {
    "joanne":    (lambda sid: _native_joanne(sid),   "joanne.nc"),
    "beach":     (lambda sid: _native_beach(sid),    "beach.nc"),
    "otrec":     (lambda sid: _native_nc_gpsalt(sid, "otrec"),  "otrec.nc"),
    "dynamo":    (lambda sid: _native_nc_gpsalt(sid, "dynamo"), "dynamo.nc"),
    "haloac3":   (lambda sid: _native_haloac3(sid),  "haloac3.nc"),
    "activate":  (lambda sid: _native_activate(sid), "activate.nc"),
    "shout":     (_native_shout,     "shout.nc"),
    "hurricane": (_native_hurricane, "hurricane.nc"),
    "hs3":       (_native_hs3,       "hs3.nc"),
    "predict":   (_native_predict,   "predict.nc"),
    "arrecon":   (_native_arrecon,   "arrecon.nc"),
    "enrr":      (_native_enrr,      "enrr.nc"),
}


def _haversine_m(lat1, lon1, lat2, lon2, R=6_371_000.0):
    phi1 = np.deg2rad(lat1); phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1); dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


@pytest.mark.parametrize("dataset_name", list(DATASETS.keys()))
def test_drift_matches_native(dataset_name):
    native_lookup, output_name = DATASETS[dataset_name]
    output_path = os.path.join(OUTPUT_DIR, output_name)
    if not os.path.exists(output_path):
        pytest.skip(f"Output file {output_name} not present — regenerate first")

    ds = xr.open_dataset(output_path)
    z_grid = ds["altitude"].values
    lat_grid = ds["lat"].values
    lon_grid = ds["lon"].values
    sonde_ids = [str(s) for s in ds["sonde_id"].values]
    ds.close()

    rng = np.random.default_rng(RNG_SEED)
    profile_order = rng.permutation(len(sonde_ids))

    errors = []
    n_profiles_used = 0
    n_profiles_native_found = 0
    n_profiles_empty_gridded = 0

    for i in profile_order:
        if n_profiles_used >= N_PROFILES_PER_DATASET:
            break
        native = native_lookup(sonde_ids[i])
        if native is None:
            continue
        alt_n, lat_n, lon_n = native

        valid_n = np.isfinite(alt_n) & np.isfinite(lat_n) & np.isfinite(lon_n)
        if valid_n.sum() < N_POINTS_PER_PROFILE:
            continue
        n_profiles_native_found += 1

        native_idx = rng.choice(np.where(valid_n)[0],
                                  size=N_POINTS_PER_PROFILE, replace=False)

        profile_has_match = False
        for j in native_idx:
            nearest = int(np.argmin(np.abs(z_grid - alt_n[j])))
            lat_g = lat_grid[i, nearest]
            lon_g = lon_grid[i, nearest]
            if not (np.isfinite(lat_g) and np.isfinite(lon_g)):
                continue
            d = _haversine_m(lat_n[j], lon_n[j], lat_g, lon_g)
            errors.append(d)
            profile_has_match = True

        if profile_has_match:
            n_profiles_used += 1
        else:
            n_profiles_empty_gridded += 1

    # A dataset-level skip is only allowed if the source data itself was
    # unavailable (profiles couldn't be re-read with enough valid native
    # lat/lon).  Once native data was found but the gridded product has
    # no finite lat/lon, that is a failure, not a skip.
    if n_profiles_native_found == 0:
        pytest.skip(f"{dataset_name}: no native profiles with per-level lat/lon "
                     "could be re-read from source")

    assert n_profiles_used > 0, (
        f"{dataset_name}: {n_profiles_native_found} native profiles carry "
        f"per-level lat/lon, but the gridded output has no finite lat/lon "
        f"on any of them (checked {n_profiles_empty_gridded})"
    )

    errors = np.array(errors)
    max_err = float(errors.max())
    rms_err = float(np.sqrt(np.mean(errors ** 2)))
    mean_err = float(errors.mean())

    print(f"\n{dataset_name}: n={len(errors)} samples from "
          f"{n_profiles_used} profiles; max={max_err:.1f} m, "
          f"RMS={rms_err:.1f} m, mean={mean_err:.1f} m")

    assert max_err < MAX_ERROR_M, (
        f"{dataset_name}: max drift error {max_err:.1f} m exceeds "
        f"{MAX_ERROR_M} m (RMS {rms_err:.1f}, mean {mean_err:.1f})"
    )


if __name__ == "__main__":
    for name in DATASETS:
        try:
            test_drift_matches_native(name)
        except Exception as exc:
            print(f"{name}: FAILED — {exc}")

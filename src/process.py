"""
Process all datasets: read → regrid → diagnose → write NetCDF.

Produces one NetCDF per dataset in output/, with dimensions (sonde, altitude)
and per-sonde metadata as described in doc/regridding.tex.

Usage:
    python process.py                 # process all datasets
    python process.py joanne otrec    # process specific datasets
"""

import os
import sys
import time

import numpy as np
import xarray as xr

from regrid import regrid_sonde, DEFAULT_DZ
from readers import (
    read_joanne,
    read_beach,
    read_otrec,
    read_activate,
    read_hurricane,
    read_dynamo,
    read_shout,
    read_arrecon,
    read_igra,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")

Z_MIN = 0.0
Z_MAX_DEFAULT = 20000.0   # 20 km for dropsonde datasets
Z_MAX_IGRA = 40000.0      # 40 km for IGRA radiosondes
DZ = DEFAULT_DZ

DATA_VARIABLES = ["u", "v", "p", "T", "RH", "q", "theta", "theta_e", "MSE", "DSE"]

DATASETS = {
    "joanne":    {"reader": read_joanne,    "path": os.path.join(DATA_DIR, "joanne"),
                  "z_max": Z_MAX_DEFAULT},
    "beach":     {"reader": read_beach,     "path": os.path.join(DATA_DIR, "beach"),
                  "z_max": Z_MAX_DEFAULT},
    "otrec":     {"reader": read_otrec,     "path": os.path.join(DATA_DIR, "otrec"),
                  "z_max": Z_MAX_DEFAULT},
    "activate":  {"reader": read_activate,  "path": os.path.join(DATA_DIR, "activate"),
                  "z_max": Z_MAX_DEFAULT},
    "hurricane": {"reader": read_hurricane, "path": os.path.join(DATA_DIR, "hurricane",
                   "1996-2012.NOAA.Hurricane.Dropsonde.Archive.v2"),
                  "z_max": Z_MAX_DEFAULT},
    "dynamo":    {"reader": read_dynamo,    "path": os.path.join(DATA_DIR, "dynamo"),
                  "z_max": Z_MAX_DEFAULT},
    "shout":     {"reader": read_shout,     "path": os.path.join(DATA_DIR, "shout"),
                  "z_max": Z_MAX_DEFAULT},
    "arrecon":   {"reader": read_arrecon,   "path": os.path.join(DATA_DIR, "arrecon"),
                  "z_max": Z_MAX_DEFAULT},
    "igra":      {"reader": read_igra,      "path": os.path.join(DATA_DIR, "igra"),
                  "z_max": Z_MAX_IGRA},
}


def process_dataset(name, reader, data_path, z_max=None):
    """Read, regrid, and save one dataset to output/{name}.nc."""
    if z_max is None:
        z_max = Z_MAX_DEFAULT

    print(f"\n{'='*60}")
    print(f"Processing {name}")
    print(f"{'='*60}")

    t0 = time.time()
    profiles = reader(data_path)
    t_read = time.time() - t0
    print(f"  Read {len(profiles)} profiles in {t_read:.1f}s")

    if not profiles:
        print("  No profiles — skipping.")
        return

    # Build the altitude coordinate (cell centers)
    edges = np.arange(Z_MIN, z_max + DZ, DZ)
    altitude = 0.5 * (edges[:-1] + edges[1:])
    n_alt = len(altitude)
    n_sondes = len(profiles)

    # Pre-allocate output arrays
    data_arrays = {var: np.full((n_sondes, n_alt), np.nan) for var in DATA_VARIABLES}
    sonde_ids = []
    launch_times = []
    launch_lats = np.full(n_sondes, np.nan)
    launch_lons = np.full(n_sondes, np.nan)
    station_ids = []

    t0 = time.time()
    for i, prof in enumerate(profiles):
        variables = {}
        for var in ("u", "v", "p", "T", "RH"):
            if var in prof and prof[var] is not None:
                variables[var] = prof[var]
        if "q" in prof and prof["q"] is not None:
            variables["q"] = prof["q"]

        ds = regrid_sonde(prof["altitude"], variables,
                          z_min=Z_MIN, z_max=z_max, dz=DZ)

        for var in DATA_VARIABLES:
            if var in ds:
                data_arrays[var][i, :] = ds[var].values

        sonde_ids.append(prof.get("sonde_id", f"sonde_{i:06d}"))
        launch_times.append(prof.get("launch_time"))
        launch_lats[i] = prof.get("launch_lat", np.nan)
        launch_lons[i] = prof.get("launch_lon", np.nan)
        station_ids.append(prof.get("station_id", prof.get("sonde_id", "")))

    t_regrid = time.time() - t0
    print(f"  Regridded in {t_regrid:.1f}s")

    # Handle launch_time: convert to datetime64 array
    launch_time_arr = np.array(launch_times, dtype="datetime64[ns]")

    # Build output dataset
    out = xr.Dataset(
        {var: (("sonde", "altitude"), data_arrays[var]) for var in DATA_VARIABLES},
        coords={
            "altitude": altitude,
            "sonde_id": ("sonde", sonde_ids),
            "launch_time": ("sonde", launch_time_arr),
            "launch_lat": ("sonde", launch_lats),
            "launch_lon": ("sonde", launch_lons),
            "station_id": ("sonde", station_ids),
        },
    )

    # Coordinate attributes
    out["altitude"].attrs = {
        "units": "m",
        "long_name": "altitude above mean sea level",
        "axis": "Z",
        "positive": "up",
    }
    out["launch_lat"].attrs = {
        "units": "degrees_north",
        "long_name": "latitude at profile start",
        "standard_name": "latitude",
    }
    out["launch_lon"].attrs = {
        "units": "degrees_east",
        "long_name": "longitude at profile start",
        "standard_name": "longitude",
    }
    out["launch_time"].attrs = {
        "long_name": "date and time of profile start",
    }
    out["sonde_id"].attrs = {
        "long_name": "provider sonde or profile identifier",
    }
    out["station_id"].attrs = {
        "long_name": "station or platform identifier",
        "comment": "set equal to sonde_id for all datasets",
    }

    # Data variable attributes
    variable_attrs = {
        "u":       {"units": "m s-1",   "long_name": "zonal wind",
                    "standard_name": "eastward_wind"},
        "v":       {"units": "m s-1",   "long_name": "meridional wind",
                    "standard_name": "northward_wind"},
        "p":       {"units": "Pa",      "long_name": "pressure",
                    "standard_name": "air_pressure"},
        "T":       {"units": "K",       "long_name": "temperature",
                    "standard_name": "air_temperature"},
        "RH":      {"units": "%",       "long_name": "relative humidity",
                    "standard_name": "relative_humidity"},
        "q":       {"units": "kg kg-1", "long_name": "water vapor mixing ratio",
                    "standard_name": "humidity_mixing_ratio"},
        "theta":   {"units": "K",       "long_name": "potential temperature",
                    "standard_name": "air_potential_temperature"},
        "theta_e": {"units": "K",       "long_name": "equivalent potential temperature",
                    "standard_name": "equivalent_potential_temperature",
                    "comment": "Bolton (1980) formulation"},
        "MSE":     {"units": "J kg-1",  "long_name": "moist static energy",
                    "comment": "cp*T + g*z + Lv*q"},
        "DSE":     {"units": "J kg-1",  "long_name": "dry static energy",
                    "comment": "cp*T + g*z"},
    }
    for var, attrs in variable_attrs.items():
        if var in out:
            out[var].attrs = attrs

    # Global attributes
    out.attrs = {
        "title": f"Regridded sonde profiles — {name}",
        "source": f"Bin-averaged from {name} dataset",
        "history": f"Created by sonde-regrid/src/process.py on {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "Conventions": "CF-1.8",
        "regridding_method": "bin averaging (no interpolation)",
        "grid_spacing_m": DZ,
        "grid_min_m": Z_MIN,
        "grid_max_m": z_max,
        "n_sondes": n_sondes,
        "n_altitude_levels": n_alt,
        "variables_directly_gridded": "u, v, p, T, RH",
        "variables_diagnosed_post_gridding": "q, theta, theta_e, MSE, DSE",
        "missing_value_note": "NaN indicates grid cells with no valid observations",
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, f"{name}.nc")
    out.to_netcdf(outpath)
    print(f"  Saved {outpath} ({n_sondes} sondes × {n_alt} levels)")


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())

    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}")
            print(f"Available: {', '.join(DATASETS.keys())}")
            sys.exit(1)

    for name in names:
        cfg = DATASETS[name]
        process_dataset(name, cfg["reader"], cfg["path"], z_max=cfg["z_max"])

    print("\nDone.")


if __name__ == "__main__":
    main()

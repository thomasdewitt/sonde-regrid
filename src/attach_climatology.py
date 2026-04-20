"""
Attach ERA5 monthly climatology (u_clim, v_clim) to regridded NetCDF
output files, in place.

Run after ``src/process.py`` has populated ``output/``.  The script
iterates over output NetCDFs one ERA5 variable at a time — first
``u_clim`` across all files, then ``v_clim`` — so that only a single
20 GB slab resides in memory simultaneously.

See doc/regridding.tex §3.2 for the algorithm.

Usage:
    python attach_climatology.py                 # all output NetCDFs
    python attach_climatology.py joanne otrec    # specific dropsonde files
    python attach_climatology.py igra            # all IGRA stations
    python attach_climatology.py igra:USM00072451,USM00072764   # selected
"""

import glob
import os
import sys
import time

import netCDF4 as nc
import numpy as np
import xarray as xr

from climatology import (
    clear_cache,
    climatology_available,
    interpolate_climatology_at_points,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT, "output")

CLIM_ATTRS = {
    "u_clim": {
        "units": "m s-1",
        "long_name": "ERA5 climatological zonal wind",
        "comment": "Monthly mean u from ERA5 (1995-2025), bilinearly "
                    "interpolated in lat/lon to the launch point and "
                    "linearly interpolated in altitude to the output grid. "
                    "Dropsonde files: one profile per sounding at the launch "
                    "month. IGRA files: 12 monthly profiles per station.",
    },
    "v_clim": {
        "units": "m s-1",
        "long_name": "ERA5 climatological meridional wind",
        "comment": "Monthly mean v from ERA5 (1995-2025), bilinearly "
                    "interpolated in lat/lon to the launch point and "
                    "linearly interpolated in altitude to the output grid. "
                    "Dropsonde files: one profile per sounding at the launch "
                    "month. IGRA files: 12 monthly profiles per station.",
    },
}


def _is_igra(path):
    return os.sep + "igra" + os.sep in path


def _launch_months(launch_times):
    """Calendar month (1..12) from datetime64 array.  NaT → 0."""
    lt_M = np.asarray(launch_times, dtype="datetime64[M]")
    lt_Y = np.asarray(launch_times, dtype="datetime64[Y]")
    return np.where(np.isnat(lt_M), 0,
                     (lt_M - lt_Y).astype(int) + 1).astype(np.int64)


def _discover(names):
    """Expand argv into a list of output NetCDF paths."""
    if not names:
        paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.nc")))
        paths += sorted(glob.glob(os.path.join(OUTPUT_DIR, "igra", "*.nc")))
        return paths

    paths = []
    for n in names:
        if n.startswith("igra:"):
            for sid in n.split(":", 1)[1].split(","):
                p = os.path.join(OUTPUT_DIR, "igra", f"{sid}.nc")
                if os.path.exists(p):
                    paths.append(p)
                else:
                    print(f"warning: {p} not found, skipping")
        elif n == "igra":
            paths += sorted(glob.glob(os.path.join(OUTPUT_DIR, "igra", "*.nc")))
        else:
            p = os.path.join(OUTPUT_DIR, f"{n}.nc")
            if os.path.exists(p):
                paths.append(p)
            else:
                print(f"warning: {p} not found, skipping")
    return paths


def _station_lat_lon(launch_lats, launch_lons):
    """For an IGRA file, return the single (lat, lon) of the station by
    taking the first finite pair (station location is fixed)."""
    for lat, lon in zip(launch_lats, launch_lons):
        if np.isfinite(lat) and np.isfinite(lon):
            return float(lat), float(lon)
    return np.nan, np.nan


def _write_var(path, vname, dims, values):
    """Create or overwrite one variable in an existing NetCDF file."""
    with nc.Dataset(path, mode="a") as ds:
        # Ensure every required dimension exists
        if "month" in dims and "month" not in ds.dimensions:
            ds.createDimension("month", 12)
            m = ds.createVariable("month", "i1", ("month",))
            m[:] = np.arange(1, 13, dtype=np.int8)
            m.setncatts({"long_name": "calendar month", "units": "1"})
        if vname in ds.variables:
            ds.variables[vname][:] = values
        else:
            v = ds.createVariable(vname, "f4", dims, zlib=True, complevel=4)
            v[:] = values
            v.setncatts(CLIM_ATTRS[vname])


def _update_global_attr(path):
    """Amend the variables_diagnosed_post_gridding global attribute."""
    with nc.Dataset(path, mode="a") as ds:
        existing = ds.getncattr("variables_diagnosed_post_gridding") \
            if "variables_diagnosed_post_gridding" in ds.ncattrs() else ""
        if "u_clim" in existing:
            return
        ds.setncattr("variables_diagnosed_post_gridding",
                      (existing + (", " if existing else "") + "u_clim, v_clim"))


def attach_variable(path, era5_var):
    """Attach {era5_var}_clim to one NetCDF file.

    era5_var : "u" or "v".
    """
    vname = f"{era5_var}_clim"
    ds = xr.open_dataset(path)
    altitude = ds["altitude"].values.astype(np.float64)
    launch_lats = ds["launch_lat"].values
    launch_lons = ds["launch_lon"].values
    launch_times = ds["launch_time"].values
    ds.close()

    if _is_igra(path):
        lat, lon = _station_lat_lon(launch_lats, launch_lons)
        if not (np.isfinite(lat) and np.isfinite(lon)):
            values = np.full((12, len(altitude)), np.nan, dtype=np.float32)
        else:
            lats = np.full(12, lat, dtype=np.float64)
            lons = np.full(12, lon, dtype=np.float64)
            months = np.arange(1, 13, dtype=np.int64)
            clim = interpolate_climatology_at_points(
                lats, lons, months, altitude, variables=(era5_var,))
            values = clim[era5_var]
        _write_var(path, vname, ("month", "altitude"), values)
    else:
        months = _launch_months(launch_times)
        clim = interpolate_climatology_at_points(
            launch_lats.astype(np.float64), launch_lons.astype(np.float64),
            months, altitude, variables=(era5_var,))
        _write_var(path, vname, ("sounding_id", "altitude"), clim[era5_var])


def main(argv):
    if not climatology_available():
        print("ERROR: ERA5 climatology files not found at "
              "../era5-climatology/data/era5_{u,v}_climatology_1995-2025.nc")
        sys.exit(1)

    paths = _discover(argv)
    if not paths:
        print("no output files to process")
        sys.exit(1)

    print(f"attaching ERA5 climatology to {len(paths)} NetCDF file(s)")

    for era5_var in ("u", "v"):
        print(f"\n=== pass: {era5_var}_clim ===")
        for i, path in enumerate(paths, 1):
            t0 = time.time()
            try:
                attach_variable(path, era5_var)
            except Exception as e:
                print(f"  [{i}/{len(paths)}] {path}: FAILED — {e}")
                continue
            print(f"  [{i}/{len(paths)}] {path}  {time.time()-t0:.1f}s")
        print(f"releasing {era5_var} cache...")
        clear_cache()

    print("\nupdating global attributes...")
    for path in paths:
        _update_global_attr(path)

    print("done.")


if __name__ == "__main__":
    main(sys.argv[1:])

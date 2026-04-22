"""
Attach ERA5 monthly climatology (u_clim, v_clim) to regridded NetCDF
output files, in place.

Run after ``src/process.py`` has populated ``output/``.  The script
iterates over output NetCDFs one ERA5 variable at a time — first
``u_clim`` across all files, then ``v_clim`` — so that only a single
~19 GB slab set (12 months of one variable) resides in memory at once.

With ``-j N`` (``N`` > 1), the per-file work for each variable pass is
distributed across ``N`` fork-pool workers.  Slabs are loaded once in
the parent before the pool is created, and the fork pool inherits them
via COW, so total system memory stays near the single-process peak
(~19 GB) regardless of worker count.

See doc/regridding.tex §3.2 for the algorithm.

Usage:
    python attach_climatology.py                 # all output NetCDFs
    python attach_climatology.py joanne otrec    # specific dropsonde files
    python attach_climatology.py igra            # all IGRA stations
    python attach_climatology.py igra:USM00072451,USM00072764   # selected
    python attach_climatology.py -j 8 igra       # 8 workers per variable pass
"""

import concurrent.futures as _cf
import glob
import multiprocessing as _mp
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
    _slab,
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


def _attach_task(args):
    """Worker entry point: attach one variable to one file."""
    path, era5_var = args
    t0 = time.time()
    try:
        attach_variable(path, era5_var)
    except Exception as e:
        return path, f"FAILED — {e}", time.time() - t0
    return path, "ok", time.time() - t0


def _mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0 / 1024.0
    except Exception:
        pass
    return float("nan")


def _run_variable_pass(paths, era5_var, workers):
    """Run one variable pass either serially or via a fork pool."""
    if workers <= 1:
        for i, path in enumerate(paths, 1):
            t0 = time.time()
            try:
                attach_variable(path, era5_var)
            except Exception as e:
                print(f"  [{i}/{len(paths)}] {path}: FAILED — {e}")
                continue
            print(f"  [{i}/{len(paths)}] {path}  {time.time()-t0:.1f}s")
        return

    # Pre-warm the 12 month slabs in the parent; fork-pool workers then
    # share them via COW so total memory stays near a single-process peak.
    print(f"  warming {era5_var} slab cache in parent "
          f"(avail {_mem_available_gb():.1f} GB)...")
    t0 = time.time()
    for m in range(1, 13):
        _slab(era5_var, m)
    print(f"  warm in {time.time()-t0:.1f}s; avail now {_mem_available_gb():.1f} GB")

    ctx = _mp.get_context("fork")
    tasks = [(p, era5_var) for p in paths]
    with _cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        done = 0
        for path, status, dt in ex.map(_attach_task, tasks):
            done += 1
            tag = "" if status == "ok" else f"  {status}"
            print(f"  [{done}/{len(paths)}] {path}  {dt:.1f}s{tag}")


def main(argv, workers=32):
    if not climatology_available():
        print("ERROR: ERA5 climatology files not found at "
              "../era5-climatology/data/era5_{u,v}_climatology_1995-2025.nc")
        sys.exit(1)

    paths = _discover(argv)
    if not paths:
        print("no output files to process")
        sys.exit(1)

    print(f"attaching ERA5 climatology to {len(paths)} NetCDF file(s) "
          f"with {workers} worker(s)")

    for era5_var in ("u", "v"):
        print(f"\n=== pass: {era5_var}_clim ===")
        _run_variable_pass(paths, era5_var, workers)
        print(f"releasing {era5_var} cache...")
        clear_cache()

    print("\nupdating global attributes...")
    for path in paths:
        try:
            _update_global_attr(path)
        except Exception as e:
            print(f"  {path}: FAILED — {e}")
            continue

    print("done.")


def _parse_jobs(argv):
    """Pop -j / --jobs from argv and return (workers, remaining_argv)."""
    workers = 32
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-j", "--jobs"):
            workers = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--jobs="):
            workers = int(a.split("=", 1)[1])
            i += 1
            continue
        if a.startswith("-j") and a[2:].isdigit():
            workers = int(a[2:])
            i += 1
            continue
        rest.append(a)
        i += 1
    return workers, rest


if __name__ == "__main__":
    workers, rest = _parse_jobs(sys.argv[1:])
    main(rest, workers=workers)

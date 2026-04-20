"""
ERA5 monthly climatology interpolation for sonde launch locations.

Reads the per-variable monthly climatology NetCDFs produced by the
sibling repository ``era5-climatology`` at
``../era5-climatology/data/era5_{u,v}_climatology_1995-2025.nc``.  Each
file has dimensions (month=12, altitude=401, latitude=721,
longitude=1440) with monthly mean winds on a 0.25° lat/lon grid and a
100 m altitude grid from 0 to 40 000 m.  Values below the ERA5 surface
orography are NaN.

See doc/regridding.tex §3.2 for how these climatologies are attached to
each profile.
"""

import os

import numpy as np
import xarray as xr


ERA5_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                         "era5-climatology", "data")
CLIM_SPAN = "1995-2025"


def climatology_path(var):
    """Path to the ERA5 monthly climatology NetCDF for variable `var`."""
    return os.path.join(ERA5_DIR, f"era5_{var}_climatology_{CLIM_SPAN}.nc")


def climatology_available(variables=("u", "v")):
    """True if all requested climatology files exist on disk."""
    return all(os.path.exists(climatology_path(v)) for v in variables)


def _open_var(var):
    return xr.open_dataset(climatology_path(var))[var]


# Module-level cache of loaded month slabs.  Each (var, month) slab is a
# (n_altitude, n_latitude, n_longitude) float32 DataArray of ~1.66 GB on
# disk / ~1.66 GB in memory.  Loading once per (var, month) amortises the
# I/O cost across all datasets that call into this module in a single
# Python process.  Peak memory if every month of u and v is touched is
# ~40 GB; short pipelines pay less.
_SLAB_CACHE = {}
_VAR_HANDLES = {}
_ALT_COORD = None


def _slab(var, month):
    """Return the (altitude, latitude, longitude) slab for one variable and
    month, loading from disk on the first request and caching thereafter."""
    global _ALT_COORD
    key = (var, int(month))
    cached = _SLAB_CACHE.get(key)
    if cached is not None:
        return cached
    handle = _VAR_HANDLES.get(var)
    if handle is None:
        handle = _open_var(var)
        _VAR_HANDLES[var] = handle
        if _ALT_COORD is None:
            _ALT_COORD = handle.altitude.values.astype(np.float64)
    slab = handle.sel(month=int(month)).load()
    _SLAB_CACHE[key] = slab
    return slab


def clear_cache():
    """Drop all cached month slabs and release the underlying files."""
    global _ALT_COORD
    _SLAB_CACHE.clear()
    for h in _VAR_HANDLES.values():
        try:
            h.close()
        except Exception:
            pass
    _VAR_HANDLES.clear()
    _ALT_COORD = None


def _month_slab_interp(slab, lat_da, lon_da):
    """Bilinear lat/lon interpolation of a single-month (altitude, lat, lon) slab.

    Returns (N, n_altitude) float32 array, where N = len(lat_da).
    """
    prof = slab.interp(latitude=lat_da, longitude=lon_da)
    # xarray may return dims (altitude, N); ensure (N, altitude)
    if prof.dims[0] == "altitude":
        prof = prof.transpose(lat_da.dims[0], "altitude")
    return prof.values


def _altitude_interp(src_alt, src_vals, target_alt):
    """Per-row linear interpolation over altitude, preserving NaN below the
    ERA5 surface orography.  `src_vals` has shape (N, n_alt_src)."""
    target_alt = np.asarray(target_alt, dtype=np.float64)
    n = src_vals.shape[0]
    out = np.full((n, len(target_alt)), np.nan, dtype=np.float32)
    for i in range(n):
        col = src_vals[i]
        finite = np.isfinite(col)
        if finite.sum() < 2:
            continue
        z = src_alt[finite]
        f = col[finite]
        # Outside the finite range (above ERA5 top or below orography) stay NaN
        interp = np.interp(target_alt, z, f, left=np.nan, right=np.nan)
        in_range = (target_alt >= z[0]) & (target_alt <= z[-1])
        out[i, in_range] = interp[in_range]
    return out


def interpolate_climatology_at_points(launch_lats, launch_lons, launch_months,
                                       target_altitudes, variables=("u", "v")):
    """Interpolate ERA5 climatology at N launch points and a target altitude grid.

    Parameters
    ----------
    launch_lats, launch_lons : ndarray of float, shape (N,)
        Launch coordinates [degrees].  NaN entries yield NaN output rows.
    launch_months : ndarray of int, shape (N,)
        Calendar month (1..12) for each point.  Values outside 1..12 yield
        NaN output rows (e.g. profiles with no launch_time).
    target_altitudes : ndarray of float
        Altitude grid [m] to interpolate onto.
    variables : tuple of str
        ERA5 variable short names (default ("u", "v")).

    Returns
    -------
    dict[str, ndarray]
        One (N, len(target_altitudes)) float32 array per variable.
    """
    launch_lats = np.asarray(launch_lats, dtype=np.float64)
    launch_lons = np.asarray(launch_lons, dtype=np.float64)
    launch_months = np.asarray(launch_months, dtype=np.int64)
    target_altitudes = np.asarray(target_altitudes, dtype=np.float64)
    n = len(launch_lats)
    results = {v: np.full((n, len(target_altitudes)), np.nan, dtype=np.float32)
               for v in variables}

    valid = (np.isfinite(launch_lats) & np.isfinite(launch_lons)
             & (launch_months >= 1) & (launch_months <= 12))
    if not valid.any():
        return results

    # ERA5 longitudes span [0, 360); fold negative longitudes
    lons_pos = np.where(launch_lons < 0, launch_lons + 360.0, launch_lons)

    # Warm up the handles so _ALT_COORD is populated for the altitude interp
    for v in variables:
        _slab(v, 1)
    src_alt = _ALT_COORD

    for m in range(1, 13):
        mask = valid & (launch_months == m)
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        lat_da = xr.DataArray(launch_lats[idx], dims=["s"])
        lon_da = xr.DataArray(lons_pos[idx], dims=["s"])

        for v in variables:
            slab = _slab(v, m)
            src_vals = _month_slab_interp(slab, lat_da, lon_da)
            results[v][idx, :] = _altitude_interp(src_alt, src_vals,
                                                    target_altitudes)

    return results


def interpolate_climatology_monthly(lat, lon, target_altitudes,
                                      variables=("u", "v")):
    """Interpolate the 12 monthly climatology profiles at a single location.

    Intended for IGRA station files, which are keyed by a fixed station
    location and carry one monthly climatology per station rather than
    one per sounding.

    Parameters
    ----------
    lat, lon : float
        Station location [degrees].
    target_altitudes : ndarray of float
        Altitude grid [m] to interpolate onto.
    variables : tuple of str
        ERA5 variable short names.

    Returns
    -------
    dict[str, ndarray]
        One (12, len(target_altitudes)) float32 array per variable, with
        axis-0 indexing months 1..12.
    """
    lats = np.full(12, lat, dtype=np.float64)
    lons = np.full(12, lon, dtype=np.float64)
    months = np.arange(1, 13, dtype=np.int64)
    return interpolate_climatology_at_points(lats, lons, months,
                                              target_altitudes, variables)

"""
Uniform vertical regridding of radiosonde and dropsonde profiles.

The algorithm constructs a 1-D grid of uniform vertical spacing, then
bin-averages all measurements falling within each grid cell.  Cells with
no measurements are set to NaN.  Derived thermodynamic variables are
diagnosed *after* bin averaging.

See doc/regridding.tex for a complete description.
"""

import numpy as np
import xarray as xr

from diagnostics import (
    mixing_ratio_from_rh,
    potential_temperature,
    moist_static_energy,
    dry_static_energy,
    equivalent_potential_temperature,
)
from drift import integrate_drift

DEFAULT_DZ = 10.0  # meters


def make_grid(z_min, z_max, dz=DEFAULT_DZ):
    """Return cell-edge altitudes for a uniform grid.

    Parameters
    ----------
    z_min, z_max : float
        Altitude bounds [m].
    dz : float
        Grid spacing [m].

    Returns
    -------
    edges : ndarray, shape (N+1,)
        Cell edges.  Cell k spans [edges[k], edges[k+1]).
    """
    return np.arange(z_min, z_max + dz, dz)


def bin_average(altitude, values, edges):
    """Bin-average observed values onto a uniform grid.

    Parameters
    ----------
    altitude : ndarray, shape (M,)
        Observed altitudes [m].
    values : ndarray, shape (M,)
        Observed variable values.
    edges : ndarray, shape (N+1,)
        Cell edges from make_grid.

    Returns
    -------
    gridded : ndarray, shape (N,)
        Bin-averaged values.  NaN where no valid observations exist.
    """
    n_cells = len(edges) - 1
    gridded = np.full(n_cells, np.nan)

    # Assign each observation to a bin (0-based)
    bin_idx = np.digitize(altitude, edges) - 1

    # Keep only observations that fall inside the grid and have finite values
    valid = (bin_idx >= 0) & (bin_idx < n_cells) & np.isfinite(values)
    if not valid.any():
        return gridded

    idx = bin_idx[valid]
    vals = values[valid]
    sums = np.bincount(idx, weights=vals, minlength=n_cells)
    counts = np.bincount(idx, minlength=n_cells)
    populated = counts > 0
    gridded[populated] = sums[populated] / counts[populated]

    return gridded


def bin_average_time(altitude, times, edges):
    """Bin-average datetime64 values onto a uniform grid.

    Converts to float (ns since epoch), bin-averages, converts back.
    Returns NaT where no valid observations exist.
    """
    # Convert datetime64 → float64 nanoseconds, replacing NaT with NaN
    times_ns = times.astype("datetime64[ns]")
    ns = np.where(np.isnat(times_ns), np.nan, times_ns.astype(np.int64).astype(np.float64))
    avg_ns = bin_average(altitude, ns, edges)
    # Convert back: NaN → NaT
    result = np.full(len(avg_ns), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = np.isfinite(avg_ns)
    result[valid] = avg_ns[valid].astype("int64").astype("datetime64[ns]")
    return result


def regrid_sonde(altitude, variables, z_min, z_max, dz=DEFAULT_DZ, obs_time=None,
                  estimated_obs_time=None, launch_lat=np.nan, launch_lon=np.nan):
    """Regrid a single sonde profile and diagnose derived variables.

    Parameters
    ----------
    altitude : ndarray, shape (M,)
        Observed geometric altitudes [m].
    variables : dict of str -> ndarray
        Mapping of variable name to observed values.  Expected keys:
        "u", "v", "p" (Pa), "T" (K).  Optional: "RH" (%, 0--100)
        or "q" (kg/kg).
    z_min, z_max : float
        Altitude bounds [m] for the output grid.
    dz : float
        Grid spacing [m].
    obs_time : ndarray of datetime64, optional
        Per-observation timestamps, same length as altitude.  Becomes the
        ``observation_time`` coordinate in the output.
    estimated_obs_time : ndarray of datetime64, optional
        Per-observation timestamps with missing values filled by a source-
        specific synthesis (e.g., IGRA's piecewise ascent model).  When
        given, is bin-averaged to produce ``estimated_observation_time``
        and is used for drift integration in place of ``obs_time``.
    launch_lat, launch_lon : float, optional
        Launch position.  Used to anchor the horizontal drift track
        (§3.1 of the spec).  NaN disables lat/lon outputs.

    Returns
    -------
    ds : xarray.Dataset
        Regridded profile on cell-center altitudes, with both directly
        averaged and diagnosed variables.  Includes "observation_time"
        if obs_time is provided, "estimated_observation_time" if
        estimated_obs_time is provided, and horizontal drift fields
        (x_offset, y_offset, lat, lon) when winds and a per-level time
        are all available on the grid.
    """
    edges = make_grid(z_min, z_max, dz)
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    # --- Bin-average directly measured variables ---
    gridded = {}
    for var in ("u", "v", "p", "T", "RH"):
        if var in variables:
            gridded[var] = bin_average(altitude, variables[var], edges)

    # --- Mixing ratio: from RH or directly provided ---
    if "q" in variables and "RH" not in variables:
        gridded["q"] = bin_average(altitude, variables["q"], edges)
    elif all(v in gridded for v in ("RH", "T", "p")):
        gridded["q"] = mixing_ratio_from_rh(gridded["RH"], gridded["T"], gridded["p"])

    # --- Potential temperature ---
    if "T" in gridded and "p" in gridded:
        gridded["theta"] = potential_temperature(gridded["T"], gridded["p"])

    # --- Equivalent potential temperature ---
    if all(v in gridded for v in ("theta", "q", "T", "RH")):
        gridded["theta_e"] = equivalent_potential_temperature(
            gridded["theta"], gridded["q"], gridded["T"], gridded["RH"]
        )

    # --- Moist static energy ---
    if "T" in gridded and "q" in gridded:
        gridded["MSE"] = moist_static_energy(gridded["T"], z_centers, gridded["q"])

    # --- Dry static energy ---
    if "T" in gridded:
        gridded["DSE"] = dry_static_energy(gridded["T"], z_centers)

    ds = xr.Dataset(
        {name: ("altitude", values) for name, values in gridded.items()},
        coords={"altitude": z_centers},
    )
    ds["altitude"].attrs = {"units": "m", "long_name": "altitude above mean sea level"}

    gridded_time = None
    if obs_time is not None:
        gridded_time = bin_average_time(altitude, obs_time, edges)
        ds["observation_time"] = ("altitude", gridded_time)

    gridded_est_time = None
    if estimated_obs_time is not None:
        gridded_est_time = bin_average_time(altitude, estimated_obs_time, edges)
        ds["estimated_observation_time"] = ("altitude", gridded_est_time)

    # Prefer the estimated (always-populated) time for drift integration
    # when available, so profiles lacking ETIME still get a drift track.
    drift_time = gridded_est_time if gridded_est_time is not None else gridded_time
    if drift_time is not None and "u" in gridded and "v" in gridded:
        x_off, y_off, lat, lon = integrate_drift(
            z_centers, drift_time, gridded["u"], gridded["v"],
            launch_lat, launch_lon,
        )
        ds["x_offset"] = ("altitude", x_off)
        ds["y_offset"] = ("altitude", y_off)
        ds["lat"] = ("altitude", lat)
        ds["lon"] = ("altitude", lon)

    return ds

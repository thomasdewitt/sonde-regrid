"""
Uniform vertical regridding of radiosonde and dropsonde profiles.

The algorithm is intentionally simple: construct a 1D grid of uniform
vertical spacing, then bin-average all measurements falling within each
grid cell.  Cells with no measurements are set to NaN.

See doc/regridding.tex for a complete description.
"""

import numpy as np
import xarray as xr

from diagnostics import (
    mixing_ratio_from_rh,
    potential_temperature,
    moist_static_energy,
)

# Default grid spacing [m]
DEFAULT_DZ = 10.0

# Variables that are directly regridded (bin-averaged)
DIRECT_VARS = ["u", "v", "p", "T", "RH"]

# Variables diagnosed after regridding
DIAGNOSED_VARS = ["q", "theta", "MSE"]


def make_grid(z_min: float, z_max: float, dz: float = DEFAULT_DZ) -> np.ndarray:
    """Return cell-edge altitudes for a uniform grid.

    Parameters
    ----------
    z_min, z_max : float
        Altitude bounds [m].  The grid spans from z_min to z_max.
    dz : float
        Grid spacing [m].

    Returns
    -------
    edges : ndarray, shape (N+1,)
        Monotonically increasing cell edges.  Cell *k* spans
        [edges[k], edges[k+1]).
    """
    edges = np.arange(z_min, z_max + dz, dz)
    return edges


def bin_average(z_obs: np.ndarray, values: np.ndarray,
                edges: np.ndarray) -> np.ndarray:
    """Bin-average observed values onto a uniform grid.

    For each grid cell defined by consecutive entries in *edges*, the
    output is the arithmetic mean of all observations whose altitude
    falls within that cell.  Cells containing no observations are NaN.

    Parameters
    ----------
    z_obs : ndarray, shape (M,)
        Observed altitudes [m].
    values : ndarray, shape (M,)
        Observed variable at each altitude.
    edges : ndarray, shape (N+1,)
        Cell edges produced by :func:`make_grid`.

    Returns
    -------
    gridded : ndarray, shape (N,)
        Bin-averaged values.  NaN where no observations exist.
    """
    N = len(edges) - 1
    gridded = np.full(N, np.nan)

    # np.digitize: bin index 1..N for values inside [edges[0], edges[-1])
    bin_idx = np.digitize(z_obs, edges) - 1  # shift to 0-based

    for k in range(N):
        mask = bin_idx == k
        if mask.any():
            vals = values[mask]
            good = np.isfinite(vals)
            if good.any():
                gridded[k] = np.nanmean(vals[good])

    return gridded


def regrid_profile(z_obs: np.ndarray,
                   profile: dict[str, np.ndarray],
                   z_min: float = 0.0,
                   z_max: float = 30000.0,
                   dz: float = DEFAULT_DZ) -> xr.Dataset:
    """Regrid a single sonde profile onto a uniform vertical grid.

    Parameters
    ----------
    z_obs : ndarray, shape (M,)
        Observed geometric altitudes [m].
    profile : dict
        Mapping of variable name -> observed values.  Expected keys
        are at minimum ``"u"``, ``"v"``, ``"p"``, ``"T"``.
        Optional: ``"RH"`` (relative humidity, 0--100).
    z_min, z_max : float
        Altitude bounds [m] for the output grid.
    dz : float
        Grid spacing [m].

    Returns
    -------
    ds : xarray.Dataset
        Regridded profile on cell-center altitudes, including both
        directly averaged and diagnosed variables.
    """
    edges = make_grid(z_min, z_max, dz)
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    data_vars = {}

    # --- Bin-average directly measured variables ---
    for var in DIRECT_VARS:
        if var in profile:
            data_vars[var] = ("z", bin_average(z_obs, profile[var], edges))

    # --- Diagnose derived variables ---
    T_grid = data_vars.get("T", (None, None))[1]
    p_grid = data_vars.get("p", (None, None))[1]
    RH_grid = data_vars.get("RH", (None, None))[1]

    if T_grid is not None and p_grid is not None and RH_grid is not None:
        q = mixing_ratio_from_rh(RH_grid, T_grid, p_grid)
        data_vars["q"] = ("z", q)
    elif "q" in profile:
        # If mixing ratio is provided directly, regrid it
        data_vars["q"] = ("z", bin_average(z_obs, profile["q"], edges))
        q = data_vars["q"][1]
    else:
        q = None

    if T_grid is not None and p_grid is not None:
        data_vars["theta"] = ("z", potential_temperature(T_grid, p_grid))

    if T_grid is not None and q is not None:
        data_vars["MSE"] = ("z", moist_static_energy(T_grid, z_centers, q))

    ds = xr.Dataset(
        {k: v for k, v in data_vars.items()},
        coords={"z": z_centers},
    )
    ds["z"].attrs = {"units": "m", "long_name": "Altitude (geometric)"}

    return ds

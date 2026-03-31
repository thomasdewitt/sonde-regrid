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

    for k in range(n_cells):
        mask = bin_idx == k
        if mask.any():
            vals = values[mask]
            good = np.isfinite(vals)
            if good.any():
                gridded[k] = np.mean(vals[good])

    return gridded


def regrid_sonde(altitude, variables, z_min, z_max, dz=DEFAULT_DZ):
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

    Returns
    -------
    ds : xarray.Dataset
        Regridded profile on cell-center altitudes, with both directly
        averaged and diagnosed variables.
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

    return ds

"""
Thermodynamic diagnostic variables for sonde profiles.

All functions operate element-wise on arrays and propagate NaN.
"""

import numpy as np

# --- Physical constants ---
CP = 1005.7       # Specific heat of dry air at constant pressure [J kg-1 K-1]
G = 9.80665       # Standard gravitational acceleration [m s-2]
LV = 2.501e6      # Latent heat of vaporization at 0 °C [J kg-1]
RD = 287.04       # Specific gas constant for dry air [J kg-1 K-1]
P0 = 100000.0     # Reference pressure for potential temperature [Pa]
KAPPA = RD / CP   # Poisson constant (~0.2854)


def saturation_vapor_pressure(T: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure over liquid water [Pa].

    Uses the August-Roche-Magnus formula:
        e_s = 610.94 * exp(17.625 * T_c / (T_c + 243.04))
    where T_c is temperature in Celsius.

    Parameters
    ----------
    T : ndarray
        Temperature [K].

    Returns
    -------
    e_s : ndarray
        Saturation vapor pressure [Pa].
    """
    Tc = T - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


def mixing_ratio_from_rh(RH: np.ndarray, T: np.ndarray,
                          p: np.ndarray) -> np.ndarray:
    """Water vapor mixing ratio from relative humidity.

    Parameters
    ----------
    RH : ndarray
        Relative humidity [%] (0--100).
    T : ndarray
        Temperature [K].
    p : ndarray
        Pressure [Pa].

    Returns
    -------
    q : ndarray
        Mixing ratio [kg kg-1].
    """
    es = saturation_vapor_pressure(T)
    e = (RH / 100.0) * es
    # Mixing ratio: q = 0.622 * e / (p - e)
    return 0.622 * e / (p - e)


def potential_temperature(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Potential temperature [K].

    theta = T * (p0 / p)^kappa

    Parameters
    ----------
    T : ndarray
        Temperature [K].
    p : ndarray
        Pressure [Pa].

    Returns
    -------
    theta : ndarray
        Potential temperature [K].
    """
    return T * (P0 / p) ** KAPPA


def moist_static_energy(T: np.ndarray, z: np.ndarray,
                        q: np.ndarray) -> np.ndarray:
    """Moist static energy [J kg-1].

    MSE = c_p T + g z + L_v q

    Parameters
    ----------
    T : ndarray
        Temperature [K].
    z : ndarray
        Geometric altitude [m].
    q : ndarray
        Water vapor mixing ratio [kg kg-1].

    Returns
    -------
    MSE : ndarray
        Moist static energy [J kg-1].
    """
    return CP * T + G * z + LV * q

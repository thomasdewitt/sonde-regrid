"""
Horizontal drift integration for regridded sonde profiles.

Integrates gridded horizontal winds in time, anchored at the launch
position, to produce a per-level drift track on the altitude grid.
See doc/regridding.tex §3.1 for the algorithm.
"""

import numpy as np

R_EARTH = 6_371_000.0   # Mean Earth radius [m]
MAX_GAP_M = 1000.0      # Integration aborts once a single gap exceeds this


def _time_to_seconds(obs_time):
    """Convert datetime64[ns] array to float seconds (NaT → NaN)."""
    t = np.asarray(obs_time, dtype="datetime64[ns]")
    out = np.where(np.isnat(t), np.nan, t.astype(np.int64).astype(np.float64) * 1e-9)
    return out


def integrate_drift(obs_time, u, v, launch_lat, launch_lon,
                    max_gap_m=MAX_GAP_M, R_earth=R_EARTH):
    """Integrate horizontal drift from gridded winds and observation times.

    The altitude grid is reindexed in ascending order of observation time,
    treated as if the sonde starts at the launch position at the earliest
    time, and marched forward in time using trapezoidal integration on the
    mean of adjacent bins' winds.  Missing wind values are filled by
    carrying forward the last valid value; once the accumulated drift in a
    single contiguous gap exceeds max_gap_m, the rest of the track (in
    time order) is set to NaN.

    Parameters
    ----------
    obs_time : ndarray of datetime64[ns], shape (N,)
        Observation time at each altitude bin (NaT if missing).
    u, v : ndarray of float, shape (N,)
        Bin-averaged zonal and meridional winds [m/s].  May contain NaNs.
    launch_lat, launch_lon : float
        Launch position [degrees].  If NaN, lat/lon outputs are NaN but
        x_offset/y_offset are still computed.
    max_gap_m : float
        Maximum horizontal distance that may be accumulated during a single
        contiguous wind-gap before the rest of the track is NaN-flagged.
    R_earth : float
        Mean Earth radius [m] for the meters-to-degrees conversion.

    Returns
    -------
    x_offset, y_offset : ndarray, shape (N,)
        Eastward and northward drift from launch [m].
    lat, lon : ndarray, shape (N,)
        Per-level latitude and longitude [degrees].  NaN if launch
        coordinates are missing.
    """
    n = len(u)
    x_offset = np.full(n, np.nan)
    y_offset = np.full(n, np.nan)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)

    t_sec = _time_to_seconds(obs_time)
    finite_t = np.isfinite(t_sec)
    if finite_t.sum() == 0:
        return x_offset, y_offset, lat, lon

    # Altitude indices with finite time, sorted by time
    idx_t = np.where(finite_t)[0]
    order = idx_t[np.argsort(t_sec[idx_t])]

    # Single-timestamp case: only the earliest-time bin is anchored.
    if len(order) == 1:
        x_offset[order[0]] = 0.0
        y_offset[order[0]] = 0.0
        if np.isfinite(launch_lat) and np.isfinite(launch_lon):
            lat[order[0]] = launch_lat
            lon[order[0]] = launch_lon
        return x_offset, y_offset, lat, lon

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    # Offsets in time-sorted order
    x_ord = np.full(len(order), np.nan)
    y_ord = np.full(len(order), np.nan)
    x_ord[0] = 0.0
    y_ord[0] = 0.0

    last_u = u_arr[order[0]] if np.isfinite(u_arr[order[0]]) else np.nan
    last_v = v_arr[order[0]] if np.isfinite(v_arr[order[0]]) else np.nan
    gap_budget = 0.0
    aborted = False

    for i in range(1, len(order)):
        if aborted:
            break

        k_prev, k_curr = order[i - 1], order[i]
        dt = t_sec[k_curr] - t_sec[k_prev]

        u_prev_raw = u_arr[k_prev]
        v_prev_raw = v_arr[k_prev]
        u_curr_raw = u_arr[k_curr]
        v_curr_raw = v_arr[k_curr]

        u_prev = u_prev_raw if np.isfinite(u_prev_raw) else last_u
        v_prev = v_prev_raw if np.isfinite(v_prev_raw) else last_v
        u_curr = u_curr_raw if np.isfinite(u_curr_raw) else last_u
        v_curr = v_curr_raw if np.isfinite(v_curr_raw) else last_v

        # Cannot integrate if we have never seen a valid wind
        if not (np.isfinite(u_prev) and np.isfinite(v_prev)
                and np.isfinite(u_curr) and np.isfinite(v_curr)):
            # Leave x_ord[i] / y_ord[i] NaN, but keep the anchor for later bins
            # that may recover wind data by updating last_u/last_v below.
            if np.isfinite(u_curr_raw):
                last_u = u_curr_raw
            if np.isfinite(v_curr_raw):
                last_v = v_curr_raw
            continue

        dx = 0.5 * (u_prev + u_curr) * dt
        dy = 0.5 * (v_prev + v_curr) * dt

        # Accumulate into the last known offset (skip over any NaN steps)
        last_x, last_y = np.nan, np.nan
        for j in range(i - 1, -1, -1):
            if np.isfinite(x_ord[j]):
                last_x, last_y = x_ord[j], y_ord[j]
                break
        if not np.isfinite(last_x):
            # Nothing to anchor to yet
            continue

        x_ord[i] = last_x + dx
        y_ord[i] = last_y + dy

        # Gap budget tracks drift accumulated using a carried-forward wind
        step_extrapolated = not (np.isfinite(u_curr_raw) and np.isfinite(v_curr_raw)
                                  and np.isfinite(u_prev_raw) and np.isfinite(v_prev_raw))
        if step_extrapolated:
            gap_budget += np.hypot(dx, dy)
            if gap_budget > max_gap_m:
                aborted = True
        else:
            gap_budget = 0.0

        if np.isfinite(u_curr_raw):
            last_u = u_curr_raw
        if np.isfinite(v_curr_raw):
            last_v = v_curr_raw

    # Un-sort back onto the altitude grid
    x_offset[order] = x_ord
    y_offset[order] = y_ord

    # Meters → degrees using small-angle spherical correction at launch_lat
    if np.isfinite(launch_lat) and np.isfinite(launch_lon):
        deg_per_rad = 180.0 / np.pi
        cos_phi0 = np.cos(np.deg2rad(launch_lat))
        if abs(cos_phi0) < 1e-6:
            cos_phi0 = np.sign(cos_phi0) * 1e-6 if cos_phi0 != 0 else 1e-6
        lat[:] = launch_lat + deg_per_rad * y_offset / R_earth
        lon[:] = launch_lon + deg_per_rad * x_offset / (R_earth * cos_phi0)

    return x_offset, y_offset, lat, lon

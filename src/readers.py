"""
Data readers for 13 sonde datasets.

Each read_* function takes a data directory path and returns a list of
profile dicts with standardized keys and units:

    sonde_id   : str
    launch_time: numpy datetime64 or None
    launch_lat : float (degrees N)
    launch_lon : float (degrees E)
    altitude   : ndarray [m] (geometric altitude above MSL for all datasets;
                               IGRA geopotential height is converted to geometric)
    p          : ndarray [Pa]
    T          : ndarray [K]
    RH         : ndarray [%] (0--100)
    u          : ndarray [m/s]
    v          : ndarray [m/s]

Missing values are np.nan.  All arrays share the same length for a
given profile.

QC filtering follows the spec (doc/regridding.tex).
"""

import glob
import io
import os
import re
import zipfile
from datetime import datetime

import numpy as np
import xarray as xr


MISSING = -999.0
MISSING_ICT = -9999.0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _wind_components(wspd, wdir_deg):
    """Convert wind speed and meteorological direction to u, v."""
    wdir_rad = np.deg2rad(wdir_deg)
    u = -wspd * np.sin(wdir_rad)
    v = -wspd * np.cos(wdir_rad)
    return u, v


def _replace_missing(arr, sentinel):
    """Replace sentinel values with NaN in-place."""
    arr = np.asarray(arr, dtype=np.float64)
    arr[arr <= sentinel] = np.nan
    return arr


def _first_finite(arr):
    """Return the first finite value in arr, or NaN."""
    finite = np.isfinite(arr)
    if np.any(finite):
        return float(arr[finite][0])
    return np.nan


def _first_obs_position(time, lat, lon):
    """Return (lat, lon) at the earliest-time sample where all three are finite.

    Used to derive ``launch_lat``/``launch_lon`` from the sonde's own GPS
    stream rather than from provider metadata, which for some datasets
    (notably HALO-(AC)3) can disagree with the first GPS fix by several km.
    Works for both time-ascending (EOL/FRD) and time-descending
    (surface-first NetCDF like JOANNE, HALO-(AC)3) orderings.  Positive
    out-of-range sentinels (e.g. lat = 999, lon = 99 in old hurricane
    EOL files) are rejected along with negative sentinels and NaN.
    Falls back to (nan, nan) if no valid (time, lat, lon) triple exists.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    time = np.asarray(time)
    if time.dtype.kind == "M":
        t_f = np.where(np.isnat(time), np.nan,
                        time.astype("datetime64[ns]").astype(np.int64).astype(np.float64))
    else:
        t_f = np.asarray(time, dtype=np.float64)
    in_range = (np.abs(lat) <= 90.0) & (np.abs(lon) <= 360.0)
    valid = np.isfinite(t_f) & np.isfinite(lat) & np.isfinite(lon) & in_range
    if not valid.any():
        return np.nan, np.nan
    k = np.where(valid)[0][np.argmin(t_f[valid])]
    return float(lat[k]), float(lon[k])


# Mean Earth radius [m] for geopotential ↔ geometric height conversion
_RE = 6_371_000.0


def _geopotential_to_geometric(gph):
    """Convert geopotential height [m] to geometric height [m].

    z = R_e * H / (R_e - H), where H is geopotential height.
    The difference is ~0.3% at 30 km.
    """
    gph = np.asarray(gph, dtype=np.float64)
    return _RE * gph / (_RE - gph)


# ---------------------------------------------------------------------------
#  JOANNE  (EUREC4A, Level 2 NetCDFs)
# ---------------------------------------------------------------------------

def read_joanne(data_dir):
    """Read JOANNE Level 2 dropsonde profiles.

    The Level 2 directory contains one NetCDF per sonde.  These files
    are already filtered to the 'good' quality tier (1068 of 1215).
    Units: ta [K], p [Pa], rh [0--1], alt [m], wspd/wdir.
    """
    pattern = os.path.join(data_dir, "Level_2", "*.nc")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        ds = xr.open_dataset(fpath)

        altitude = ds["alt"].values.astype(np.float64)
        p = ds["p"].values.astype(np.float64)            # Pa
        T = ds["ta"].values.astype(np.float64)            # K
        rh_frac = ds["rh"].values.astype(np.float64)      # 0--1
        wspd = ds["wspd"].values.astype(np.float64)
        wdir = ds["wdir"].values.astype(np.float64)
        u, v = _wind_components(wspd, wdir)

        # Launch position: prefer the earliest-time per-level GPS fix
        # over the aircraft-centroid attribute, which can be off by
        # several km for some providers.  See doc/regridding.tex §3.3.
        if "lat" in ds and "lon" in ds:
            launch_lat, launch_lon = _first_obs_position(
                ds["time"].values, ds["lat"].values, ds["lon"].values)
        else:
            launch_lat = float(ds.attrs.get("aircraft_latitude_(deg_N)", np.nan))
            launch_lon = float(ds.attrs.get("aircraft_longitude_(deg_E)", np.nan))

        profiles.append({
            "sonde_id": str(ds["sonde_id"].values),
            "launch_time": ds["time"].values[0],
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": ds["time"].values,
            "p": p,
            "T": T,
            "RH": rh_frac * 100.0,  # convert to %
            "u": u,
            "v": v,
        })
        ds.close()

    return profiles


# ---------------------------------------------------------------------------
#  BEACH  (PERCUSION, Level 2 Zarrs)
# ---------------------------------------------------------------------------

# BEACH sonde_qc encoding: 0=GOOD, 1=BAD, 2=UGLY
# 17 misconfigured sondes (supplement Table E2 of Gloeckner et al. 2025)
# and factory-reset sonde 0bd0e322 are already excluded by sonde_qc != 0.
# Verified: 0bd0e322 has sonde_qc=1; all 17+1 fall within the 145 non-GOOD
# sondes (14 BAD + 131 UGLY out of 1121 total zarr stores).

def read_beach(data_dir):
    """Read BEACH Level 2 dropsonde profiles (Zarr).

    QC: only sonde_qc == 0 ('GOOD') profiles are retained.
    Units: p [Pa], ta [K], rh [0--1 fractional], alt [m], u/v [m/s].
    """
    pattern = os.path.join(data_dir, "Level_2", "*", "*.zarr")
    zarrs = sorted(glob.glob(pattern))
    profiles = []

    for zpath in zarrs:
        try:
            ds = xr.open_dataset(zpath, engine="zarr", consolidated=False)
        except Exception:
            continue

        # QC filter: only GOOD sondes (0=GOOD, 1=BAD, 2=UGLY)
        if "sonde_qc" in ds and int(ds["sonde_qc"].values) != 0:
            ds.close()
            continue

        # Skip incomplete stores (partial download)
        required = {"alt", "p", "ta", "rh", "u", "v"}
        if not required.issubset(ds.data_vars):
            ds.close()
            continue

        altitude = ds["alt"].values.astype(np.float64)
        p = ds["p"].values.astype(np.float64)            # Pa
        T = ds["ta"].values.astype(np.float64)            # K
        rh_frac = ds["rh"].values.astype(np.float64)      # 0--1
        u = ds["u"].values.astype(np.float64)
        v = ds["v"].values.astype(np.float64)

        # Launch position: earliest-time per-level GPS (§3.3)
        if "lat" in ds.coords and "lon" in ds.coords and "time" in ds.coords:
            launch_lat, launch_lon = _first_obs_position(
                ds.coords["time"].values, ds["lat"].values, ds["lon"].values)
        else:
            launch_lat = _first_finite(ds["lat"].values) if "lat" in ds.coords else np.nan
            launch_lon = _first_finite(ds["lon"].values) if "lon" in ds.coords else np.nan

        profiles.append({
            "sonde_id": str(ds["sonde_id"].values),
            "launch_time": ds["launch_time"].values if "launch_time" in ds else None,
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": ds.coords["time"].values if "time" in ds.coords else None,
            "p": p,
            "T": T,
            "RH": rh_frac * 100.0,  # convert to %
            "u": u,
            "v": v,
        })
        ds.close()

    return profiles


# ---------------------------------------------------------------------------
#  OTREC  (NetCDFs, AVAPS-processed)
# ---------------------------------------------------------------------------

# One warm-biased sounding to exclude per spec.
OTREC_EXCLUDE = {"20190925_154412"}


def read_otrec(data_dir):
    """Read OTREC dropsonde profiles.

    Units: pres [hPa], tdry [°C], rh [%], gpsalt [m], u/v from wspd/wdir.
    """
    pattern = os.path.join(data_dir, "*.nc")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        # Check exclusion list by filename timestamp
        fname = os.path.basename(fpath)
        timestamp = fname.split("_v1_")[1].replace(".nc", "") if "_v1_" in fname else ""
        # Pattern: OTREC_AVAPS_NRD41_v1_YYYYMMDD_HHMMSS.nc
        parts = fname.replace(".nc", "").split("_")
        if len(parts) >= 6:
            timestamp = parts[4] + "_" + parts[5]
        if timestamp in OTREC_EXCLUDE:
            continue

        ds = xr.open_dataset(fpath)

        altitude = ds["gpsalt"].values.astype(np.float64)
        p = ds["pres"].values.astype(np.float64) * 100.0    # hPa → Pa
        T = ds["tdry"].values.astype(np.float64) + 273.15   # °C → K
        rh = ds["rh"].values.astype(np.float64)              # %
        wspd = ds["wspd"].values.astype(np.float64)
        wdir = ds["wdir"].values.astype(np.float64)
        u, v = _wind_components(wspd, wdir)

        launch_time = ds["launch_time"].values
        # Launch position: earliest-time per-level GPS (§3.3)
        lat = ds["lat"].values.astype(np.float64)
        lon = ds["lon"].values.astype(np.float64)
        if "time" in ds:
            launch_lat, launch_lon = _first_obs_position(
                ds["time"].values, lat, lon)
        else:
            launch_lat = float(lat[np.isfinite(lat)][0]) if np.any(np.isfinite(lat)) else np.nan
            launch_lon = float(lon[np.isfinite(lon)][0]) if np.any(np.isfinite(lon)) else np.nan

        profiles.append({
            "sonde_id": fname.replace(".nc", ""),
            "launch_time": launch_time,
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": ds["time"].values if "time" in ds else None,
            "p": p,
            "T": T,
            "RH": rh,
            "u": u,
            "v": v,
        })
        ds.close()

    return profiles


# ---------------------------------------------------------------------------
#  ACTIVATE  (ICT text files)
# ---------------------------------------------------------------------------

# Soundings where the humidity sensor was not reconditioned before flight
# (Table 6 of Vömel et al. 2023, doi:10.1038/s41597-023-02647-5).
# RH is excluded for these profiles; T, p, and wind are retained.
# Timestamps are truncated to minutes to match ICT filenames.
ACTIVATE_UNCONDITIONED_RH = {
    "201912161840",  # 20191216_184052
    "201912161913",  # 20191216_191320
    "201912161920",  # 20191216_192040
    "201912161923",  # 20191216_192316
    "202002151725",  # 20200215_172514
    "202002151855",  # 20200215_185506
    "202003111342",  # 20200311_134234
    "202105142102",  # 20210514_210237
    "202106021845",  # 20210602_184503
    "202106021850",  # 20210602_185004
    "202106071840",  # 20210607_184023
    "202106071935",  # 20210607_193522
    "202106281435",  # 20210628_143502
    "202111301715",  # 20211130_171557
    "202111301743",  # 20211130_174337
    "202112071852",  # 20211207_185240
    "202205311335",  # 20220531_133514
    "202206101806",  # 20220610_180657
    "202206111819",  # 20220611_181926
    "202206111933",  # 20220611_193332
}


def read_activate(data_dir):
    """Read ACTIVATE dropsonde profiles (ICT format).

    QC: profiles whose operator comment is not 'Good Drop' are excluded.
    RH is excluded for sondes with unreconditioned humidity sensors (Table 6).
    Units: Pressure [mb], Temperature [°C], RH [%], GPS Altitude [m],
           Uwnd/Vwnd [m/s].  Missing = -9999.
    """
    pattern = os.path.join(data_dir, "*.ict")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        with open(fpath) as fh:
            lines = fh.readlines()

        n_header = int(lines[0].split(",")[0])

        # Extract metadata from header
        sonde_id = os.path.basename(fpath).replace(".ict", "")
        operator_comment = ""
        launch_lat = np.nan
        launch_lon = np.nan
        launch_time = None
        for line in lines[:n_header]:
            if "Operator comments" in line:
                operator_comment = line.split(":", 1)[1].strip()
            if "Latitude (deg)" in line:
                try:
                    launch_lat = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            if "Longitude (deg)" in line:
                try:
                    launch_lon = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            if "Launch Time" in line:
                lt_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
                if lt_match:
                    launch_time = np.datetime64(
                        datetime.strptime(lt_match.group(1), "%Y-%m-%d %H:%M:%S")
                    )

        # QC: exclude non-"Good Drop" profiles
        if "Good Drop" not in operator_comment:
            continue

        # Column names are on line n_header-1
        cols = lines[n_header - 1].strip().split(",")
        col_idx = {name.strip(): i for i, name in enumerate(cols)}

        data = np.genfromtxt(
            io.StringIO("".join(lines[n_header:])),
            delimiter=",",
            missing_values="-9999",
            filling_values=np.nan,
        )
        if data.ndim == 1:
            continue  # single row, skip

        altitude = _replace_missing(data[:, col_idx["GPS Altitude"]], MISSING_ICT)
        p = _replace_missing(data[:, col_idx["Pressure"]], MISSING_ICT) * 100.0  # mb → Pa
        T = _replace_missing(data[:, col_idx["Temperature"]], MISSING_ICT) + 273.15  # °C → K
        rh = _replace_missing(data[:, col_idx["RH"]], MISSING_ICT)
        u = _replace_missing(data[:, col_idx["Uwnd"]], MISSING_ICT)
        v = _replace_missing(data[:, col_idx["Vwnd"]], MISSING_ICT)

        # Fallback: parse launch time from filename if not found in header
        if launch_time is None:
            match = re.search(r"(\d{12})", os.path.basename(fpath))
            if match:
                launch_time = np.datetime64(datetime.strptime(match.group(1), "%Y%m%d%H%M"))

        # Exclude RH for sondes with unreconditioned humidity sensors
        ts_match = re.search(r"(\d{12})", os.path.basename(fpath))
        if ts_match and ts_match.group(1) in ACTIVATE_UNCONDITIONED_RH:
            rh = None

        # Prefer earliest-time per-level GPS fix over header; fall back to
        # header when the data column is missing.  See doc/regridding.tex §3.3.
        if "Latitude" in col_idx and "Longitude" in col_idx and "Time_Start" in col_idx:
            lat_col = _replace_missing(data[:, col_idx["Latitude"]], MISSING_ICT)
            lon_col = _replace_missing(data[:, col_idx["Longitude"]], MISSING_ICT)
            t_col = _replace_missing(data[:, col_idx["Time_Start"]], MISSING_ICT)
            ll_lat, ll_lon = _first_obs_position(t_col, lat_col, lon_col)
            if np.isfinite(ll_lat) and np.isfinite(ll_lon):
                launch_lat, launch_lon = ll_lat, ll_lon

        # Observation time: Time_Start is UTC seconds from midnight.
        # Combine with launch date to get absolute datetime.
        obs_time = None
        if "Time_Start" in col_idx and launch_time is not None:
            secs = _replace_missing(data[:, col_idx["Time_Start"]], MISSING_ICT)
            launch_date = launch_time.astype("datetime64[D]")
            obs_time = launch_date + (secs * 1e9).astype("timedelta64[ns]")

        profiles.append({
            "sonde_id": sonde_id,
            "launch_time": launch_time,
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": obs_time,
            "p": p,
            "T": T,
            "RH": rh,
            "u": u,
            "v": v,
        })

    return profiles


# ---------------------------------------------------------------------------
#  EOL sounding format (shared by SHOUT, Hurricane, AR Recon)
# ---------------------------------------------------------------------------

def _parse_eol(fpath):
    """Parse an EOL Sounding Format 1.1 file.

    Returns a dict with standardized keys, or None if unparseable.
    The format has a multi-line header terminated by '------', then
    whitespace-delimited data columns.
    """
    with open(fpath) as fh:
        lines = fh.readlines()

    # Parse header for metadata
    sonde_id = os.path.basename(fpath)
    launch_time = None
    launch_lat = np.nan
    launch_lon = np.nan
    operator_comment = ""

    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("------"):
            header_end = i + 1
        if "UTC Launch Time" in line:
            # Format: 2016, 08, 30, 02:55:24
            match = re.search(r"(\d{4}),\s*(\d{2}),\s*(\d{2}),\s*(\d{2}):(\d{2}):(\d{2})", line)
            if match:
                y, mo, d, h, mi, s = (int(x) for x in match.groups())
                launch_time = np.datetime64(datetime(y, mo, d, h, mi, s))
        if "Launch Location" in line:
            # Format: 58 50.28'W -58.837930, 27 53.59'N 27.893248, 17416.50
            # Decimal degrees are the 2nd and 4th floats (after DMS minutes)
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if len(nums) >= 4:
                launch_lon = float(nums[1])
                launch_lat = float(nums[3])
        if "Operator/Comments" in line or "System Operator/Comments" in line:
            operator_comment = line.split("/")[-1].strip()

    if header_end == 0:
        return None

    # Find the column header line (just above the dashes)
    # Columns vary between datasets but always include Press, Temp, RH, Uwind, Vwind, GPSAlt
    col_line = ""
    for i in range(header_end - 1, -1, -1):
        if lines[i].strip().startswith("Time"):
            col_line = lines[i]
            break

    # Parse data — whitespace-delimited, -999 sentinel
    data_lines = []
    for line in lines[header_end:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("/"):
            continue
        data_lines.append(stripped)

    if not data_lines:
        return None

    data = np.genfromtxt(
        io.StringIO("\n".join(data_lines)),
        invalid_raise=False,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 5:
        return None

    # EOL column indices (standard across SHOUT/Hurricane/ARRecon):
    # 0:Time 1:hh 2:mm 3:ss 4:Press 5:Temp 6:Dewpt 7:RH 8:Uwind 9:Vwind
    # 10:Wspd 11:Dir 12:dZ 13:GeoPoAlt 14:Lon 15:Lat 16:GPSAlt
    # Hurricane adds: 17:Radius 18:Azimuth 19:Wwind 20:Wwind_f
    ncols = data.shape[1]

    # Prefer GPS altitude (col 16); fall back to geopotential altitude (col 13)
    if ncols > 16:
        altitude = _replace_missing(data[:, 16], MISSING)
        geopot = _replace_missing(data[:, 13], MISSING)
        no_gps = ~np.isfinite(altitude)
        altitude[no_gps] = geopot[no_gps]
    else:
        altitude = _replace_missing(data[:, 13], MISSING)
    p = _replace_missing(data[:, 4], MISSING) * 100.0    # mb → Pa
    T = _replace_missing(data[:, 5], MISSING) + 273.15   # °C → K
    rh = _replace_missing(data[:, 7], MISSING)            # %
    u = _replace_missing(data[:, 8], MISSING)             # m/s
    v = _replace_missing(data[:, 9], MISSING)             # m/s

    # Prefer earliest-time per-level GPS over header launch location.
    # Column 0 is elapsed time from launch (can be slightly negative for
    # the pre-release GPS fix), so use it as the anchor time.
    if ncols > 15:
        lat_col = _replace_missing(data[:, 15], MISSING)
        lon_col = _replace_missing(data[:, 14], MISSING)
        t_col = _replace_missing(data[:, 0], MISSING)
        ll_lat, ll_lon = _first_obs_position(t_col, lat_col, lon_col)
        if np.isfinite(ll_lat) and np.isfinite(ll_lon):
            launch_lat, launch_lon = ll_lat, ll_lon
    # Final sanity: reject positive-sentinel header values (e.g. 999/99)
    if not (np.isfinite(launch_lat) and abs(launch_lat) <= 90):
        launch_lat = np.nan
    if not (np.isfinite(launch_lon) and abs(launch_lon) <= 360):
        launch_lon = np.nan

    # Observation time: hh (col 1), mm (col 2), ss (col 3) → absolute datetime.
    # These are clock times (UTC), not elapsed. Combine with launch date.
    # Handle midnight rollover: if a sonde launches near midnight, some
    # observation clock times (e.g. 00:05) belong to the next calendar day.
    # Detect by checking for large backward jumps (>12 h before launch).
    obs_time = None
    if launch_time is not None:
        hh = _replace_missing(data[:, 1], MISSING)
        mm = _replace_missing(data[:, 2], MISSING)
        ss = _replace_missing(data[:, 3], MISSING)
        launch_date = launch_time.astype("datetime64[D]")
        secs = hh * 3600 + mm * 60 + ss
        obs_time = launch_date + (secs * 1e9).astype("timedelta64[ns]")
        # Correct midnight rollover: obs times >12 h before launch → next day
        rollover = (launch_time - obs_time) > np.timedelta64(12, "h")
        obs_time[rollover] += np.timedelta64(1, "D")

    return {
        "sonde_id": sonde_id,
        "launch_time": launch_time,
        "launch_lat": launch_lat,
        "launch_lon": launch_lon,
        "altitude": altitude,
        "obs_time": obs_time,
        "p": p,
        "T": T,
        "RH": rh,
        "u": u,
        "v": v,
        "operator_comment": operator_comment,
    }


# ---------------------------------------------------------------------------
#  SHOUT  (EOL format, Global Hawk)
# ---------------------------------------------------------------------------

def read_shout(data_dir):
    """Read SHOUT dropsonde profiles (EOL format).

    Units: Press [mb], Temp [°C], RH [%], GPSAlt [m], Uwind/Vwind [m/s].
    """
    pattern = os.path.join(data_dir, "*.eol")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        profile = _parse_eol(fpath)
        if profile is not None:
            profile.pop("operator_comment", None)
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  Hurricane  (EOL format with radius/azimuth, 1996--2012)
# ---------------------------------------------------------------------------

def read_hurricane(data_dir):
    """Read NOAA Hurricane dropsonde archive (EOL format).

    Files are in subdirectories by year/storm/aircraft.
    """
    pattern = os.path.join(data_dir, "**", "*.eol.radazm.Wwind")
    files = sorted(glob.glob(pattern, recursive=True))
    profiles = []

    for fpath in files:
        profile = _parse_eol(fpath)
        if profile is not None:
            profile.pop("operator_comment", None)
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  AR Recon  (FRD format, ASPEN-processed)
# ---------------------------------------------------------------------------

def _parse_frd(fpath):
    """Parse an ASPEN FRD file.

    Format is similar to EOL but with different column layout:
    IX  t(s)  P(mb)  T(C)  RH(%)  Z(m)  WD  WS(m/s)  U(m/s)  V(m/s)
    NS  WZ(m/s)  ZW(m)  FP  FT  FH  FW  LAT(N)  LON(E)
    """
    with open(fpath) as fh:
        lines = fh.readlines()

    sonde_id = os.path.basename(fpath)
    launch_time = None
    launch_lat = np.nan
    launch_lon = np.nan

    header_end = 0
    for i, line in enumerate(lines):
        # Header ends with the column label line followed by data
        if line.strip().startswith("IX"):
            header_end = i + 1
            break
        if "Date:" in line and "Lat:" in line:
            lat_match = re.search(r"Lat:\s+([\d.]+)\s*([NS])?", line)
            if lat_match:
                launch_lat = float(lat_match.group(1))
                if lat_match.group(2) == "S":
                    launch_lat = -launch_lat
        if "Time:" in line and "Lon:" in line:
            lon_match = re.search(r"Lon:\s+([\d.]+)\s*([EW])?", line)
            if lon_match:
                launch_lon = float(lon_match.group(1))
                if lon_match.group(2) == "W":
                    launch_lon = -launch_lon
        if "Date:" in line and "Time:" in line:
            date_match = re.search(r"Date:\s+(\d{6})", line)
            time_match = re.search(r"Time:\s+(\d{6})", line)
            if date_match and time_match:
                dt_str = date_match.group(1) + time_match.group(1)
                launch_time = np.datetime64(datetime.strptime(dt_str, "%y%m%d%H%M%S"))
        if "COMMENT:" in line:
            comment = line.split("COMMENT:")[1].strip()

    if header_end == 0:
        return None

    data_lines = []
    for line in lines[header_end:]:
        stripped = line.strip()
        if not stripped:
            continue
        data_lines.append(stripped)

    if not data_lines:
        return None

    data = np.genfromtxt(
        io.StringIO("\n".join(data_lines)),
        invalid_raise=False,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 5:
        return None

    # FRD columns:
    # 0:IX 1:t 2:P(mb) 3:T(C) 4:RH(%) 5:Z(m) 6:WD 7:WS 8:U 9:V
    # 10:NS 11:WZ 12:ZW 13:FP 14:FT 15:FH 16:FW 17:LAT 18:LON
    altitude = _replace_missing(data[:, 5], MISSING)        # Z in meters
    p = _replace_missing(data[:, 2], MISSING) * 100.0       # mb → Pa
    T = _replace_missing(data[:, 3], MISSING) + 273.15      # °C → K
    rh = _replace_missing(data[:, 4], MISSING)              # %
    u = _replace_missing(data[:, 8], MISSING)               # m/s
    v = _replace_missing(data[:, 9], MISSING)               # m/s

    # Prefer earliest-time per-level GPS over FRD header launch location.
    # Column 1 is elapsed time from launch in seconds.
    ncols = data.shape[1]
    if ncols > 18:
        lat_col = _replace_missing(data[:, 17], MISSING)
        lon_col = _replace_missing(data[:, 18], MISSING)
        t_col = _replace_missing(data[:, 1], MISSING)
        ll_lat, ll_lon = _first_obs_position(t_col, lat_col, lon_col)
        if np.isfinite(ll_lat) and np.isfinite(ll_lon):
            launch_lat, launch_lon = ll_lat, ll_lon
    # Final sanity: reject positive-sentinel header values (e.g. 999/99)
    if not (np.isfinite(launch_lat) and abs(launch_lat) <= 90):
        launch_lat = np.nan
    if not (np.isfinite(launch_lon) and abs(launch_lon) <= 360):
        launch_lon = np.nan

    # Observation time: column 1 is elapsed seconds from launch
    obs_time = None
    if launch_time is not None:
        elapsed_s = _replace_missing(data[:, 1], MISSING)
        obs_time = launch_time + (elapsed_s * 1e9).astype("timedelta64[ns]")

    return {
        "sonde_id": sonde_id,
        "launch_time": launch_time,
        "launch_lat": launch_lat,
        "launch_lon": launch_lon,
        "altitude": altitude,
        "obs_time": obs_time,
        "p": p,
        "T": T,
        "RH": rh,
        "u": u,
        "v": v,
    }


def read_arrecon(data_dir):
    """Read AR Recon dropsonde profiles (FRD format).

    Files are in subdirectories: YYYY/IOP*/AIRCRAFT/*.frd
    """
    pattern = os.path.join(data_dir, "**", "*QC.frd")
    files = sorted(glob.glob(pattern, recursive=True))
    profiles = []

    for fpath in files:
        profile = _parse_frd(fpath)
        if profile is not None:
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  HALO-(AC)³  (Level 2 NetCDFs, PANGAEA)
# ---------------------------------------------------------------------------

HALOAC3_EXCLUDE = {
    "163313035",  # GPS receiver failure, no altitude data
}


def read_haloac3(data_dir):
    """Read HALO-(AC)³ Level 2 dropsonde profiles (NetCDF).

    Units: p [Pa], ta [K], rh [0--1 fractional], gpsalt [m], u/v [m/s].
    """
    pattern = os.path.join(data_dir, "Level_2", "*.nc")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        ds = xr.open_dataset(fpath)

        sonde_id = str(ds["sonde_id"].values)
        if sonde_id in HALOAC3_EXCLUDE:
            ds.close()
            continue

        altitude = ds["gpsalt"].values.astype(np.float64)
        p = ds["p"].values.astype(np.float64)            # Pa
        T = ds["ta"].values.astype(np.float64)            # K
        rh_frac = ds["rh"].values.astype(np.float64)      # 0--1
        u = ds["u"].values.astype(np.float64)
        v = ds["v"].values.astype(np.float64)

        # Launch position: earliest-time per-level GPS (§3.3).  HALO-(AC)3
        # stores its aircraft-centroid attribute that can be ~15 km off
        # from the first GPS fix.
        if "lat" in ds and "lon" in ds and "time" in ds:
            launch_lat, launch_lon = _first_obs_position(
                ds["time"].values, ds["lat"].values, ds["lon"].values)
        else:
            launch_lat = float(ds.attrs.get("aircraft_latitude_(deg_N)", np.nan))
            launch_lon = float(ds.attrs.get("aircraft_longitude_(deg_E)", np.nan))
            if launch_lat > 90 or launch_lat < -90:
                launch_lat = np.nan
            if launch_lon > 360 or launch_lon < -360:
                launch_lon = np.nan

        profiles.append({
            "sonde_id": str(ds["sonde_id"].values),
            "launch_time": ds["time"].values[-1],  # time array is surface-first; launch is last
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": ds["time"].values,
            "p": p,
            "T": T,
            "RH": rh_frac * 100.0,  # convert to %
            "u": u,
            "v": v,
        })
        ds.close()

    return profiles


# ---------------------------------------------------------------------------
#  ENRR  (G-IV FRD + C-130 FRD + Global Hawk EOL)
# ---------------------------------------------------------------------------

def read_enrr(data_dir):
    """Read ENRR dropsonde profiles from all three platforms.

    G-IV corrected profiles (FRD): corrected/*/*.frd
    C-130 dry-bias-corrected profiles (FRD): c130_drybiascor/*.frd
    Global Hawk corrected profiles (EOL): globalhawk_corrected/*/*.eol
    """
    profiles = []

    # G-IV: FRD files in corrected/YYYYMMDD_RFNN/ subdirectories
    giv_pattern = os.path.join(data_dir, "corrected", "**", "*.frd")
    for fpath in sorted(glob.glob(giv_pattern, recursive=True)):
        profile = _parse_frd(fpath)
        if profile is not None:
            profiles.append(profile)

    # C-130: FRD files in c130_drybiascor/ (flat or in subdirectories)
    c130_pattern = os.path.join(data_dir, "c130_drybiascor", "**", "*.frd")
    for fpath in sorted(glob.glob(c130_pattern, recursive=True)):
        profile = _parse_frd(fpath)
        if profile is not None:
            profiles.append(profile)

    # Global Hawk: EOL files in globalhawk_corrected/YYYYMMDD_RFNN/
    gh_pattern = os.path.join(data_dir, "globalhawk_corrected", "**", "*.eol")
    for fpath in sorted(glob.glob(gh_pattern, recursive=True)):
        profile = _parse_eol(fpath)
        if profile is not None:
            profile.pop("operator_comment", None)
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  HS3  (EOL format, Global Hawk, by year)
# ---------------------------------------------------------------------------

def read_hs3(data_dir):
    """Read HS3 dropsonde profiles (EOL format).

    Files are in subdirectories by year and basin: YYYY/{Gulf,Pacific}/*.eol
    """
    pattern = os.path.join(data_dir, "**", "*.eol")
    files = sorted(glob.glob(pattern, recursive=True))
    profiles = []

    for fpath in files:
        profile = _parse_eol(fpath)
        if profile is not None:
            profile.pop("operator_comment", None)
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  PREDICT  (EOL format, NSF/NCAR GV)
# ---------------------------------------------------------------------------

# Profiles with failed sensors (no usable pressure or temperature).
PREDICT_EXCLUDE = {
    "D20100815_113933_P.QC.eol",  # bad pressure sensor (operator: "no good")
    "D20100906_162944_P.QC.eol",  # no temperature, lost telemetry
    "D20100928_155055_P.QC.eol",  # lost temperature at launch
}


def read_predict(data_dir):
    """Read PREDICT dropsonde profiles (EOL format).

    Files are in the top-level data directory: *.eol
    Three profiles with failed sensors are excluded.
    """
    pattern = os.path.join(data_dir, "*.eol")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        if os.path.basename(fpath) in PREDICT_EXCLUDE:
            continue
        profile = _parse_eol(fpath)
        if profile is not None:
            profile.pop("operator_comment", None)
            profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
#  DYNAMO  (NetCDFs, dropsondes only)
# ---------------------------------------------------------------------------

def read_dynamo(data_dir):
    """Read DYNAMO dropsonde profiles (NetCDF).

    Per spec, only dropsonde profiles are used (not rawinsonde).
    Units: pres [hPa], tdry [°C], rh [%], gpsalt [m], u/v [m/s].
    Missing sentinel: -9999.
    """
    pattern = os.path.join(data_dir, "*.nc")
    files = sorted(glob.glob(pattern))
    profiles = []

    for fpath in files:
        ds = xr.open_dataset(fpath)

        altitude = ds["gpsalt"].values.astype(np.float64)
        p_hpa = ds["pres"].values.astype(np.float64)
        T_c = ds["tdry"].values.astype(np.float64)
        rh = ds["rh"].values.astype(np.float64)
        wspd = ds["wspd"].values.astype(np.float64)
        wdir = ds["wdir"].values.astype(np.float64)

        # Replace sentinel values
        for arr in [altitude, p_hpa, T_c, rh, wspd, wdir]:
            arr[arr <= -999] = np.nan

        p = p_hpa * 100.0       # hPa → Pa
        T = T_c + 273.15        # °C → K
        u, v = _wind_components(wspd, wdir)

        # Parse launch time from filename: aircraft.dgar.p3.dropsonde.YYYYMMDD_HHMMSS.nc
        fname = os.path.basename(fpath)
        match = re.search(r"(\d{8}_\d{6})", fname)
        launch_time = None
        if match:
            launch_time = np.datetime64(datetime.strptime(match.group(1), "%Y%m%d_%H%M%S"))

        # Launch position: earliest-time per-level GPS (§3.3)
        lat = ds["lat"].values.astype(np.float64)
        lon = ds["lon"].values.astype(np.float64)
        lat[lat <= -999] = np.nan
        lon[lon <= -999] = np.nan
        # DYNAMO stores a (0, 0) sentinel for missing GPS in some rows
        zero_sent = (lat == 0.0) & (lon == 0.0)
        lat[zero_sent] = np.nan
        lon[zero_sent] = np.nan
        if "time" in ds:
            launch_lat, launch_lon = _first_obs_position(
                ds["time"].values, lat, lon)
        else:
            launch_lat = float(lat[np.isfinite(lat)][0]) if np.any(np.isfinite(lat)) else np.nan
            launch_lon = float(lon[np.isfinite(lon)][0]) if np.any(np.isfinite(lon)) else np.nan

        # Observation time from Hour/Min/Sec UTC clock variables.
        # (time_offset metadata says "hour" but values are actually seconds;
        # using H/M/S directly is unambiguous.)
        obs_time = None
        if launch_time is not None and "Hour" in ds and "Min" in ds and "Sec" in ds:
            h = ds["Hour"].values.astype(np.float64)
            mn = ds["Min"].values.astype(np.float64)
            sc = ds["Sec"].values.astype(np.float64)
            for arr in [h, mn, sc]:
                arr[arr <= -999] = np.nan
            # H=0, M=0, S=0 entries are end-of-data sentinels
            sentinel = (h == 0) & (mn == 0) & (sc == 0)
            h[sentinel] = np.nan
            total_sec = h * 3600 + mn * 60 + sc
            launch_date = np.datetime64(
                str(launch_time)[:10], "ns"
            )
            obs_time = launch_date + (total_sec * 1e9).astype("timedelta64[ns]")
            # Mask invalid entries
            invalid = ~(np.isfinite(h) & np.isfinite(mn) & np.isfinite(sc))
            obs_time[invalid] = np.datetime64("NaT")

        profiles.append({
            "sonde_id": fname.replace(".nc", ""),
            "launch_time": launch_time,
            "launch_lat": launch_lat,
            "launch_lon": launch_lon,
            "altitude": altitude,
            "obs_time": obs_time,
            "p": p,
            "T": T,
            "RH": rh,
            "u": u,
            "v": v,
        })
        ds.close()

    return profiles


# ---------------------------------------------------------------------------
#  IGRA  (zipped fixed-width text, radiosondes)
# ---------------------------------------------------------------------------

def _parse_igra_value(text):
    """Parse an IGRA fixed-width integer field, stripping A/B flags."""
    text = text.strip()
    if not text or text == "-9999" or text == "-8888":
        return np.nan
    # Strip trailing flag character (A or B)
    if text[-1] in ("A", "B"):
        text = text[:-1]
    if not text or text.isspace():
        return np.nan
    try:
        return int(text)
    except ValueError:
        return np.nan


def read_igra(data_dir, year=None, year_min=2000, year_max=2025, subsample=1,
              stations=None):
    """Read IGRA v2 radiosonde profiles from zipped station files.

    Parameters
    ----------
    data_dir : str
        Path to directory containing *-data.txt.zip files.
    year : int, optional
        If given, only soundings from this year are retained (overrides
        year_min/year_max).
    year_min, year_max : int
        Year range (inclusive).  Ignored if year is set.
    subsample : int
        Keep only soundings whose day-of-year satisfies
        (DOY - 1) % subsample == 0.  Default 1 keeps all soundings.
    stations : list of str, optional
        If given, only read these station IDs.  Skips all other zip files.

    IGRA stores:
      pressure   [Pa]
      GPH        [m]  (geopotential height, converted to geometric altitude)
      temperature[tenths of °C]
      RH         [tenths of %]
      wind dir   [degrees]
      wind speed [tenths of m/s]

    All converted to standard units on output.
    """
    if year is not None:
        year_min = year
        year_max = year

    stations_set = set(stations) if stations is not None else None

    pattern = os.path.join(data_dir, "*-data.txt.zip")
    zips = sorted(glob.glob(pattern))
    profiles = []

    for zpath in zips:
        station_id = os.path.basename(zpath).split("-data")[0]

        if stations_set is not None and station_id not in stations_set:
            continue

        with zipfile.ZipFile(zpath) as zf:
            txt_name = zf.namelist()[0]
            with zf.open(txt_name) as f:
                raw = f.read().decode("ascii", errors="replace")

        # Split into soundings by header lines
        lines = raw.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.startswith("#"):
                i += 1
                continue

            # Parse header
            # #ID         YEAR MO DY HR RELTIME NUMLEV ...  LAT     LON
            # Cols: 2-12  14-17 19-20 22-23 25-26 28-31 33-36 ...
            header = line
            try:
                year_val = int(header[13:17])
                month = int(header[18:20])
                day = int(header[21:23])
                hour = int(header[24:26])
                reltime_raw = header[27:31].strip()
                numlev = int(header[32:36])
                lat_raw = int(header[55:62].strip())
                lon_raw = int(header[63:71].strip())
            except (ValueError, IndexError):
                i += 1
                continue

            launch_lat = lat_raw / 10000.0
            launch_lon = lon_raw / 10000.0

            # Skip soundings outside year range
            if year_val < year_min or year_val > year_max:
                i += 1 + numlev
                continue

            # Subsample by day-of-year
            if subsample > 1:
                try:
                    doy = datetime(year_val, month, day).timetuple().tm_yday
                except ValueError:
                    i += 1 + numlev
                    continue
                if (doy - 1) % subsample != 0:
                    i += 1 + numlev
                    continue

            # IGRA HOUR field: 0-23 or 99 (missing). Not guaranteed synoptic —
            # off-synoptic ascents are valid IGRA records. Used here as a
            # fallback when RELTIME is missing, not as a nominal-synoptic
            # label.
            header_time = None
            if 0 <= hour <= 23:
                try:
                    header_time = np.datetime64(datetime(year_val, month, day, hour))
                except ValueError:
                    pass

            # Rollover anchor: only the synoptic HOUR values trigger the
            # pre-midnight-launch shift below. That convention is a
            # synoptic-labeling artifact (HR=00 assigned to a balloon released
            # ~23:xx the night before), not a documented off-synoptic pattern.
            synoptic_anchor = header_time if hour in (0, 6, 12, 18) else None

            # Actual launch time from RELTIME (HHMM format, 9999=missing).
            # RELTIME is anchored to the header date; for synoptic HOUR
            # soundings the launch often occurs the previous evening (e.g.
            # RELTIME=2312 on a Jan 2 00Z header means Dec 31 23:12). Fix: if
            # RELTIME places launch_time >12h after the synoptic anchor,
            # shift by -1 day.
            launch_time = None
            try:
                reltime_int = int(reltime_raw)
                if reltime_int != 9999:
                    rel_h = reltime_int // 100
                    rel_m = reltime_int % 100
                    if 0 <= rel_h <= 23 and 0 <= rel_m <= 59:
                        launch_time = np.datetime64(
                            datetime(year_val, month, day, rel_h, rel_m))
                        if synoptic_anchor is not None:
                            diff = launch_time - synoptic_anchor
                            if diff > np.timedelta64(12, "h"):
                                launch_time -= np.timedelta64(1, "D")
            except (ValueError, IndexError):
                pass

            # Fall back to the header hour if RELTIME is missing
            if launch_time is None:
                launch_time = header_time

            # Parse data levels
            altitudes = []
            pressures = []
            temps = []
            rhs = []
            wdirs = []
            wspds = []
            etimes = []

            for j in range(1, numlev + 1):
                if i + j >= len(lines):
                    break
                dline = lines[i + j]
                if len(dline) < 40:
                    continue

                # Fixed-width columns (0-indexed):
                # LVLTYP1:1  LVLTYP2:2  ETIME:3-8  PRESS:10-15  PFLAG:16
                # GPH:17-21  ZFLAG:22  TEMP:23-27  TFLAG:28
                # RH:29-33  DPDP:35-39  WDIR:41-45  WSPD:47-51
                etime_val = _parse_igra_value(dline[3:9])    # MMMSS format
                press_val = _parse_igra_value(dline[9:16])   # includes flag at pos 16
                gph_val = _parse_igra_value(dline[16:22])    # includes flag at pos 22
                temp_val = _parse_igra_value(dline[22:28])   # includes flag at pos 28
                rh_val = _parse_igra_value(dline[28:34]) if len(dline) > 33 else np.nan
                wdir_val = _parse_igra_value(dline[40:45]) if len(dline) > 44 else np.nan
                wspd_val = _parse_igra_value(dline[46:51]) if len(dline) > 50 else np.nan

                # ETIME is MMMSS: minutes * 100 + seconds
                if np.isfinite(etime_val):
                    mins = int(etime_val) // 100
                    secs = int(etime_val) % 100
                    etimes.append(mins * 60.0 + secs)
                else:
                    etimes.append(np.nan)

                pressures.append(press_val)          # already in Pa
                altitudes.append(gph_val)            # m (geopotential height)
                temps.append(temp_val / 10.0 if np.isfinite(temp_val) else np.nan)  # tenths → °C
                rhs.append(rh_val / 10.0 if np.isfinite(rh_val) else np.nan)       # tenths → %
                wdirs.append(wdir_val)                                               # degrees
                wspds.append(wspd_val / 10.0 if np.isfinite(wspd_val) else np.nan)  # tenths → m/s

            i += 1 + numlev

            if len(altitudes) < 3:
                continue

            altitude = _geopotential_to_geometric(altitudes)  # GPH → geometric
            p = np.array(pressures, dtype=np.float64)         # Pa
            T = np.array(temps, dtype=np.float64) + 273.15    # °C → K
            rh = np.array(rhs, dtype=np.float64)              # %
            wdir_arr = np.array(wdirs, dtype=np.float64)
            wspd_arr = np.array(wspds, dtype=np.float64)
            u, v = _wind_components(wspd_arr, wdir_arr)

            # Observation time: launch_time + elapsed seconds
            obs_time = None
            if launch_time is not None:
                elapsed_s = np.array(etimes, dtype=np.float64)
                obs_time = launch_time + (elapsed_s * 1e9).astype("timedelta64[ns]")

            profiles.append({
                "sonde_id": f"{station_id}_{year_val:04d}{month:02d}{day:02d}{hour:02d}",
                "launch_time": launch_time,
                "launch_lat": launch_lat,
                "launch_lon": launch_lon,
                "station_id": station_id,
                "altitude": altitude,
                "obs_time": obs_time,
                "p": p,
                "T": T,
                "RH": rh,
                "u": u,
                "v": v,
            })

    return profiles


# ---------------------------------------------------------------------------
#  SAM/LES simulated sondes  (sibling repo ../simulated-sondes)
# ---------------------------------------------------------------------------

# Source `launch_time` is seconds since simulation start.  The LES is based on
# the TWPICE campaign; we anchor the relative times to the TWPICE observation
# window so that the output `launch_time` coordinate is a real datetime64.
SAM_EPOCH = np.datetime64("2006-01-23T00:00:00", "ns")


def _read_sam_file(path, sonde_prefix):
    """Read one SAM/LES NetCDF into the standard profile-dict list.

    The file has dims (sonde=1000, altitude=1000) with U, V [m/s], QV [g/kg],
    P [Pa], plus per-sonde launch_time [s since sim start], launch_x/y [m].
    No temperature or relative humidity are reported, so the pipeline's q
    branch is used: QV is converted to kg/kg and passed as `q`, and the
    thermodynamic diagnostics that depend on T will remain all-NaN.
    """
    ds = xr.open_dataset(path, decode_times=False)
    altitude = ds["altitude"].values.astype(np.float64)
    U = ds["U"].values
    V = ds["V"].values
    QV = ds["QV"].values         # g/kg
    P = ds["P"].values           # Pa
    launch_seconds = ds["launch_time"].values.astype(np.float64)
    # LES-domain launch position (periodic x/y in metres).  Optional for
    # older SAM files; if absent, leave as NaN.
    launch_x = ds["launch_x"].values.astype(np.float64) if "launch_x" in ds \
        else np.full(U.shape[0], np.nan)
    launch_y = ds["launch_y"].values.astype(np.float64) if "launch_y" in ds \
        else np.full(U.shape[0], np.nan)
    ds.close()

    n = U.shape[0]
    launch_times = SAM_EPOCH + (launch_seconds * 1e9).astype("timedelta64[ns]")

    profiles = []
    for i in range(n):
        profiles.append({
            "sonde_id": f"{sonde_prefix}_{i:04d}",
            "launch_time": launch_times[i],
            "launch_lat": np.nan,
            "launch_lon": np.nan,
            "launch_x": float(launch_x[i]),
            "launch_y": float(launch_y[i]),
            "altitude": altitude,
            "u": U[i, :].astype(np.float64),
            "v": V[i, :].astype(np.float64),
            "p": P[i, :].astype(np.float64),
            "q": QV[i, :].astype(np.float64) * 1e-3,  # g/kg → kg/kg
        })
    return profiles


def read_sam_dropsondes(data_dir):
    """Read simulated LES dropsondes (1000 profiles)."""
    return _read_sam_file(os.path.join(data_dir, "simulated_dropsondes.nc"),
                          sonde_prefix="sam_drop")


def read_sam_radiosondes(data_dir):
    """Read simulated LES radiosondes (1000 profiles)."""
    return _read_sam_file(os.path.join(data_dir, "simulated_radiosondes.nc"),
                          sonde_prefix="sam_radio")


def read_sam_columns(data_dir):
    """Read instantaneous LES columns (1000 profiles, no sonde drift)."""
    return _read_sam_file(os.path.join(data_dir, "instantaneous_columns.nc"),
                          sonde_prefix="sam_col")


# ---------------------------------------------------------------------------
#  Multifractal synthetic profiles  (sibling repo ../simulated-sondes)
# ---------------------------------------------------------------------------

def _read_multifractal_file(path, sonde_prefix):
    """Read one multifractal NetCDF into the standard profile-dict list.

    The file has dims (sonde=1000, altitude=20000) with a single variable
    `u` (m/s) at 1 m spacing.  No other physical variables exist, and the
    profiles are synthetic, so launch_time / launch_lat / launch_lon are
    NaT / NaN.
    """
    ds = xr.open_dataset(path)
    altitude = ds["altitude"].values.astype(np.float64)
    u = ds["u"].values
    ds.close()

    profiles = []
    for i in range(u.shape[0]):
        profiles.append({
            "sonde_id": f"{sonde_prefix}_{i:04d}",
            "launch_time": None,
            "launch_lat": np.nan,
            "launch_lon": np.nan,
            "altitude": altitude,
            "u": u[i, :].astype(np.float64),
        })
    return profiles


def read_multifractal_uniform_H06(data_dir):
    """Read multifractal uniform H=0.6 synthetic profiles (1000)."""
    return _read_multifractal_file(
        os.path.join(data_dir, "simulated_multifractal_uniform_H06.nc"),
        sonde_prefix="mf_uniform_H06")


def read_multifractal_uniform_H06_nosmooth(data_dir):
    """Read multifractal uniform H=0.6 without B-spline smoothing (1000)."""
    return _read_multifractal_file(
        os.path.join(data_dir, "simulated_multifractal_uniform_H06_nosmooth.nc"),
        sonde_prefix="mf_uniform_H06_nosmooth")


def read_multifractal_broken_10m(data_dir):
    """Read multifractal broken regime (H=0.3 below 10 m, H=1 above) (1000)."""
    return _read_multifractal_file(
        os.path.join(data_dir, "simulated_multifractal_broken_10m.nc"),
        sonde_prefix="mf_broken_10m")


def read_multifractal_broken_1km(data_dir):
    """Read multifractal broken regime (H=0.3 below 1 km, H=1 above) (1000)."""
    return _read_multifractal_file(
        os.path.join(data_dir, "simulated_multifractal_broken_1km.nc"),
        sonde_prefix="mf_broken_1km")

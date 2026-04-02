"""
Validate the structure and metadata of all generated output NetCDF files.

Loops over every .nc file in output/ and checks that dimensions,
coordinates, data variables, and attributes match what process.py produces.
These tests operate on the generated datasets themselves — they are not
useful unless the datasets have been regenerated after code changes.
"""

import os
import glob

import numpy as np
import pytest
import xarray as xr

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

DATA_VARIABLES = ["u", "v", "p", "T", "RH", "q", "theta", "theta_e", "MSE", "DSE"]

EXPECTED_UNITS = {
    "u": "m s-1",
    "v": "m s-1",
    "p": "Pa",
    "T": "K",
    "RH": "%",
    "q": "kg kg-1",
    "theta": "K",
    "theta_e": "K",
    "MSE": "J kg-1",
    "DSE": "J kg-1",
}

REQUIRED_GLOBAL_ATTRS = [
    "title", "source", "history", "Conventions", "regridding_method",
    "grid_spacing_m", "grid_min_m", "grid_max_m",
    "n_launch_locations", "n_soundings", "n_altitude_levels",
]


def _find_output_files():
    """Return list of (name, path) for all .nc files in output/."""
    paths = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.nc")))
    return [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]


OUTPUT_FILES = _find_output_files()

if not OUTPUT_FILES:
    pytest.skip("No output files found in output/", allow_module_level=True)


@pytest.fixture(params=OUTPUT_FILES, ids=[name for name, _ in OUTPUT_FILES])
def dataset(request):
    name, path = request.param
    ds = xr.open_dataset(path)
    yield name, ds
    ds.close()


def _is_igra(name):
    return name.startswith("igra")


class TestDimensions:
    def test_has_three_dimensions(self, dataset):
        name, ds = dataset
        assert set(ds.dims) == {"launch_location", "sounding", "altitude"}

    def test_altitude_is_uniform(self, dataset):
        name, ds = dataset
        alt = ds["altitude"].values
        dz = np.diff(alt)
        assert np.allclose(dz, dz[0]), "altitude grid is not uniformly spaced"

    def test_altitude_positive_up(self, dataset):
        name, ds = dataset
        alt = ds["altitude"].values
        assert alt[0] < alt[-1], "altitude should increase"

    def test_dropsonde_sounding_dim_is_1(self, dataset):
        name, ds = dataset
        if _is_igra(name):
            pytest.skip("IGRA has multiple soundings per station")
        assert ds.sizes["sounding"] == 1


class TestCoordinates:
    def test_has_launch_lat(self, dataset):
        _, ds = dataset
        assert "launch_lat" in ds.coords
        assert ds["launch_lat"].dims == ("launch_location",)
        assert ds["launch_lat"].attrs.get("units") == "degrees_north"

    def test_has_launch_lon(self, dataset):
        _, ds = dataset
        assert "launch_lon" in ds.coords
        assert ds["launch_lon"].dims == ("launch_location",)
        assert ds["launch_lon"].attrs.get("units") == "degrees_east"

    def test_has_launch_time(self, dataset):
        _, ds = dataset
        assert "launch_time" in ds
        assert set(ds["launch_time"].dims) == {"launch_location", "sounding"}
        assert np.issubdtype(ds["launch_time"].dtype, np.datetime64)
        assert "long_name" in ds["launch_time"].attrs

    def test_has_observation_time(self, dataset):
        _, ds = dataset
        assert "observation_time" in ds
        assert set(ds["observation_time"].dims) == {"launch_location", "sounding", "altitude"}
        assert np.issubdtype(ds["observation_time"].dtype, np.datetime64)
        assert "long_name" in ds["observation_time"].attrs

    def test_has_location_id(self, dataset):
        name, ds = dataset
        if _is_igra(name):
            assert "station_id" in ds.coords
        else:
            assert "sonde_id" in ds.coords

    def test_altitude_attrs(self, dataset):
        _, ds = dataset
        attrs = ds["altitude"].attrs
        assert attrs.get("units") == "m"
        assert attrs.get("positive") == "up"

    def test_lat_lon_ranges(self, dataset):
        _, ds = dataset
        lats = ds["launch_lat"].values
        lons = ds["launch_lon"].values
        finite_lats = lats[np.isfinite(lats)]
        finite_lons = lons[np.isfinite(lons)]
        assert len(finite_lats) > 0, "no finite latitudes at all"
        assert len(finite_lons) > 0, "no finite longitudes at all"
        assert np.all((-90 <= finite_lats) & (finite_lats <= 90)), "latitude out of [-90, 90]"
        assert np.all((-360 <= finite_lons) & (finite_lons <= 360)), "longitude out of [-360, 360]"

    def test_profile_duration_under_6h(self, dataset):
        """All profiles should span less than 6 hours from launch to last observation."""
        _, ds = dataset
        max_duration = np.timedelta64(6, "h")
        launch = ds["launch_time"].values        # (location, sounding)
        obs = ds["observation_time"].values       # (location, sounding, altitude)
        n_loc, n_snd = launch.shape
        for i in range(n_loc):
            for j in range(n_snd):
                lt = launch[i, j]
                if np.isnat(lt):
                    continue
                obs_ij = obs[i, j, :]
                valid = obs_ij[~np.isnat(obs_ij)]
                if len(valid) == 0:
                    continue
                all_times = np.concatenate([[lt], valid])
                span = all_times.max() - all_times.min()
                assert span < max_duration, (
                    f"profile ({i}, {j}) spans {span} "
                    f"(launch {lt}, obs {all_times.min()} to {all_times.max()})"
                )


class TestDataVariables:
    def test_all_data_variables_present(self, dataset):
        _, ds = dataset
        for var in DATA_VARIABLES:
            assert var in ds, f"missing data variable {var}"

    def test_data_variable_shape(self, dataset):
        _, ds = dataset
        expected_shape = (
            ds.sizes["launch_location"],
            ds.sizes["sounding"],
            ds.sizes["altitude"],
        )
        for var in DATA_VARIABLES:
            assert ds[var].shape == expected_shape, f"{var} shape mismatch"

    def test_data_variable_units(self, dataset):
        _, ds = dataset
        for var, expected_unit in EXPECTED_UNITS.items():
            assert ds[var].attrs.get("units") == expected_unit, \
                f"{var} units: expected {expected_unit!r}, got {ds[var].attrs.get('units')!r}"

    def test_data_variables_have_long_name(self, dataset):
        _, ds = dataset
        for var in DATA_VARIABLES:
            assert "long_name" in ds[var].attrs, f"{var} missing long_name"



class TestGlobalAttributes:
    def test_required_global_attrs(self, dataset):
        _, ds = dataset
        for attr in REQUIRED_GLOBAL_ATTRS:
            assert attr in ds.attrs, f"missing global attribute {attr!r}"

    def test_conventions(self, dataset):
        _, ds = dataset
        assert ds.attrs["Conventions"] == "CF-1.8"

    def test_regridding_method(self, dataset):
        _, ds = dataset
        assert "bin averaging" in ds.attrs["regridding_method"]

    def test_provenance_attrs(self, dataset):
        _, ds = dataset
        assert "source_campaign" in ds.attrs, "missing provenance: source_campaign"
        assert "source_citation" in ds.attrs, "missing provenance: source_citation"

    def test_grid_params_consistent_with_altitude(self, dataset):
        _, ds = dataset
        alt = ds["altitude"].values
        dz = ds.attrs["grid_spacing_m"]
        assert np.isclose(np.diff(alt).mean(), dz, rtol=0.01)
        assert alt[0] >= ds.attrs["grid_min_m"]
        assert alt[-1] <= ds.attrs["grid_max_m"]

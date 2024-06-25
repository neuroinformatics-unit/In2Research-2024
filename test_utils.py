import pytest
import numpy as np
from utils import reshape_loaded_ds

import xarray as xr 

from movement.io import load_poses

array_before = np.arange(100*1*20*2).reshape(100, 1, 20, 2)
array_after = np.arange(100*2*10*2).reshape(100, 2, 10, 2)

ds_before = load_poses.from_numpy(
    position_array=array_before,
    confidence_array=np.ones((100, 1, 20)),
    individual_names=["ind1"],
    keypoint_names=[f"kpt{i}" for i in range(20)],
    fps=50
)

ds_expected_after = load_poses.from_numpy(
    position_array=array_after,
    confidence_array=np.ones((100, 2, 10)),
    individual_names=["ind1", "ind2"],
    keypoint_names=[f"kpt{i}" for i in range(10)],
    fps=50
)

def test_reshape_loaded_ds():
    ds_after = reshape_loaded_ds(
        ds_before, ["resident", "intruder"], [f"kpt{i}" for i in range(10)]
    )
    xr.testing.assert_allclose(ds_after, ds_expected_after)

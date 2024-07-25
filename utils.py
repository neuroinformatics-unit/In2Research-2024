import xarray as xr
from movement.io import load_poses


def reshape_loaded_ds(
    ds: xr.Dataset,
    true_ind_names: list[str],
    true_kpt_names: list[str],
) -> xr.Dataset:
    """
    Reshape the loaded dataset to have the correct dimensions.

    This assumes that multiple-animals were tracked using sinlge-animal
    DeepLabCut, and corrects the dataset given the known number of
    individuals and keypoints.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to reshape.
    true_ind_names : list[str]
        The names of the individuals.
    true_kpt_names : list[str]
        The names of the keypoints.

    Returns
    -------
    xr.Dataset
        The reshaped dataset.
    """

    position_array = ds.position.values
    confidence_array = ds.confidence.values

    n_true_inds = len(true_ind_names)
    n_true_kpts = len(true_kpt_names)
    n_frames = position_array.shape[0]
    n_space = position_array.shape[-1]
    new_shape = (n_frames, n_true_inds, n_true_kpts, n_space)


    ds_new = load_poses.from_numpy(
        position_array=position_array.reshape(new_shape),
        confidence_array=confidence_array.reshape(new_shape[:3]),
        individual_names=true_ind_names,
        keypoint_names=true_kpt_names,
        fps=ds.fps,
        source_software="DeepLabCut",
    )
    ds_new.attrs["source_file"] = ds.attrs["source_file"]
    return ds_new

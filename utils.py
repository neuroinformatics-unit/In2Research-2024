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


def clean_data(
    ds: xr.Dataset,
    confidence_threshold: float = 0.9,
    interp_max_gap: int = 25,
    smooth_window_size: int = 7,
    smooth_min_periods: int = 2,
):
    """
    Clean the position data in the dataset using the given parameters.

    The following steps are taken in order:
    1. Drop values with confidence below the threshold `confidence_threshold`.
    2. Linearly interpolate missing values over time, up to `interp_max_gap` frames.
    3. Smooth the data using a rolling median with window size `smooth_window`.
    
    Parameters
    ----------
    ds : xr.Dataset
        The dataset to clean. Must contain `position` and `confidence` variables.
    confidence_threshold : float, optional
        The confidence threshold below which to drop values, between 0 and 1,
        by default 0.8.
    interp_max_gap : int, optional
        The maximum gap over which to interpolate missing values, by default
        25 frames.
    smooth_window_size : int, optional
        The window size for the rolling median, by default 7 frames.
    smooth_min_periods : int, optional
        The minimum number of observations in the smoothing window required
        to compute the median (otherwise result is NaN). Default is 2.

    Returns
    -------
    xr.Dataset
        The cleaned dataset.


    Examples
    --------
    >>> ds_clean = clean_data(
    >>>     ds,
    >>>     confidence_threshold=0.9,
    >>>     interp_max_gap=25,
    >>>     smooth_window=7,
    >>>     smooth_min_periods=2,
    >>> )
    """

    # Copy the dataset to avoid modifying the original
    ds_clean = ds.copy()

    # Drop values with confidence below the threshold
    ds_clean.update(
        {
            "position": ds_clean.move.filter_by_confidence(
                threshold=confidence_threshold, print_report=False
            )
        }
    )

    # Interpolate missing values
    ds_clean.update(
        {
            "position": ds_clean.move.interpolate_over_time(
                method="linear", max_gap=interp_max_gap, print_report=False
            )
        }
    )

    # Smooth the data
    ds_clean.update(
        {
            "position": ds_clean.move.median_filter(
                window=smooth_window_size,
                min_periods=smooth_min_periods,
                print_report=False,
            )
        }
    )

    return ds_clean

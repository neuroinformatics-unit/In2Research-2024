from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import seaborn as sns

from matplotlib import pyplot as plt


def plot_trajectory(
    da: xr.DataArray,
    keypoint_name: str,
    individual_name: str,
    time_interval: Optional[tuple[float, float]] = None,
    time_unit: str = "seconds",
    cmap: str = "rainbow",
):
    """
    Plot the trajectory of a single keypoint over time.

    Parameters
    ----------
    da : xr.DataArray
        The data array containing the keypoint data.
    keypoint_name : str
        The keypoint to plot.
    individual_name : str
        The individual to plot.
    time_interval : a tuple of 2 floats, optional
        The start and end times of the time interval to plot, by default None,
        which plots the entire time range of the data array.
    time_unit : str, optional
        The unit of time to display on the x-axis, by default "seconds".
    cmap : str, optional
        The colormap to use, by default "rainbow".
    """

    fig, ax = plt.subplots(1, 1)

    if time_interval is not None:
        da = da.sel(time=slice(*time_interval))

    if "individuals" in da.dims:
        da = da.sel(individuals=individual_name)
    else:  # assume there is only one individual and expand the dimension
        da = da.expand_dims({"individuals": [individual_name]})

    if "keypoints" in da.dims:
        da = da.sel(keypoints=keypoint_name)
    else:  # assume there is only one keypoint and expand the dimension
        da = da.expand_dims({"keypoints": [keypoint_name]})
    
    sc = ax.scatter(
        da.sel(space="x"),
        da.sel(space="y"),
        s=10,
        c=da.time,
        cmap=cmap,
        marker="o",
    )

    ax.axis("equal")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()
    ax.set_title(f"{keypoint_name} trajectory for {individual_name}")
    fig.colorbar(sc, ax=ax, label=f"time ({time_unit})")
    plt.show()


def plot_histogram_with_percentiles(
    da: xr.DataArray,
    percentiles = [25, 50, 75],
    n_bins = 30,
    xlabel: Optional[str] = None,
    save_path: Optional[str | Path] = None,
):
    """
    Plot a histogram of the data with vertical lines at the given percentiles.

    Parameters
    ----------
    da : xr.DataArray
        The data to plot.
    percentiles : list[float], optional
        The percentiles to plot as vertical lines, by default [25, 50, 75].
    n_bins : int, optional
        The number of bins to use in the histogram, by default 30.
    xlabel : str, optional
        The label for the x-axis, by default None.
    save_path : str or Path, optional
        The path to save the figure to, by default None (do not save).
    """
    values = np.nanpercentile(da, percentiles)

    fig, ax = plt.subplots(1, 1)
    da.plot.hist(bins=n_bins, ax=ax)
    for perc, value in zip(percentiles, values):
        ax.axvline(value, color="red", linestyle="--")
        ax.text(
            value + 2, 0.85, f"{perc:d}%", rotation=90, va="bottom",
            color="red", transform=ax.get_xaxis_transform()
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")

    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_head_orientation_polar_histogram(
    angles: xr.DataArray,
    resident_id: str,
    intruder_id: str,
    ax=None,
):
    """
    Plot a polar histogram of the resident's head orientation relative to the 
    intruder's head vector.

    The angles should be given in radians and represent the arccos of the
    dot product between the normalised intruder's head vector (in practice
    a vector perependicular to the mid-ear line, pointing forwards)
    and the normalised vector pointing from the intruder's head towards the
    resident's head (in practice, a vector starting at intruder's ear midpoint
    and ending at the resident's 'neck' keypoint). The range of the angles
    should be [0, pi] (i.e. [0, 180] degrees), with 0 meaning that the resident
    is exactly ahead of the intruder, pi meaning that the resident is exactly
    behind the intruder, and pi/2 meaning that the resident is perpendicular
    to the intruder's head vector.

    Parameters
    ----------
    angles : xr.DataArray
        The angles in radians.
    resident_id : str
        The resident's name.
    intruder_id : str
        The intruder's name.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on, by default None.
    """
    if ax is None:
        fig, ax  = plt.subplots(
            1, 1,
            figsize=(6, 6.5),
            subplot_kw={'projection': 'polar'}
        )

    # compute number of bins
    bin_width_deg = 15  # width of the bins in degrees
    n_bins = int(180 / bin_width_deg)

    # initialise figure with polar projection

    intruder_sex = intruder_id[:-1  ]
    color = "blue" if intruder_sex == "male" else "orange"

    # plot histogram using xarray's built-in histogram function
    angles.plot.hist(
        bins=np.linspace(0, np.pi, n_bins + 1), color=color, ax=ax
    )

    # axes settings
    ax.set_title(f"Resident: {resident_id} | Intruder: {intruder_id}")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_direction(-1)  # theta increases in clockwise direction
    ax.set_theta_offset(np.pi/2)  # set zero at the right
    #ax.set_xlabel("")  # remove default x-label from xarray's plot.hist()

    # set xticks to match the phi values in degrees
    n_xtick_edges = 5
    ax.set_xticks(np.linspace(0, np.pi, n_xtick_edges))
    xticklabels = [str(t) + "\N{DEGREE SIGN}" for t in np.linspace(0, 180, n_xtick_edges)]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels([])  # remove y-tick labels

# %%
from pathlib import Path
import numpy as np
from movement.io import load_poses

from utils import reshape_loaded_ds

# %%
user_home = Path.home()
data_folder = user_home / "Data" / "behav"
assert data_folder.is_dir(), f"{data_folder} is not a directory."

file_name = "220804_SB021_FM001_female2_2022-08-04-223620DLC_resnet50_shanice_allNov29shuffle1_196000_filtered.csv"
file_path = data_folder / file_name
assert file_path.is_file(), f"{file_path} is not a file."

# %%
ds = load_poses.from_dlc_file(file_path, fps=50)
ds

# %%
# Create a new ds

keypoint_names = ds.keypoints.values
n_kpts = len(keypoint_names)
new_keypoint_names = []
for name in keypoint_names[:n_kpts // 2]:
    new_keypoint_names.append(name.split("_")[1])
print(new_keypoint_names)
assert len(new_keypoint_names) == 10
new_indiv_names = ["resident", "intruder"]

# %%
ds_new = reshape_loaded_ds(
    ds,
    n_true_inds=2,
    n_true_kpts=10,
    true_ind_names=new_indiv_names,
    true_kpt_names=new_keypoint_names
)

ds_new

# %%

# test the results
 
np.testing.assert_allclose(
    ds.position.sel(individuals="individual_0", keypoints="resident_nose").values,
    ds_new.position.sel(individuals="resident", keypoints="nose").values
)






# %%

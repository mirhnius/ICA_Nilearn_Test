import pathlib
import numpy as np
from nilearn import masking
from nilearn.image import iter_img
from nilearn.masking import apply_mask
from nilearn.image import binarize_img
from nilearn.image import new_img_like
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas
from nilearn.plotting import plot_stat_map, show

NonePath = type('NonePath', (), {'resolve': lambda: None})

ICS_PATH = pathlib.Path('/data/origami/niusha/code/local-experiment/io/ICAs')
io_path = ICS_PATH.parent
n_components = 30

def ICA_decomposition(filenames, group, i, path=ICS_PATH, n=n_components):
    """
    This function receives nifiti images and calculates 30 independent components

    inputs: 
        filenames: list of input filenames or a 4D image containing all inputs concatenated.
        group: a group of subjects (PD/Healthy/Pooled). It is used for naming IC files.
        path: parent directory for storing ICs.
        i: iteration number is used for naming IC files.
        n: number of ICs.

    outputs: 
        ICA_s: A 4D image that contains 30 ICs.
    """
    fast_ica = CanICA(n_components=n,
                    memory="nilearn_cache", memory_level=2,
                    mask_strategy='whole-brain-template',
                    do_cca=False,
                    random_state=0,
                    standardize=True, 
                    standardize_confounds=True,
                    )
    fast_ica.fit(filenames)

    ICA_s = fast_ica.components_img_
    ICA_s.to_filename(path / (f'ICAs_{group}_{i}.nii.gz'))
    return ICA_s



def Means_after_masking(ICAs,DBM_maps,path=NonePath,group=""):
    """
    This function first extracts a mask from each IC and then applies it to each subject.
    After extracting regions of interest, the function calculates the mean value of these regions
    for each subject.
      
    inputs:
        ICAs: IC components
        DBM_maps: input images; a 4 D image
        path: Directory to save masks ans masked data.
 
    outputs:
        means_after_mask: an array contains the mean value of each input DBM after applying IC masks.
    """
    mask_dir = path / "masks"
    masked_data_dir = path / "masked_data"

    if path.resolve()is not None:
      mask_dir.mkdir(parents=True, exist_ok=True)  
      masked_data_dir.mkdir(parents=True, exist_ok=True)

    size = DBM_maps.shape[3]
    n = ICAs.shape[3]
    means_after_mask = np.zeros((n,size), dtype='<f4')

    for i, cur_ic in enumerate(iter_img(ICAs)):

        mask = binarize_img(masking.compute_brain_mask(
            target_img=cur_ic,
            mask_type='whole-brain',
            ))
        masked_data = apply_mask(imgs=DBM_maps,mask_img=mask) # is it ok to apply on whole subjects?
        means_after_mask[i,:] = np.nanmean(masked_data,axis=1)

        if path.resolve() is not None:
            # new_img_like(DBM_maps, masked_data).to_filename(masked_data_dir / f"masked_{group}_IC_{i}.nii.gz")
            np.savetxt(masked_data_dir / f"masked_{group}_IC_{i}.txt",masked_data)
            mask.to_filename(mask_dir / f"IC_{group}_{i}.nii.gz")

    return means_after_mask


def plot_ICA_components(ICs):

    plot_prob_atlas(ICs, title="All ICA components")

    for i, cur_img in enumerate(iter_img(ICs)):
        plot_stat_map(cur_img, display_mode="z", title="IC %d" %i,
        cut_coords=1, colorbar=False)

    # add show

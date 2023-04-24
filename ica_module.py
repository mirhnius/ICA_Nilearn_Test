import pathlib
import numpy as np
from nilearn import masking
from nilearn.image import iter_img
from nilearn.masking import apply_mask
from nilearn.image import binarize_img
from nilearn.decomposition import CanICA


ICS_PATH = pathlib.PurePath('/data/origami/niusha/code/local-experiment/io/ICAs')
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



def Means_after_masking(ICAs,DBM_maps,n=n_components):
    """
    This function first extracts a mask from each IC and then applies it to each subject.
    After extracting regions of interest, the function calculates the mean value of these regions
    for each subject.
      
    inputs:
        ICAs: IC components
        DBM_maps: input images; a 4 D image
    outputs:
        means_after_mask: an array contains the mean value of each input DBM after applying IC masks.
    """
    size = DBM_maps.shape[3]
    means_after_mask = np.zeros((n,size))
    for i, cur_ic in enumerate(iter_img(ICAs)):

        mask = binarize_img(masking.compute_brain_mask(
            target_img=cur_ic,
            mask_type='whole-brain',
            ))
        masked_data = apply_mask(imgs=DBM_maps,mask_img=mask) # is it ok to apply on whole subjects?
        means_after_mask[i,:] = np.mean(np.around(np.nan_to_num(masked_data),decimals=5), axis=1)
        
    return means_after_mask
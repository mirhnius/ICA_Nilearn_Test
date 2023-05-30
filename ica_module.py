import copy
import pathlib
import numpy as np
from nilearn.image import binarize_img, new_img_like, iter_img, threshold_img
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_prob_atlas, plot_stat_map, show

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

def mean_masking(data, mask, threshold="99.6%", two_sided=False):
    """  
    This function first extracts a mask from each IC and then applies it to each subject.
    After extracting regions of interest, the function calculates the mean value of these regions
    for each subject.
    
    inputs:
        data: All subjects' data as a numpy array
        mask: Mask image
        threshold:
        two_sided:
        
    outputs:
        masked_data: Data after applying the mask as a numpy array.
        filtered_mask: Mask image after replacing voxels with values smaller than zero with zero. 
    
    """
    mask_thr = threshold_img(img=mask, threshold=threshold, two_sided=two_sided)
    mask_d = mask_thr.get_fdata()
    masked_data = data * np.reshape(mask_d, newshape=list(mask_d.shape)+[1])
    
    return masked_data, mask_thr

def mean_calc(masked_data):
    
    n_subjects = masked_data.shape[3]
    means_after_mask = np.zeros((1,n_subjects))
    for j in range(n_subjects):
            subject = masked_data[...,j]
            index = np.nonzero(subject)
            means_after_mask[0,j] = np.nanmean(subject[index])
    
    return means_after_mask

def weighted_sum_masking(data, mask):
    """
    This function first extracts a mask from each IC and then applies it to each subject.
    After extracting regions of interest, the function calculates the weighted sum of these regions
    for each subject.
    
    inputs:
        data: All subjects' data as a numpy array
        mask: Mask image
        
    outputs:
        masked_data: Data after applying the mask as a numpy array.
        filtered_mask: Mask image after replacing voxels with values smaller than zero with zero. 
    
    """
    mask_d = copy.deepcopy(mask.get_fdata())
    mask_d[mask_d < 0] = 0
    masked_data = data * np.reshape(mask_d, newshape=list(mask_d.shape)+[1])
    filtered_mask = new_img_like(ref_niimg=mask, data=mask_d)
    
    return masked_data, filtered_mask

def weighted_sum_calc(masked_data, mask_d):
    """
    means_after_mask: an array contains the mean value of each input DBM after applying IC masks.
    """
    weights_after_mask = np.sum(masked_data, axis=(0,1,2)) / np.sum(mask_d)  
    return weights_after_mask

def apply_masker(ICAs, DBM_maps, path=NonePath, group="", func=mean_masking):
    """
    This function applies a masker to the input data and save the maked data if a path is provided for it
      
    inputs:
        ICAs: IC components
        DBM_maps:  a 4-D input image. Subjects are concatinated to a single 4-D file.
        path: Directory to save masks ans masked data.
        group: Cohort's name for storing IC components
        func: chossing the way to calculate the masking procedure, options: 
            "mean_masking", "weighted_sum_masking"
 
    outputs:
        extracted_from_mask: A numpy array that containes extracted data per each subject for each mask image.
    """
    mask_dir = path / "masks"
    masked_data_dir = path / "masked_data"

    if path.resolve()is not None:
      mask_dir.mkdir(parents=True, exist_ok=True)  
      masked_data_dir.mkdir(parents=True, exist_ok=True)
      
    n = ICAs.shape[3]
    n_subjects = data.shape[3]
    extracted_from_mask = np.zeros((n,n_subjects), dtype='<f8') #, dtype='<f4'
    data = DBM_maps.get_fdata()
    
    for i, cur_ic in enumerate(iter_img(ICAs)):
        masked_data, mask = func(data, cur_ic)
        
        if str(func) == "mean_masking":
            extracted_from_mask[i,:] = mean_calc(masked_data, mask.get_fdata())
            
        elif  str(func) == "weighted_sum_masking": 
            extracted_from_mask[i,:] = weighted_sum_calc(masked_data, mask.get_fdata())
        
        if path.resolve() is not None:
            # new_img_like(DBM_maps, masked_data).to_filename(masked_data_dir / f"masked_{group}_IC_{i}.nii.gz")
            np.savetxt(masked_data_dir / f"masked_{group}_IC_{str(func)}_{i}.txt", extracted_from_mask)
            mask.to_filename(mask_dir / f"IC_{group}_{i}.nii.gz")

    return extracted_from_mask

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.image import index_img, iter_img
from nilearn.plotting import plot_stat_map, plot_prob_atlas

WIDTH_SIZE = 6
HEIGHT_SIZE = 4

#I assumed the img is 4D for now
def image_report(img):

    n_bins = 3
    data = img.get_fdata()
    
    max_each_img = np.nanmax(data, axis=(0,1,2))
    max_ = np.max(max_each_img)
    
    mean_each_img = np.nanmean(data, axis=(0,1,2))
    mean_ = np.mean(mean_each_img)
    
    min_each_img = np.nanmin(data, axis=(0,1,2))
    min_ = np.min(min_each_img)
    
    var_each_img = np.nanvar(data, axis=(0,1,2))
    var_ = np.nanvar(data)
    
    nans = np.count_nonzero(np.isnan(data))
   
    names = ["Maximum", "Minimum", "Mean", "Variance"]
    point_estimators = [max_, min_, mean_, var_]
    distributions = [max_each_img, min_each_img, mean_each_img, var_each_img]
    
    print(f"     ======== General Information ========")
    print(f"Number of nan voxels: {nans}")
    
    for i in range(4):    
        print(f"Overall {names[i]}: {point_estimators[i]:.4f}")
        
    f, axes = plt.subplots(nrows=1, ncols=4, figsize=(WIDTH_SIZE*3,HEIGHT_SIZE))    
    for i in range(4):
        sns.histplot(distributions[i], 
            color=sns.color_palette('Set2')[1],
            bins=n_bins, ax=axes[i]
            )
        axes[i].title.set_text(f"{names[i]} Value For Each Subject")
    plt.show()


def outlier_indices(imgs):
    """
    This function detects image slices with oddly big values.

    Input:
        img: a 4D image as input
        
    Output:
        outliers: indices of problematic slices
    """    
    thr = pow(10, 20)
    outliers = []
    data = imgs.get_fdata()
    for i in range(data.shape[3]):
        if np.abs(np.nanmax(data[...,i])) > thr or np.abs(np.nanmin(data[...,i])) > thr:
            outliers.append(i)

    return outliers


def two_array_diagrams(g1,g2,labels,title):

    for i in range(g1.shape[0]):
        plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))
        plt.title(title+f'{i}')
        ax = sns.histplot(data=g1[i,:], 
            color=sns.color_palette('Set2')[1]
            )
        ax = sns.histplot(data=g2[i,:], 
            color=sns.color_palette('Set2')[2]
            )
        ax.legend(title="Group", loc='upper right', labels=labels)
        plt.show()


def random_plotter(imgs, n_rows=4, n_cols=5, z_slice=1):

    n_imgs = imgs.shape[3]
    plot_size = n_rows * n_cols
    samples_indices = np.random.choice(n_imgs,plot_size, replace=False)
    selected_imgs = index_img(imgs,samples_indices)

    f, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5+n_cols,4+n_rows))
    for (i, ax), cur_img in zip(enumerate(axes.flatten()),iter_img(selected_imgs)):

        display = plot_stat_map(cur_img, display_mode="z", cut_coords=z_slice, figure=f, 
        axes=ax, colorbar=False, annotate=True) 
        display.title(f'Sub {samples_indices[i]}', size=8)
    # f.save("test.png")


def plot_ICA_components(imgs, n_cols=5, z_slice=1): #maybe I should combine it with the other plotter

    n_imgs = imgs.shape[3]
    n_rows = int(np.ceil(n_imgs / n_cols))

    plot_prob_atlas(imgs, title="All ICA components")

    f, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5+n_cols,4+n_rows))
    for (i, ax), cur_img in zip(enumerate(axes.flatten()[:n_imgs]), iter_img(imgs)):
        display = plot_stat_map(cur_img, display_mode="z", figure=f, 
        axes=ax,cut_coords=z_slice, colorbar=True)
        display.title(f'Component {i}', size=8)
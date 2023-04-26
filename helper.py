import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

WIDTH_SIZE = 6
HEIGHT_SIZE = 4

#I assumed the img is 4D for now
def image_report(img):

    n_bins = 3
    data = img.get_fdata()
    type_ = type(data)
    max_each_img = np.max(np.max(np.nanmax(data, axis=0),axis=0), axis=0)
    max_ = np.nanmax(data)
    mean_each_img = np.mean(np.mean(np.nanmean(data, axis=0),axis=0), axis=0)
    mean_ = np.nanmean(data)
    min_each_img = np.min(np.min(np.nanmin(data, axis=0),axis=0), axis=0)
    min_ = np.nanmin(data)

    print(f"     ======== data type ======== \n {type_}") #change it. I need the image datatype also I need memmap data type
 
    names = ["Maximum", "Mean", "Minimum"]
    point_estimators = [max_, mean_, min_]
    distributions = [max_each_img, mean_each_img, min_each_img]
    for i in range(3):
        print(f"      ======== {names[i]} ========")
        print(f"Overall {names[i]}: {point_estimators[i]:.2f}")
        plt.hist(distributions[i],n_bins)
        plt.title(f"{names[i]} Value For Each Subject")
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
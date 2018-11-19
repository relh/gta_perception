import torch
import os
import numpy as np
from skimage import io
from pdb import set_trace
from torch.utils.data import Dataset, DataLoader

from carnet import CarDataset 

def get_all_images(base):
    """This function collects image names and their associated labels from a directory."""
    item = []
    for f in os.listdir(base):
        if os.path.isdir(os.path.join(base,f)):  
            for ff in os.listdir(os.path.join(base, f)):
                if ".jpg" in ff:
                    root = ff.split('_')[0]
                    bbox_cols = np.fromfile(os.path.join(base,f,root+'_bbox.bin'), dtype=np.float32)
                    #proj_cols = np.fromfile(os.path.join(base,f,root+'_proj.bin'), dtype=np.float32)
                    #cloud_cols = np.fromfile(os.path.join(base,f,root+'_cloud.bin'), dtype=np.float32)
                    item.append((os.path.join(base,f,ff), bbox_cols[9]))
    return item 

    
def compute_mean_std(loader):
    """This function computes the mean and standard deviation of the 3 image channels for all the images"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.

    for i, data_tup in enumerate(loader):
        print("{} / {}".format(i, len(loader)))
        
        data = data_tup[0]
        batch_samples = data.size(0)
        data = data.float()
      	#data = data.view(batch_samples, data.size(1), -1)
        
        mean += torch.mean(torch.mean(data, 1), 1)
        std += torch.mean(torch.std(data, 1), 1)
        nb_samples += batch_samples
        if i == 10:
          break
    
    mean /= nb_samples
    std /= nb_samples
    return mean, std
    

if __name__ == "__main__":
    base = "/hdd/trainval/" # Change this to point to your datapath

    print('--- dataset creator ---')
    trainval_carData = CarDataset(base)
    
    print('--- mean + std calc ---')
    train_loader = _get_dataloader(10, trainval_carData)
    mean, std = compute_mean_std(train_loader)
    print(mean)
    print(std)

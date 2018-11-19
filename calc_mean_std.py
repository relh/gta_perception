#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import torch
#import PIL

import torch
import os
import numpy as np
from skimage import io
from pdb import set_trace
from torch.utils.data import Dataset, DataLoader

def get_all_images(base):
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
    
def main():
    print('--- dataset creator ---')
    base = "/hdd/trainval/" # Change this to point to your path
    carData = CarDataset(base)
    
    print('--- mean + std calc ---')
    data_loader = car_dataloader(carData)
    mean, std = compute_mean_std(data_loader)
    print(mean)
    print(std)

if __name__ == "__main__":
  main()

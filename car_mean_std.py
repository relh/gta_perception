# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#import torch
#import PIL

import torch
import os
from skimage import io
from torch.utils.data import Dataset, DataLoader
    
def compute_mean_std(dataset):
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=0,
        shuffle=False
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.

    for i, data in enumerate(loader):
        print("{} / {}".format(i, len(loader)))
        batch_samples = data.size(0)
        data = data.float()
      	#data = data.view(batch_samples, data.size(1), -1)
        
        mean += torch.mean(torch.mean(torch.mean(data, 0), 0), 0)
        std += torch.mean(torch.mean(torch.std(data, 0), 0), 0)
        nb_samples += batch_samples
        #if i == 10:
        #  break
    
    mean /= nb_samples
    std /= nb_samples
    return mean, std
    
def get_all_images(base):
    images = []
    for f in os.listdir(base):
        if os.path.isdir(os.path.join(base,f)):  
            for ff in os.listdir(os.path.join(base, f)):
                if ".jpg" in ff:
                    #print(ff)
                    images.append(os.path.join(base,f,ff))
    return images

class CarDataset(Dataset):
    def __init__(self, base):
        self.img_names = get_all_images(base)
        #self.data = get_all_images(base)
        
    def __getitem__(self, index):
        return io.imread(self.img_names[index])

    def __len__(self):
        return len(self.img_names)
    
def main():
    # Any results you write to the current directory are saved as output.
    #classes = open('../input/classes.csv').readlines()

    print('--- dataset creator ---')
    base = "/hdd/trainval/"
    carData = CarDataset(base)
    
    print('--- mean + std calc ---')
    mean, std = compute_mean_std(carData)
    print(mean)
    print(std)

#if __name__ "__main__":
main()

import torch
import os
import numpy as np
from skimage import io
from PIL import Image
from pdb import set_trace
from torch.utils.data import Dataset, DataLoader

class CarDataset(Dataset):
    def __init__(self, item_names, transform=None):
        """This Dataset takes in a folder root, the frequency with which to include samples in validation, and a flag for validation."""
        self.item_names = item_names
        self.transform = transform
        
    def __getitem__(self, index):
        im_path, im_class = self.item_names[index]
        loaded_im = io.imread(im_path)
        #loaded_im = Image.open(im_path)
        #trans_im = self.transform(loaded_im)
        #loaded_im = loaded_im.permute(2,0,1) 
        return im_path, torch.tensor(loaded_im).float(), torch.from_numpy(np.array(im_class)).long()

    def __len__(self):
        return len(self.item_names)


def _get_dataloader(batch_size, dataset):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

def get_all_image_label_pairs(base):
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
                    item.append((os.path.join(base,f,ff), bbox_cols[0:9]))
    return item 

    
def compute_mean_std(loader):
    """This function computes the mean and standard deviation of the 3 image channels for all the images"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.

    for i, data_tup in enumerate(loader):
        print("{} / {}".format(i, len(loader)))
        
        data = data_tup[2][:, 3:6]
        batch_samples = data.size(0)
        data = data.float()
      	#data = data.view(batch_samples, data.size(1), -1)
        
        mean += torch.mean(data, 0)
        #std += torch.mean(torch.std(data, 1), 1)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    return mean, std
    

if __name__ == "__main__":
    base = "/hdd/trainval/" # Change this to point to your datapath

    print('--- dataset creator ---')
    item_names = get_all_image_label_pairs(base)
    carData = CarDataset(item_names)
    
    print('--- mean + std calc ---')
    train_loader = _get_dataloader(10, carData)
    mean, std = compute_mean_std(train_loader)
    print(mean)
    print(std)

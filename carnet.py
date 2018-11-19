from pathlib import Path
import os
import numpy as np
from skimage import io

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from se_resnet import se_resnet_custom
from utils import Trainer


def get_all_images(base, mod, val_flag):
    item = []
    for f in os.listdir(base):
        if os.path.isdir(os.path.join(base,f)):  
            val_counter = 0
            for iii, ff in enumerate(os.listdir(os.path.join(base, f))):
                if ".jpg" in ff:
                    root = ff.split('_')[0]
                    bbox_cols = np.fromfile(os.path.join(base,f,root+'_bbox.bin'), dtype=np.float32)
                    #proj_cols = np.fromfile(os.path.join(base,f,root+'_proj.bin'), dtype=np.float32)
                    #cloud_cols = np.fromfile(os.path.join(base,f,root+'_cloud.bin'), dtype=np.float32)

                    if val_flag and val_counter % mod == 0:
                      item.append((os.path.join(base,f,ff), bbox_cols[9]))
                    else:
                      item.append((os.path.join(base,f,ff), bbox_cols[9]))
                    val_counter += 1
            #if iii > 4:
            #  break
    return item 

    
class CarDataset(Dataset):
    def __init__(self, base, mod, val_flag):
        self.item_names = get_all_images(base, mod, val_flag)
        
    def __getitem__(self, index):
        im_path, im_class = self.item_names[index]
        loaded_im = io.imread(im_path).transpose((2,0,1))
        #im_class = np.ones(1) * im_class
        #im_class = torch.tensor(im_class)#.unsqueeze(1)
        return torch.tensor(loaded_im).float(), torch.from_numpy(np.array(im_class)).long()

    def __len__(self):
        return len(self.item_names)


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.CenterCrop(1024),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[92.458, 91.290, 88.659], std=[35.646, 33.245, 31.304])]

    data_augmentation = [transforms.RandomSizedCrop(1024),
                         transforms.RandomHorizontalFlip(), ]

    base = "/hdd/trainval/" # Change this to point to your path
    val_step = 4
    train_carData = CarDataset(base, 4, val_step)
    val_carData = CarDataset(base, 4, val_step)

    train_loader = DataLoader(
        train_carData,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )
    val_loader = DataLoader(
        val_carData,
        batch_size=batch_size,
        num_workers=1,
        shuffle=True
    )

    return train_loader, val_loader


def main(batch_size, root):
    train_loader, test_loader = get_dataloader(batch_size, root)
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))
    se_resnet = nn.DataParallel(se_resnet_custom(num_classes=23), device_ids=gpus)
    #se_resnet = se_resnet20(num_classes=23)#, device_ids=torch.device("cpu"))
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6 / 1024 * batch_size, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default='/hdd/', type=str, help="carnet data root")
    p.add_argument("--batch_size", default=2, type=int)
    args = p.parse_args()
    main(args.batch_size, args.root)

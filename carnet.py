import os
import numpy as np
from skimage import io

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split 

from se_resnet import se_resnet_custom
from utils import Trainer


def get_all_image_label_pairs(root):
    item = []
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root,f)):  
            val_counter = 0
            for ff in os.listdir(os.path.join(root, f)):
                if ".jpg" in ff:
                    base = ff.split('_')[0]
                    bbox_cols = np.fromfile(os.path.join(root,f,base+'_bbox.bin'), dtype=np.float32)

                    # Append items to dataset
                    item.append((os.path.join(root,f,ff), bbox_cols[9]))

                    # Commented out code for having the bounding box be the label
                    # Each row contains information of a bounding box: rotation vector, position (centroid x, y, z), size of the bounding box (length, width, height)
                    #item.append((os.path.join(root,f,ff), bbox_cols[0:9]))
    return item 

    
class CarDataset(Dataset):
    def __init__(self, root):
        """This Dataset takes in a folder root, the frequency with which to include samples in validation, and a flag for validation."""
        self.item_names = get_all_image_label_pairs(root)
        
    def __getitem__(self, index):
        im_path, im_class = self.item_names[index]
        loaded_im = io.imread(im_path).transpose((2,0,1))
        return torch.tensor(loaded_im).float(), torch.from_numpy(np.array(im_class)).long()

    def __len__(self):
        return len(self.item_names)


def _get_dataloader(batch_size, dataset):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )


def get_dataloader(batch_size, root):
    to_normalized_tensor = [transforms.CenterCrop(224), # 1024
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[92.458, 91.290, 88.659], std=[35.646, 33.245, 31.304])]

    data_augmentation = [transforms.RandomSizedCrop(224), # 1024
                         transforms.RandomHorizontalFlip(), ]

    val_step = 4

    # Create the datasets
    carData = CarDataset(root)
    train_len = int(len(carData) * 0.75)
    val_len = len(carData) - train_len 
    print("--- Train Length: {} ---".format(train_len))
    print("--- Val Length: {} ---".format(val_len))
    train_carData, val_carData = random_split(carData, (train_len, val_len))

    # Create the dataloaders
    train_loader = _get_dataloader(batch_size, train_carData)
    val_loader = _get_dataloader(batch_size, val_carData)
    return train_loader, val_loader


def main(batch_size, root, lr):
    # Get the train and validation data loaders
    train_loader, test_loader = get_dataloader(batch_size, root)

    # Specify the GPUs to use
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))

    # Build the model to run
    se_resnet = nn.DataParallel(se_resnet_custom(num_classes=23), device_ids=gpus)
    #se_resnet = se_resnet20(num_classes=23)#, device_ids=torch.device("cpu"))

    # Declare the optimizer, learning rate scheduler, and training loops. Note that models are saved to the current directory.
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default='/hdd/trainval/', type=str, help="carnet data root")
    p.add_argument("--batch_size", default=1, type=int, help="batch size")
    p.add_argument("--lr", default=1e-1, type=float, help="learning rate")
    args = p.parse_args()
    main(args.batch_size, args.root, args.lr)

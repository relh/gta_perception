import os
import numpy as np
import pickle
import csv
import numpy as np
from skimage import io
from pdb import set_trace
from PIL import Image

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split 

from se_resnet import se_resnet_custom
from utils import Trainer

def get_all_image_label_pairs(root):
    with open('classes.csv', 'r') as class_key:
      reader = csv.reader(class_key)
      list_mapping = list(reader)

    item = []
    for f in os.listdir(root):
        if os.path.isdir(os.path.join(root,f)):  
            for ff in os.listdir(os.path.join(root, f)):
                if ".jpg" in ff:
                    base = ff.split('_')[0]
                    if os.path.exists(os.path.join(root,f,base+'_bbox.bin')):
                      bbox_cols = np.fromfile(os.path.join(root,f,base+'_bbox.bin'), dtype=np.float32)
                    else:
                      bbox_cols = [0]*10

                    # Append items to dataset
                    mod_val = int(list_mapping[int(bbox_cols[9])+1][-1])
                    item.append((os.path.join(root,f,ff), mod_val))

                    # Commented out code for having the bounding box be the label
                    # Each row contains information of a bounding box: rotation vector, position (centroid x, y, z), size of the bounding box (length, width, height)
                    #item.append((os.path.join(root,f,ff), bbox_cols[0:9]))
    return item 

    
class CarDataset(Dataset):
    def __init__(self, item_names, transform):
        """This Dataset takes in a folder root, the frequency with which to include samples in validation, and a flag for validation."""
        self.item_names = item_names
        self.transform = transform
        
    def __getitem__(self, index):
        im_path, im_class = self.item_names[index]
        #loaded_im = io.imread(im_path)
        loaded_im = Image.open(im_path)
        trans_im = self.transform(loaded_im)
        trans_im.permute(2,0,1) 
        return im_path, torch.tensor(trans_im).float(), torch.from_numpy(np.array(im_class)).long()

    def __len__(self):
        return len(self.item_names)


def _get_dataloader(batch_size, dataset):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )


def get_dataloader(batch_size, root, split_size=0.75):
    #to_normalized_tensor = transforms.Compose([transforms.ToTensor()])#,
    #transforms.Normalize(mean=[92.458, 91.290, 88.659], std=[35.646, 33.245, 31.304])])
    #transforms.CenterCrop(1024)
    data_augmentation = transforms.Compose([transforms.RandomResizedCrop(1024), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[.362, .358, .347], std=[.139, .130, .123])])
    val_step = 4

    # Create the datasets
    item_names = get_all_image_label_pairs(root)
    carData = CarDataset(item_names, data_augmentation)
    train_len = int(len(carData) * split_size)
    val_len = len(carData) - train_len 
    print("--- Train Length: {} ---".format(train_len))
    print("--- Val Length: {} ---".format(val_len))
    train_carData, val_carData = random_split(carData, (train_len, val_len))

    # Create the dataloaders
    train_loader = _get_dataloader(batch_size, train_carData)
    val_loader = _get_dataloader(batch_size, val_carData)
    return train_loader, val_loader


def main(batch_size, root, lr, load, load_epoch, train, testing):
    # Get the train and validation data loaders
    train_loader, test_loader = get_dataloader(batch_size, root)

    # Specify the GPUs to use
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))

    # Build the model to run
    #se_resnet = nn.DataParallel(se_resnet_custom(num_classes=3), device_ids=gpus)
    se_resnet = se_resnet_custom(num_classes=3)#, device_ids=gpus)
    #se_resnet = se_resnet20(num_classes=23)#, device_ids=torch.device("cpu"))

    if load_epoch > 0:
      details = torch.load("models/v6/model_epoch_{}.pth".format(str(load_epoch)))
      #new_details = dict([(k[7:], v) for k, v in details['weight'].items()])
      se_resnet.load_state_dict(details['weight'])

    # Declare the optimizer, learning rate scheduler, and training loops. Note that models are saved to the current directory.
    optimizer = optim.Adam(params=se_resnet.parameters(), lr=lr)#, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir=".")

    if train:
      trainer.loop(100, train_loader, test_loader, scheduler)

    if testing:
      se_resnet.eval()
      t_l_1, t_l_2 = get_dataloader(batch_size, '/hdd/test/', 1.0)
      outputs = trainer.test(t_l_1)
      with open('submission.csv', 'w') as sub:
        sub.write('guid/image,label\n')
        for name, val in outputs:
          mod_name = name[0].split('/')[3] + '/' + name[0].split('/')[4].split('_')[0]
          mod_val = int(val) #list_mapping[int(val)+1][-1]
          print(mod_name + ',' + str(mod_val) + '\n')
          sub.write(mod_name + ',' + str(mod_val) + '\n')
      print('done!')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default='/hdd/trainval/', type=str, help="carnet data root")
    p.add_argument("--batch_size", default=1, type=int, help="batch size")
    p.add_argument("--lr", default=1e-1, type=float, help="learning rate")
    p.add_argument("--load_epoch", default=18, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--train", default=False, type=bool, help="whether to train a model")
    p.add_argument("--test", default=True, type=bool, help="whether to test a model")
    args = p.parse_args()
    main(args.batch_size, args.root, args.lr, args.load, args.load_epoch, args.train, args.test)

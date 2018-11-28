import csv
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import io
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from se_resnet import se_resnet_custom
from utils import Runner 


def build_image_label_pairs(folders, data_path):
    # Loads the CSV for converting 23 classes to 3 classes
    with open('classes.csv', 'r') as class_key:
      reader = csv.reader(class_key)
      list_mapping = list(reader)

    image_label_pairs = []
    # Iterate over the chosen folders
    for folder in folders:
        for file_name in os.listdir(os.path.join(data_path, folder)):
            if ".jpg" in file_name:
                # Get the ID for the image
                key_id = file_name.split('_')[0]

                # Check that the label exist
                if os.path.exists(os.path.join(data_path,file_name,key_id+'_bbox.bin')):
                  label_data = np.fromfile(os.path.join(data_path,file_name,key_id+'_bbox.bin'), dtype=np.float32)
                else:
                  label_data = [0]*10 # Doesn't exist, must be test, set to 0

                # Append items to dataset
                class_label = int(list_mapping[int(label_data[9])+1][-1])
                image_label_pairs.append((os.path.join(data_path,folder,file_name), class_label))
    return image_label_pairs 

    
class CarDataset(Dataset):
    def __init__(self, image_label_pairs, transforms):
        """This Dataset takes in a folder data_path, the frequency with which to include samples in validation, and a flag for validation."""
        self.image_label_pairs = image_label_pairs 
        self.transforms = transforms
        
    def __getitem__(self, index):
        im_path, im_class = self.image_label_pairs[index]
        image_obj = Image.open(im_path) # Open image
        transformed_image = self.transforms(image_obj) # Apply transformations
        transformed_image.permute(2,0,1) # Swap color channels
        return (im_path,
               torch.tensor(transformed_image).float(),
               torch.from_numpy(np.array(im_class)).long())

    def __len__(self):
        return len(self.image_label_pairs)


def make_dataloader(folder_names, data_path, batch_size):
    # Declare the transforms
    preprocessing_transforms = transforms.Compose([transforms.RandomResizedCrop(1024),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[.362, .358, .347],
                                                                 std=[.139, .130, .123])])

    # Create the datasets
    pairs = build_image_label_pairs(folder_names, data_path)
    dataset = CarDataset(pairs, preprocessing_transforms)

    # Create the dataloaders
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True
    )


#def main(batch_size, data_path, lr, load_dir, load_epoch, train, testing):
def main(args):
    # List the trainval folders
    trainval_folder_names = [x for x in os.listdir(args.trainval_data_path)
                    if os.path.isdir(os.path.join(args.trainval_data_path, x))]

    # Figure out how many folders to use for training and validation
    num_train_folders = int(len(trainval_folder_names) * args.trainval_split_percentage)
    num_val_folders = len(trainval_folder_names) - num_train_folders
    print("Building dataset split...")
    print("--- Number of train folders: {} ---".format(num_train_folders))
    print("--- Number of val folders: {} ---".format(num_val_folders))

    # Choose the training and validation folders
    random.shuffle(trainval_folder_names) # TODO if loading a model, be careful
    train_folder_names = trainval_folder_names[:num_train_folders]
    val_folder_names = trainval_folder_names[num_train_folders:]

    # Make dataloaders
    print("Making dataloaders...")
    train_loader = make_dataloader(train_folder_names, args.trainval_data_path, args.batch_size)
    val_loader = make_dataloader(val_folder_names, args.trainval_data_path, args.batch_size)

    # Specify the GPUs to use
    print("Finding GPUs...")
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))

    # Build the model to run
    print("Building a model...")
    se_resnet = nn.DataParallel(se_resnet_custom(num_classes=3), device_ids=gpus)

    # Load an existing model, be careful with train/validation
    if args.load_epoch > 0:
        print("Loading a model...")
        details = torch.load(args.load_dir + "/model_epoch_{}.pth".format(str(args.load_epoch)))

        # Saving models can be weird, so be careful using these
        #new_details = dict([(k[7:], v) for k, v in details['weight'].items()])
        #new_details = dict([("module."+k, v) for k, v in details['weight'].items()])
        new_details = dict([(k, v) for k, v in details['weight'].items()])
        se_resnet.load_state_dict(new_details)

    # Declare the optimizer, learning rate scheduler, and training loops. Note that models are saved to the current directory.
    optimizer = optim.Adam(params=se_resnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)#, 30, gamma=0.1)

    # This trainer class does all the work
    runner = Runner(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    if args.train:
        runner.loop(args.num_epoch, train_loader, val_loader, scheduler, args.batch_size)

    if args.test:
        # Get test folder names
        test_folder_names = [x for x in os.listdir(args.test_data_path)
                        if os.path.isdir(os.path.join(args.test_data_path, x))]
        
        # Switch to eval mode
        se_resnet.eval()

        # Make test dataloader
        test_loader = make_dataloader(test_folder_names, args.test_data_path, 1)

        # Run the dataloader through the neural network
        outputs, _ = runner.test(test_loader, 1)

        # Write the submission to CSV
        with open('submission_task1.csv', 'w') as sub:
            sub.write('guid/image,label\n')
            for name, val in outputs:
                # Build path
                mod_name = name[0].split('/')[3] + '/' + name[0].split('/')[4].split('_')[0]
                mod_val = int(val)

                # Print and write row
                print(mod_name + ',' + str(mod_val))
                sub.write(mod_name + ',' + str(mod_val) + '\n')
        print('Done!')


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trainval_data_path", default='/hdd/trainval/', type=str, help="carnet trainval data_path")
    p.add_argument("--test_data_path", default='/hdd/test/', type=str, help="carnet test data_path")
    p.add_argument("--trainval_split_percentage", default=0.85, type=float, help="percentage of data to use in training")

    p.add_argument("--batch_size", default=7, type=int, help="batch size")
    p.add_argument("--lr", default=1e-1, type=float, help="learning rate")
    p.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")

    p.add_argument("--load_dir", default='models/v11/', type=str, help="what model version to load")
    p.add_argument("--load_epoch", default=-1, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--num_epoch", default=300, type=int, help="number of epochs to train")
    p.add_argument("--train", default=True, type=bool, help="whether to train a model")
    p.add_argument("--test", default=False, type=bool, help="whether to test a model")
    args = p.parse_args()
    main(args)

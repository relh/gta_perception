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
    """This function takes in a set of folders and their root path. It returns a list 
    of tuples of (image paths, class label) where class label is either 0,1,2 as in classes.csv"""

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
                if os.path.exists(os.path.join(data_path,folder,key_id+'_bbox.bin')):
                  label_data = np.fromfile(os.path.join(data_path,folder,key_id+'_bbox.bin'), dtype=np.float32)
                else:
                  label_data = [0]*10 # Doesn't exist, must be test, set to 0

                # Append items to dataset
                class_label = int(list_mapping[int(label_data[9])+1][-1])
                image_label_pairs.append((os.path.join(data_path,folder,file_name), class_label))
    return image_label_pairs 

    
class CarDataset(Dataset):
    def __init__(self, image_label_pairs, transforms):
        """This Dataset takes in image and label pairs (tuples) and a list of transformations to apply 
        and returns tuples of (image_path, transformed_image_tensor, label_tensor)"""
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
    """This function takes in a list of folders with images in them,
    the root directory of these images, and a batchsize and turns them into a dataloader"""

    # Declare the transforms
    preprocessing_transforms = transforms.Compose(
                                  [transforms.Resize(384),
                                    transforms.ColorJitter(brightness=0.2,
                                                           contrast=0.2,
                                                           saturation=0.2,
                                                           hue=0.2),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomAffine(15.0,
                                                            translate=(0.1, 0.1),
                                                            scale=(0.8,1.2),
                                                            shear=15.0,
                                                            fillcolor=0),
                                    transforms.RandomRotation(degrees=90),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[.362, .358, .347],
                                                         std=[.139, .130, .123])])

    # Create the datasets
    pairs = build_image_label_pairs(folder_names, data_path)
    dataset = CarDataset(pairs, preprocessing_transforms)

    # Create the dataloaders
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=24,
        shuffle=True
    )


def main(args):
    """This major function controls finding data, splitting train and validation data, building datasets,
    building dataloaders, building a model, loading a model, training a model, testing a model, and writing
    a submission"""

    # List the trainval folders
    print("Load trainval data...")
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
    print("Making train and val dataloaders...")
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
    print("Creating optimizer and scheduler...")
    optimizer = optim.Adam(params=se_resnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)#, 30, gamma=0.1)

    # This trainer class does all the work
    print("Instantiating runner...")
    runner = Runner(se_resnet, optimizer, F.cross_entropy, save_dir=".")
    if args.train:
        print("Begin training...")
        runner.loop(args.num_epoch, train_loader, val_loader, scheduler, args.batch_size)

    if args.test:
        print("Load test data...")
        # Get test folder names
        test_folder_names = [x for x in os.listdir(args.test_data_path)
                        if os.path.isdir(os.path.join(args.test_data_path, x))]
        
        # Switch to eval mode
        se_resnet.eval()

        # Make test dataloader
        print("Making test dataloaders...")
        test_loader = make_dataloader(test_folder_names, args.test_data_path, args.batch_size)

        # Run the dataloader through the neural network
        print("Conducting a test...")
        outputs, _ = runner.test(test_loader, args.batch_size)

        # Write the submission to CSV
        print("Writing a submission to \"submission_task1.csv\"...")
        with open('submission_task1.csv', 'w') as sub:
            sub.write('guid/image,label\n')
            for name, val in outputs:
                # Build path
                mod_name = name.split('/')[3] + '/' + name.split('/')[4].split('_')[0]
                mod_val = int(val)

                # Print and write row
                print(mod_name + ',' + str(mod_val))
                sub.write(mod_name + ',' + str(mod_val) + '\n')
        print('Done!')


if __name__ == '__main__':
    """This block parses command line arguments and runs the training/testing main block"""
    print("Parsing arguments...")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trainval_data_path", default='/hdd/trainval/', type=str, help="carnet trainval data_path")
    p.add_argument("--test_data_path", default='/hdd/test/', type=str, help="carnet test data_path")
    p.add_argument("--trainval_split_percentage", default=0.80, type=float, help="percentage of data to use in training")

    p.add_argument("--batch_size", default=24, type=int, help="batch size")
    p.add_argument("--lr", default=1e-2, type=float, help="learning rate")
    p.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")

    p.add_argument("--load_dir", default='models/v22', type=str, help="what model version to load")
    p.add_argument("--load_epoch", default=8, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--num_epoch", default=300, type=int, help="number of epochs to train")
    p.add_argument("--train", default=True, type=bool, help="whether to train a model")
    p.add_argument("--test", default=True, type=bool, help="whether to test a model")
    args = p.parse_args()
    main(args)

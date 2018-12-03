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


def get_classes_to_label_map():
    # Loads the CSV for converting 23 classes to 3 classes
    with open('classes.csv', 'r') as class_key:
      reader = csv.reader(class_key)
      list_mapping = list(reader)[1:]

    new_list_mapping = {}
    for i, x in enumerate(list_mapping):
      new_list_mapping[i] = int(x[-1])
    return new_list_mapping


def build_image_label_pairs(folders, data_path, task):
    """This function takes in a set of folders and their root path. It returns a list 
    of tuples of (image paths, class label) where class label is either 0,1,2 as in classes.csv"""

    list_mapping = get_classes_to_label_map()

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
                if task == 1:
                  # Index 0 is 23 classes, -1 is 3 classes 
                  class_label = int(label_data[9])
                elif task == 2:
                  class_label = [int(x) for x in label_data[3:6]]

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


def make_dataloader(folder_names, data_path, batch_size, task):
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
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[.362, .358, .347],
                                                         std=[.139, .130, .123])])

    # Create the datasets
    pairs = build_image_label_pairs(folder_names, data_path, task)
    dataset = CarDataset(pairs, preprocessing_transforms)

    # Create the dataloaders
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(batch_size/2),
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
    train_loader = make_dataloader(train_folder_names, args.trainval_data_path, args.batch_size, args.task)
    val_loader = make_dataloader(val_folder_names, args.trainval_data_path, args.batch_size, args.task)

    # Specify the GPUs to use
    print("Finding GPUs...")
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))

    # Build the model to run
    print("Building a model...")
    if args.task == 1:
      se_resnet = nn.DataParallel(se_resnet_custom(size=args.model_num_blocks,
                                                   dropout_p=args.dropout_p, num_classes=23),
                                                   device_ids=gpus)
    elif args.task == 2:
      se_resnet = nn.DataParallel(se_resnet_custom(size=args.model_num_blocks,
                                                   dropout_p=args.dropout_p, num_classes=3),
                                                   device_ids=gpus)
      # TODO make this use MSE and have 3 heads, one for X,Y,Z

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    print("Building 23 to 3 class mapper...")
    list_mapping = get_classes_to_label_map()

    print("Declaring multi_loss function...")
    def sum_cross_entropy(inp, target):
      new_p_vals = torch.zeros(inp.shape[0], 3).cuda() # TODO hard coded
      new_t_vals = target.clone()

      for x in range(inp.shape[1]): # For each class currenetly existing
        new_p_vals[:, list_mapping[x]] += inp[:, x] # Mapping to the new class

      for x in range(inp.shape[0]):
        new_t_vals[x] = list_mapping[int(target[x])]

      return F.cross_entropy(inp, target) + F.cross_entropy(new_p_vals, new_t_vals)

    # This trainer class does all the work
    print("Instantiating runner...")
    runner = Runner(se_resnet, optimizer, sum_cross_entropy, args.save_dir)
    if "train" in args.modes.lower():
        print("Begin training...")
        runner.loop(args.num_epoch, train_loader, val_loader, scheduler, args.batch_size)

    if "test" in args.modes.lower():
        print("Load test data...")
        # Get test folder names
        test_folder_names = [x for x in os.listdir(args.test_data_path)
                        if os.path.isdir(os.path.join(args.test_data_path, x))]
        
        # Switch to eval mode
        se_resnet.eval()

        # Make test dataloader
        print("Making test dataloaders...")
        test_loader = make_dataloader(test_folder_names, args.test_data_path, args.batch_size, args.task)

        # Run the dataloader through the neural network
        print("Conducting a test...")
        outputs, _ = runner.test(test_loader, args.batch_size)

        # Write the submission to CSV
        print("Writing a submission to \"submission_task1.csv\"...")
        with open('submission_task1.csv', 'w') as sub:
            sub.write('guid/image,label\n')
            for name, val in outputs:
                print(val)
                # Build path
                mod_name = name.split('/')[3] + '/' + name.split('/')[4].split('_')[0]
                mod_val = int(list_mapping[int(val)])

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

    # Increasing these adds regularization
    p.add_argument("--batch_size", default=25, type=int, help="batch size")
    p.add_argument("--dropout_p", default=0.0, type=float, help="final layer p of neurons to drop")
    p.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")

    # Increasing this increases model ability 
    p.add_argument("--model_num_blocks", default=1, type=int, help="how deep the network is")
    p.add_argument("--lr", default=1e-1, type=float, help="learning rate")

    p.add_argument("--save_dir", default='models/v28', type=str, help="what model dir to save")
    p.add_argument("--load_dir", default='models/v28', type=str, help="what model dir to load")
    p.add_argument("--load_epoch", default=-1, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--num_epoch", default=100, type=int, help="number of epochs to train")
    p.add_argument("--modes", default="Train|Test", type=str, help="string containing modes")

    p.add_argument("--task", default=1, type=int, help="whether to test a model")
    args = p.parse_args()
    main(args)

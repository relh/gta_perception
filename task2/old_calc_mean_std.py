import os
import random
from pdb import set_trace

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def build_image_label_pairs(folders, data_path, task):
    """This function takes in a set of folders and their root path. It returns a list 
    of tuples of (image paths, class label) where class label is either 0,1,2 as in classes.csv"""

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
                if task == 2:
                  class_label = label_data[3:6]
                else:
                  # Index 0 is 23 classes, -1 is 3 classes 
                  class_label = int(label_data[9])

                image_label_pairs.append((os.path.join(data_path,folder,file_name), class_label))
    return image_label_pairs 

    
class CarDataset(Dataset):
    def __init__(self, image_label_pairs, transforms, isTrain = False):
        """This Dataset takes in image and label pairs (tuples) and a list of transformations to apply 
        and returns tuples of (image_path, transformed_image_tensor, label_tensor)"""
        self.image_label_pairs = image_label_pairs 
        self.transforms = transforms
        self.isTrain = isTrain
    def __getitem__(self, index):
        im_path, im_class = self.image_label_pairs[index]
        image_obj = Image.open(im_path) # Open image

        transformed_image = self.transforms(image_obj) # Apply transformations
        transformed_image.permute(2,0,1) # Swap color channels
        #transformed_image_np = transformed_image.numpy()
        #if self.isTrain :
        #    transformed_image = torch.tensor(add_noise_to_image(transformed_image.numpy())).float()
        return (im_path,
               torch.tensor(transformed_image).float(),
               torch.from_numpy(np.array(im_class)))

    def __len__(self):
        return len(self.image_label_pairs)


def make_dataloader(folder_names, data_path, batch_size, task, isTrain = False):
    """This function takes in a list of folders with images in them,
    the root directory of these images, and a batchsize and turns them into a dataloader"""
    # added flag isTrain - only augment/transform training set, not validation/test set

    # Declare the transforms
    preprocessing_transforms = transforms.Compose([transforms.ToTensor()])

    # Create the datasets
    pairs = build_image_label_pairs(folder_names, data_path, task)
    dataset = CarDataset(pairs, preprocessing_transforms, isTrain)

    # Create the dataloaders
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=int(batch_size/2),
        shuffle=True
    )

    
def compute_mean_std(loader):
    """This function computes the mean and standard deviation of the 3 image channels for all the images"""
    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.

    for i, data_tup in enumerate(loader):
        print("{} / {}".format(i, len(loader)))
        
        data = data_tup[2]
        batch_samples = data.size(0)
        data = data.float()
      	#data = data.view(batch_samples, data.size(1), -1)
        
        mean += torch.mean(data, 0)
        #std += torch.mean(torch.std(data, 1), 1)
        nb_samples += batch_samples
    
    mean /= nb_samples
    std /= nb_samples
    return mean, std


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
    train_loader = make_dataloader(train_folder_names, args.trainval_data_path, args.batch_size, args.task, True)
    
    return train_loader

if __name__ == "__main__":
    print("Parsing arguments...")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trainval_data_path", default='/home/relh/.kaggle/trainval/', type=str, help="carnet trainval data_path")
    p.add_argument("--test_data_path", default='/home/relh/.kaggle/test/', type=str, help="carnet test data_path")
    p.add_argument("--trainval_split_percentage", default=0.95, type=float, help="percentage of data to use in training")

    # Increasing these adds regularization
    p.add_argument("--batch_size", default=10, type=int, help="batch size")
    p.add_argument("--dropout_p", default=0.40, type=float, help="final layer p of neurons to drop")
    p.add_argument("--weight_decay", default=1e-3, type=float, help="weight decay")

    # Increasing this increases model ability 
    p.add_argument("--model_num_blocks", default=3, type=int, help="how deep the network is")
    p.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    p.add_argument("--momentum", default=0.9, type=float, help="momentum value")

    p.add_argument("--save_dir", default='models/v73', type=str, help="what model dir to save")
    p.add_argument("--load_dir", default='models/v72', type=str, help="what model dir to load")
    p.add_argument("--load_epoch", default=-1, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--num_epoch", default=30, type=int, help="number of epochs to train")
    p.add_argument("--modes", default="Train|Test", type=str, help="string containing modes")

    p.add_argument("--task", default=2, type=int, help="what task to train a model, or pretrained model")
    args = p.parse_args()

    print('--- dataset creator ---')
    train_loader = main(args)
    
    print('--- mean + std calc ---')
    mean, std = compute_mean_std(train_loader)
    print(mean)
    print(std)

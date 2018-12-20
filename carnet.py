import csv
import os
import pickle
import random
import traceback
from pdb import set_trace

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from skimage import io
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import xml.etree.ElementTree as ET

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import Runner, sum_cross_entropy, get_classes_to_label_map, sum_mse

from cnn_finetune import make_model

#pip install dependencies from https://github.com/aleju/imgaug
import imgaug as ia
from imgaug import augmenters as iaa

# print("Building 23 to 3 class mapper...")
# from utils import list_mapping

def add_noise_to_image(image):
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
    iaa.SomeOf((0, 5),
                [
                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        #iaa.MedianBlur(k=(3, 11)),
                    ]),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255)# , per_channel=0.5
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    # iaa.OneOf([
                    #     iaa.Dropout((0.05, 0.2), per_channel=0.5),
                    #     iaa.CoarseDropout(
                    #         (0.03, 0.15), size_percent=(0.02, 0.05),
                    #         per_channel=0.2
                    #     ),
                    # ]),
        iaa.CoarseDropout((0, 0.15), size_percent=(0.02, 0.25)),

        #iaa.ElasticTransformation(alpha=(2.5, 5.0), sigma=0.25),
        iaa.SaltAndPepper(0.15, False),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    #iaa.Grayscale(alpha=(0.0, 1.0)),

                    # In some images move pixels locally around (with random
                    # strengths).
                    # sometimes(
                    #     iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    # ),

                    # In some images distort local areas with varying strength.
                    # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
                # do all of the above augmentations in random order
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )
    image = (image * 255).astype('uint8')
    return ((seq.augment_images(image))/255.0).astype('float64')

def build_image_label_pairs(names, data_path, task, xml=False):
    """This function takes in a set of folders or images 'names' and their root path. It returns a list 
    of tuples of (image paths, class label) where class label is either 0,1,2 as in classes.csv"""

    missing = 0
    image_label_pairs = []
    if xml:
        # Iterative over images
        for img in names:
            name = img[:-4]
            if os.path.exists(os.path.join(data_path,'Annotations',name+'.xml')):
                whole_name = os.path.join(data_path,'Annotations',name+'.xml')
                e = ET.parse(whole_name).getroot()
                class_name = e[5][0].text
            else:
                #print('annotation for img', img,' does not exist!') # this should never happen!
                missing += 1
                continue
            
            # Append items to dataset
            if task == 2:
                class_label = 0
                if class_name == 'car':
                    class_label = 1
                elif class_name == 'bus' or class_name == 'motorbike':
                    class_label = 2
            else:
                # Index 0 is 23 classes, -1 is 3 classes 
                class_label = 0
                if class_name == 'car':
                    class_label = np.random.randint(7) + 1 # we're not sure, to reduce bias randomly choose car class
                elif class_name == 'motorbike':
                    class_label = 9
                elif class_name == 'bus':
                    class_label = 12
                
            image_label_pairs.append((os.path.join(data_path,'JPEGImages',img), class_label))
    else:
        # Iterate over the chosen folders
        for folder in names:
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
                      class_label = [int(x) for x in label_data[3:6]]
                    else:
                      # Index 0 is 23 classes, -1 is 3 classes 
                      class_label = int(label_data[9])

                    image_label_pairs.append((os.path.join(data_path,folder,file_name), class_label))
    print('Number of missing annotations... {}'.format(missing))
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
        #transformed_image_np = transformed_image.numpy()
        #transformed_image = torch.tensor(add_noise_to_image(transformed_image.numpy())).float()
        return (im_path,
               torch.tensor(transformed_image).float(),
               torch.from_numpy(np.array(im_class)).long())

    def __len__(self):
        return len(self.image_label_pairs)


def make_dataloader(names, data_path, batch_size, task, modes, xml=False):
    """This function takes in a list of folders with images in them,
    the root directory of these images, and a batchsize and turns them into a dataloader"""
    # added flag isTrain - only augment/transform training set, not validation/test set

    data_augmentation = [transforms.ColorJitter(brightness=0.2,
             contrast=0.2,
             saturation=0.2,
             hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15.0,
              translate=(0.1, 0.1),
              scale=(0.8,1.2),
              shear=15.0,
              fillcolor=0)]
    # Declare the transforms
    preprocessing_transforms = [transforms.Resize((384, 682)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[.362, .358, .347],
                                                         std=[.139, .130, .123])]

    # Create the datasets
    pairs = build_image_label_pairs(names, data_path, task, xml)

    if 'train' in modes.lower():
      dataset = CarDataset(pairs, transforms.Compose(data_augmentation + preprocessing_transforms))

      # Create the dataloaders
      return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=int(batch_size/2),
    shuffle=True
      )
    elif 'test' in modes.lower():
      dataset = CarDataset(pairs, transforms.Compose(preprocessing_transforms))

      # Create the dataloaders
      return DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=int(batch_size/2),
    shuffle=False
      )


def build_model(args, gpus):
    # Build the model to run
    print("Building a model...")
    if args.task == 1:
      #from se_resnet import se_resnet_custom
      #model = nn.DataParallel(se_resnet_custom(size=args.model_num_blocks,
      #                                             dropout_p=args.dropout_p, num_classes=23),
      #                                             device_ids=gpus)
      pass # TODO make model here
    elif args.task == 2:
      # TODO make this use MSE and have 3 heads, one for X,Y,Z
      model = make_model(args.model, num_classes=3, dropout_p=args.dropout_p, pretrained=True)
    elif args.task == 3 or args.task == 4:
      model = make_model(args.model, num_classes=23, dropout_p=args.dropout_p, pretrained=True)
      #model = make_model('resnet18', num_classes=23, dropout_p=args.dropout_p, pretrained=True)
      #model = make_model('resnext101_32x4d', num_classes=23, dropout_p=args.dropout_p, pretrained=True)

    return model

def load_model(args, model, load_epoch):
    # Load an existing model, be careful with train/validation
    if load_epoch > 0:
        print("Loading a model...")
        existing_models = os.listdir(args.load_dir)
        model_to_load = "model_epoch_{}.pth".format(str(load_epoch))
        if model_to_load not in existing_models:
          print("Load Epoch Not Found!!!")
          model_to_load = random.choice(existing_models)

        details = torch.load(args.load_dir + "/" + model_to_load)

        # Saving models can be weird, so be careful using these
        new_details = dict([(k, v) for k, v in details['weight'].items()])
        model.load_state_dict(new_details)
    return model


def main(args):
    """This major function controls finding data, splitting train and validation data, building datasets,
    building dataloaders, building a model, loading a model, training a model, testing a model, and writing
    a submission"""
    best_acc = 0

    # Specify the GPUs to use
    print("Finding GPUs...")
    gpus = list(range(torch.cuda.device_count()))
    print('--- GPUS: {} ---'.format(str(gpus)))

    if "train" in args.modes.lower():
        # List the trainval folders
        print("Load trainval data...")
        trainval_folder_names = [x for x in os.listdir(args.trainval_data_path)
                        if os.path.isdir(os.path.join(args.trainval_data_path, x))]
        more_train_img_names = [x for x in os.listdir(os.path.join(args.more_train_data_path, 'JPEGImages'))]

        # Figure out how many folders to use for training and validation
        num_train_folders = int(len(trainval_folder_names) * args.trainval_split_percentage)
        num_more_train_imgs = len(more_train_img_names)
        num_val_folders = len(trainval_folder_names) - num_train_folders
        print("Building dataset split...")
        print("--- Number of train folders: {} ---".format(num_train_folders))
        print("--- Number of additional train images: {} ---".format(num_more_train_imgs))
        print("--- Number of val folders: {} ---".format(num_val_folders))

        # Choose the training and validation folders
        random.shuffle(trainval_folder_names) # TODO if loading a model, be careful
        train_folder_names = trainval_folder_names[:num_train_folders]
        val_folder_names = trainval_folder_names[num_train_folders:]

        # Make dataloaders
        print("Making train and val dataloaders...")
        train_loader = make_dataloader(train_folder_names, args.trainval_data_path, args.batch_size, args.task, args.modes)
        more_train_loader = make_dataloader(more_train_img_names, args.more_train_data_path, args.batch_size, args.task, args.modes, xml=True)
        val_loader = make_dataloader(val_folder_names, args.trainval_data_path, args.batch_size, args.task, args.modes)

    # Build and load the model
    model = build_model(args, gpus)
    model = load_model(args, model, args.load_epoch)

    # Declare the optimizer, learning rate scheduler, and training loops. Note that models are saved to the current directory.

    print("Creating optimizer and scheduler...")
    if args.task == 4:
      if args.optimizer_string == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      elif args.optimizer_string == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      elif args.optimizer_string == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      elif args.optimizer_string == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      elif args.optimizer_string == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
      else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=10, verbose=True)
    else:
      optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    # This trainer class does all the work
    print("Instantiating runner...")
    if args.task == 2:
        runner = Runner(model, optimizer, sum_mse, args.task, args.save_dir,args.task)
    else:
        runner = Runner(model, optimizer, sum_cross_entropy, args.save_dir)
    best_acc = 0

    if "train" in args.modes.lower():
        print("Begin training... {}, lr:{} + wd:{} + opt:{} + bs:{} "
              .format(str(args.model), str(args.lr), str(args.weight_decay), str(args.optimizer_string), str(args.batch_size)))
        best_acc = runner.loop(args.num_epoch, train_loader, more_train_loader, val_loader, scheduler, args.batch_size)

    args.save_path = save_path = args.save_dir.split('/')[-1] + '-' + args.model + '-' + str(best_acc) + '-' + str(args.lr) + '-' + str(args.weight_decay) + '-' + str(args.optimizer_string) + '-' + str(args.batch_size)

    if "test" in args.modes.lower():
        print("Load test data...")
        # Get test folder names
        test_folder_names = [x for x in os.listdir(args.test_data_path)
                        if os.path.isdir(os.path.join(args.test_data_path, x))]
        
        # Switch to eval mode
        model = build_model(args, gpus)
        model = load_model(args, model, 9999)
        model.eval()

        # Make test dataloader
        print("Making test dataloaders...")
        test_loader = make_dataloader(test_folder_names, args.test_data_path, args.batch_size, args.task, 'test')

        # Run the dataloader through the neural network
        print("Conducting a test...")
        _, _, outputs, logits = runner.test(test_loader, args.batch_size)

        # Write the submission to CSV
        print("Writing a submission to \"csvs/{}.csv\"...".format(save_path))

        if args.task == 2:
	        with open('csvs/'+save_path+'.csv', 'w') as sub:
	          sub.write('guid/image/axis,value\n')
	          for name, val in outputs:
	              # Build path
	              mod_name = name.split('/')[5] + '/' + name.split('/')[6].split('_')[0]
	              x = val[0]
	              y = val[1]
	              z = val[2]

	              # Print and write row
	              sub.write(mod_name + '/x,' + str(x) + '\n')
	              sub.write(mod_name + '/y,' + str(y) + '\n')
	              sub.write(mod_name + '/z,' + str(z) + '\n')
	        np.save('logits/'+save_path+'.npy', np.array([l for p,l in logits]))

        else:
	        with open('csvs/'+save_path+'.csv', 'w') as sub:
	          sub.write('guid/image,label\n')
	          for name, val in outputs:
	              # Build path
	              mod_name = name.split('/')[3] + '/' + name.split('/')[4].split('_')[0]
	              mod_val = int(list_mapping[int(val)])

	              # Print and write row
	              sub.write(mod_name + ',' + str(mod_val) + '\n')

        np.save('logits/'+save_path+'.npy', np.array([l for p,l in logits]))

        # TODO average multiple logits results
        # This function loads these logits but they should be reshaped with .reshape(-1, 23)
        # test_logits = np.load('logits/'+save_path+'.npy')
        #print("0s: {}".format(str(np.count_nonzero(test_logits == 0.0)))) 
        #print("1s: {}".format(str(np.count_nonzero(test_logits == 1.0)))) 
        #print("2s: {}".format(str(np.count_nonzero(test_logits == 2.0)))) 
        print('Done!')


if __name__ == '__main__':
    """This block parses command line arguments and runs the training/testing main block"""
    print("Parsing arguments...")
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--trainval_data_path", default='/home/ubuntu/trainval/', type=str, help="carnet trainval data_path")
    p.add_argument("--more_train_data_path", default='/home/ubuntu/more_train/', type=str, help="more train data_path")
    p.add_argument("--test_data_path", default='/home/ubuntu/test/', type=str, help="carnet test data_path")
    p.add_argument("--trainval_split_percentage", default=0.90, type=float, help="percentage of data to use in training")

    # Increasing these adds regularization
    p.add_argument("--batch_size", default=20, type=int, help="batch size")
    p.add_argument("--dropout_p", default=0.35, type=float, help="final layer p of neurons to drop")
    p.add_argument("--weight_decay", default=8e-2, type=float, help="weight decay")

    # Increasing this increases model ability 
    p.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    p.add_argument("--momentum", default=0.9, type=float, help="momentum value")

    p.add_argument("--save_dir", default='models/v888', type=str, help="what model dir to save")
    p.add_argument("--load_dir", default='models/v72', type=str, help="what model dir to load")
    p.add_argument("--load_epoch", default=6, type=int, help="what epoch to load, -1 for none")
    p.add_argument("--num_epoch", default=6, type=int, help="number of epochs to train")
    p.add_argument("--modes", default='Train|Test', type=str, help="string containing modes")

    p.add_argument("--task", default=4, type=int, help="what task to train a model, or pretrained model")
    p.add_argument("--model", default='inception_v4', type=str, help="what pretrained model to start with")
    p.add_argument("--optimizer_string", default='Adam', type=str, help="what optimizer string")
    args = p.parse_args()

    main(args)

    '''
    # Output rewriting
    for f in os.listdir('./csvs/'):
        if len(f.split('-')) < 2 or 'DEFAULT' in f:
          continue
        args.model = f.split('-')[1]
        args.batch_size = 5
        args.load_dir = '/hdd/models/'+f.split('-')[0]
        args.load_epoch = 9999
        args.save_dir = 'models/'+f.split('-')[0] 
        print(f)
        print(args.model)

        try:
          main(args)
        except Exception as e:
          print('Oops failed!')
          traceback.print_exc()
    

    # Random model search
    model_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                    'densenet121', 'densenet169', 'densenet201', 'densenet161',
                    'inception_v3',
                    'alexnet', 'xception'
                    'nasnetalarge',
                    'nasnetamobile', 'pnasnet5large',
                    'inceptionresnetv2', 'polynet']
                    #'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']
    main(args)

    # for i in range(100):
    #   args.save_dir = 'models/v' + str(210 + i)
    #   args.load_dir = 'models/v' + str(210 + i)
    #   args.batch_size = 5 # To be not that safe
    #   args.model = random.choice(model_list)
    #   try:
    #     main(args)
    #   except Exception as e:
    #     print('Oops failed!')
    #     traceback.print_exc()

    for i in range(100):
      args.save_dir = 'models/v' + str(505 + i)
      args.load_dir = 'models/v' + str(505 + i)
      args.batch_size = 10 # To be not that safe

      # Random search
      args.model = random.choice(model_list)
      args.lr = random.choice([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-2])
      args.weight_decay = random.choice([0, 0, 0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 2e-2, 1e-1])
      args.optimizer_string = random.choice(['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta'])
      args.batch_size = random.choice([10,12,14,16,18,20,25])


      try:
        main(args)
      except Exception as e:
        print('Oops failed!')
        traceback.print_exc()
    '''

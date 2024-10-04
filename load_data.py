import torch
import torch.utils.data as D
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
import os
from PIL import Image
from config import Configs

class Read_data(D.Dataset):
    def __init__(self, base_dir, file_label, set, split_size, augmentation=True, flipped=False):
        self.base_dir = base_dir
        self.file_label = file_label
        self.set = set
        self.split_size = split_size
        self.augmentation = augmentation
        self.flipped = flipped

    def __getitem__(self, index):
        try:
            img_name = self.file_label[index]
            idx, deg_img, gt_img = self.readImages(img_name)
            # Print debug information to verify
            print(f"Successfully loaded image {img_name} at index {index}")
            return idx, deg_img, gt_img
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            raise NotImplementedError(f"Could not process index {index}. Error: {e}")

    def __len__(self):
        return len(self.file_label)

    def readImages(self, file_name):
        """
        Read a pair of images (degraded + clean gt)

        Args:
            file_name (str): the index (name) of the image pair
        Returns:
            file_name (str): the index (name) of the image pair
            out_deg_img (np.array): the degraded image
            out_gt_img (np.array): the clean image
        """
        url_deg = self.base_dir + '/' + self.set + '/' + file_name
        url_gt = self.base_dir + '/' + self.set + '_gt/' + file_name

        # Debugging information to verify file paths
        print(f"Reading degraded image: {url_deg}")
        print(f"Reading ground truth image: {url_gt}")

        # Read images using OpenCV
        deg_img = cv2.imread(url_deg)
        gt_img = cv2.imread(url_gt)

        if deg_img is None or gt_img is None:
            raise ValueError(f"Cannot find image: {url_deg} or {url_gt}")

        if self.flipped:
            deg_img = cv2.rotate(deg_img, cv2.ROTATE_180)
            gt_img = cv2.rotate(gt_img, cv2.ROTATE_180)

        deg_img = Image.fromarray(np.uint8(deg_img))
        gt_img = Image.fromarray(np.uint8(gt_img))

        # Resize images to ensure consistency
        deg_img = TF.resize(deg_img, [256, 256])
        gt_img = TF.resize(gt_img, [256, 256])

        # Normalize data
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert to numpy arrays and normalize
        deg_img = np.array(deg_img).astype('float32') / 255.0
        gt_img = np.array(gt_img).astype('float32') / 255.0

        for i in range(3):
            deg_img[:, :, i] = (deg_img[:, :, i] - mean[i]) / std[i]
            gt_img[:, :, i] = (gt_img[:, :, i] - mean[i]) / std[i]

        # Debugging output for shapes
        print(f"Processed image shapes: deg_img={deg_img.shape}, gt_img={gt_img.shape}")

        return file_name, deg_img, gt_img



def load_datasets(flipped=False):
    """
    Create the 3 datasets (train/valid/test) to be used by the dataloaders.

    Args:
        flipped (bool): whwther to flip the images of the val dataset (was used
                        in 1 experiment to check the effect of flipping)
    Returns:
        data_train (Dateset): train data
        data_valid (Dateset): valid data
        data_test (Dateset): test data
    """
    cfg = Configs().parse() 
    base_dir = cfg.data_path
    split_size  = cfg.split_size
    data_tr = os.listdir(cfg.data_path + 'train')
    np.random.shuffle(data_tr)
    data_va = os.listdir(cfg.data_path+'valid')
    np.random.shuffle(data_va)
    data_te = os.listdir(cfg.data_path+'test')
    np.random.shuffle(data_te)
    
    data_train = Read_data(base_dir, data_tr, 'train', split_size, augmentation=True)
    data_valid = Read_data(base_dir, data_va, 'valid', split_size, augmentation=False, flipped = flipped)
    data_test = Read_data(base_dir, data_te, 'test', split_size, augmentation=False)

    return data_train, data_valid, data_test

def sort_batch(batch):
    """
    Transform a batch of data to pytorch tensor

    Args:
        batch [str, np.array, np.array]: a batch of data
    Returns:
        data_index (tensor): the indexes of the source/target pair
        data_in (tensor): the source images (degraded)
        data_out (tensor): the target images (clean gt)
    """
    n_batch = len(batch)
    data_index = []
    data_in = []
    data_out = []

    # Determine the shape of the first element to enforce consistency
    target_shape = batch[0][1].shape

    for i in range(n_batch):
        idx, img, gt_img = batch[i]

        # Check if shapes are consistent, if not, resize to target shape
        if img.shape != target_shape:
            img = np.resize(img, target_shape)
        if gt_img.shape != target_shape:
            gt_img = np.resize(gt_img, target_shape)

        data_index.append(idx)
        data_in.append(img)
        data_out.append(gt_img)

    data_index = np.array(data_index)
    data_in = np.array(data_in, dtype='float32')
    data_out = np.array(data_out, dtype='float32')

    data_in = torch.from_numpy(data_in)
    data_out = torch.from_numpy(data_out)

    return data_index, data_in, data_out

def all_data_loader(batch_size):
    """
    Create the 3 data loaders

    Args:
        batch_size (int): the batch_size
    Returns:
        train_loader (dataloader): train data loader  
        valid_loader (dataloader): valid data loader
        test_loader (dataloader): test data loader
    """
    data_train, data_valid, data_test = load_datasets()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader
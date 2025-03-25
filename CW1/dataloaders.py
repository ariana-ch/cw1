import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, no_photos, augment: bool = False, test: bool = False):
        '''
         Custom dataset class to sample an identity and return no_photos of that identity

        :param no_photos: Integer, the number of images to return per sample
        :param augment: Whether the images should be transformed. Note that at test time any transforms should be deterministic
        :para test: If test is False, the identities are selected from the first 1768 (=8000-32) of the
         identities, and they are shuffled before being drawn. If true then the last 32 identities are used
        '''
        self.root_dir = Path('./data/casia-webface/')
        self.no_photos = no_photos
        augment_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomResizedCrop((112, 112), scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # need to write another function to un-normalise in order to visualise.
        ])
        self.test = test
        self.image_generating_transforms = augment_transforms
        self.default_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                                      transforms.CenterCrop((112, 112)),
                                                      transforms.ToTensor(),
                                                      # -> torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                                                      transforms.ConvertImageDtype(torch.float32)])
        self.augment_transforms = augment_transforms if augment else self.default_transforms
        self.augment = augment
        self.__set_labels(test=test)

    def __set_labels(self, test: bool):
        labels = sorted([x for x in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, x))])
        if test:
            self.labels = labels[-32:]
        else:
            labels = labels[:-32]
            np.random.shuffle(labels)
            self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        identity = self.labels[idx]
        img_paths = [os.path.join(self.root_dir, identity, x) for x in
                     os.listdir(os.path.join(self.root_dir, identity))]

        if len(img_paths) >= self.no_photos:
            if self.test:
                img_paths = img_paths[:self.no_photos]
            else:
                img_paths = np.random.choice(img_paths, size=self.no_photos, replace=False)
            extra = 0
        else:
            extra = self.no_photos - len(img_paths)

        img_list = []
        for img_path in img_paths:
            img = Image.open(img_path)  # already in RGB here.
            img_tensor = self.augment_transforms(img)
            img_list.append(img_tensor)

        if extra:
            for i in range(extra):
                img_path = np.random.choice(img_paths)
                img = Image.open(img_path)
                img_tensor = self.image_generating_transforms(img)
                img_list.append(img_tensor)

        img_tensor = torch.stack(img_list, dim=0)  # shape N, C, H, W
        img_tensor = img_tensor.permute(0, 2, 3, 1)
        label_tensor = torch.from_numpy(np.repeat(idx, self.no_photos).astype(np.int32))
        return label_tensor, img_tensor


def collate_fn(batch):
    '''
    Our dataloader returns samples of shape (no_images, H, W, C) which means the data loader returns batches of shape (batch_size, no_images, H, W, C) 
    by default. This collate function stacks the datasets together.
    
    :param batch: tuple of torch tensors of shapes (no_people, no_images) and (no_people, no_images, H, W, C)
    :return: tuple of torch tensors of shape (no_people x no_images, ) and (no_people x no_images, H, W, C)
    '''
    labels, images = zip(*batch)
    labels = torch.cat(labels, dim=0)
    images = torch.cat(images, dim=0)
    return labels, images


# overwrite the previous method - not clean code but this is a notebook so ...
def get_dataloader(no_people: int = 32, no_photos: int = 4, img_transforms: Optional[list] = None,
                   shuffle: bool = False, augment: bool = False, collate_fn=collate_fn, **kwargs):
    '''
    Function to get the dataloader for the dataset

    :param no_people: Number of people to sample
    :param no_photos: Number of photos to return per person
    :param img_transforms: an optional list of transformations to be applied to the image (after it is converted to a tensor so ensure compatibility)
    :param shuffle: If true, samples are shuffled
    :param augment: If True the images are augmented
    :param collate_fn: Collate function. This is used to transform the shape of the batch. By default, given our dataset, the batches would consist of a 2d tuple with dimensions:
        (no_people, no_photos) <-- labels
        (no_people, no_photos, 112, 112, 3) <-- images
    both are fine since we are anyway writing a custom training loop, but we added the collate function to be in line with the exercise
    :return: the dataloader to be used for training the model
    '''
    dataset = ImageDataset(no_photos=no_photos, augment=augment, test=not shuffle)
    return torch.utils.data.DataLoader(dataset, batch_size=no_people, shuffle=shuffle, drop_last=True,
                                       collate_fn=collate_fn, **kwargs)
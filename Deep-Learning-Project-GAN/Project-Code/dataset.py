from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
import torch


def dataset_split_shape(name, n = None, size = 32, grayScale = False, convert_tensor=True, transform_data = True, root='./datasets/', download=True):
  
  dataset_dir = os.path.join(root, f"{name}_{n}_{size}")
  if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

  if transform_data:
    transform_list = [transforms.ToTensor(),
                      transforms.Resize(size)]
    if grayScale:
      transform_list.append(transforms.Grayscale()) 
    if convert_tensor:
      transform_list.append(transforms.Normalize((0.5, ), (0.5, )))
  else:
    transform_list = []

  transformer = transforms.Compose(transform_list)

  if name == 'CIFAR10': 
    dataset = datasets.CIFAR10(
      root=dataset_dir,
      download=download,
      transform=transforms.Compose(transform_list))
  elif name == 'CIFAR100': 
    dataset = datasets.CIFAR100(
      root=dataset_dir,
      download=download,
      transform=transforms.Compose(transform_list)
  )
  elif name == 'STL10': ## Choose unlabelled dataset here
    dataset = datasets.STL10(
      root=dataset_dir,
      download=download,
      split='unlabeled',
      transform=transforms.Compose(transform_list)
  )
  elif name == 'FashionMNIST': 
    dataset = datasets.FashionMNIST(
      root=dataset_dir,
      download=download,
      transform=transforms.Compose(transform_list)
  )
  else:
    print("invalid name")
    return 
  if n is None or n > len(dataset):
    return dataset
  generator1 = torch.Generator().manual_seed(42)
  a, b = random_split(dataset, [n, len(dataset)-n], generator = generator1)
  return a
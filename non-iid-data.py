import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader,Subset,random_split
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
import torch.nn.functional as F

import numpy as np


from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import partition_report

num_clients = 10
num_classes = 100

num_rounds = 35
num_local_train = 1

seed=2
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root="./CIFAR100/",
                                        train=True, download=True,transform=transform)
val_size = int(0.2 * len(trainset))  # 20% for validation
train_size = len(trainset) - val_size
_, valset = random_split(trainset, [train_size, val_size])

hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                     num_clients,
                                     balance=None,
                                     partition="dirichlet",
                                     dir_alpha=0.3,seed=seed)
subset_list = [Subset(trainset, hetero_dir_part.client_dict[i]) for i in range(num_clients)]
loader_list = [DataLoader(subset_list[i], batch_size=128, shuffle=True) for i in range(num_clients)]
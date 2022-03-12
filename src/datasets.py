import random
import torch
import numpy as np
from torch.utils.data import Dataset, Subset

#----------------------------------------------------------------------------

class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

#----------------------------------------------------------------------------
    
def make_dataset(dataset, images_per_class=5):
    labeled_indices = []
    
    # randomly sample labeled indices
    for i in range(0, 9):
        indices = random.sample(np.where(np.array(dataset.targets) == i)[0].tolist(), images_per_class)
        labeled_indices = labeled_indices + indices
    
    # recover the corresponding labels
    labeled_targets = [dataset.targets[i] for i in labeled_indices]
    
    # create the labeled subset
    labeled_dataset = torch.utils.data.Subset(dataset, labeled_indices)
    labeled_dataset = DatasetFromSubset(labeled_dataset, transform=None)
    
    # length of original dataset
    total_indices = list(range(0,100000))
    
    # indices of unlabeled dataset
    unlabeled_indices = [index for index in total_indices if index not in labeled_indices]
    
    # create the unlabeled subset
    unlabeled_dataset = torch.utils.data.Subset(dataset, unlabeled_indices)
    unlabeled_dataset = DatasetFromSubset(unlabeled_dataset, transform=None)
    
    return labeled_targets, labeled_dataset, unlabeled_dataset
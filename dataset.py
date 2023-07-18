import torch
import random
import numpy as np
from torch.utils.data import Dataset



class StimuliDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            random_floats = [random.random() for _ in range(6)]
            max_salient_index = random_floats.index(max(random_floats))
            
            random_cue  = random.randrange(2)
            diff_shape_index = random.randint(0, 5)
            shape_list = [0] * 6
            shape_list[diff_shape_index] = 1
            x = random_floats + shape_list
            x.append(random_cue)
            label = []
            if (random_cue > 0):
                label.append(diff_shape_index) #with topdown modualation cue, chooses the different shape
            else:
                label.append(max_salient_index) #no topdown modulation cue, chooses the most salient object
            '''
            label = []
            label.append(max_salient_index)
            self.data.append(random_floats)
            self.labels.append(label)
            
            '''
            self.data.append(x)
            self.labels.append(label)
            

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        return sample, label
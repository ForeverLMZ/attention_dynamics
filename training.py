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
            matrix = np.empty((6, 4))
            random_index = random.randint(0, 5)
            random_floats = [random.random() for _ in range(6)]
            max_salient_index = random_floats.index(max(random_floats))
            for j in range(6):
                random_cue  = random.randrange(2)
                random_float = random_floats[j]
                matrix[j][0] = j
                matrix [j][1] = random_float
                if (j == random_index):
                    matrix [j][2] = 1
                else:
                    matrix[j][2] = 0
                matrix[j][3] = random_cue
            x = matrix
            if (random_cue > 0):
                label = random_index #with topdown modualation cue, chooses the different shape
            else:
                label = max_salient_index #no topdown modulation cue, chooses the most salient object
            self.data.append(x)
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.int)

        return sample, label








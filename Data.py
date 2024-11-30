from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class SpectralDataset(Dataset):
    def __init__(self, label_list):
        self.label_list = label_list  # A list like [(filename/filepath, label), ...]

    def __getitem__(self, index):
        spectrum, concentration = self.label_list[index]
        spectrum = csv_to_tensor(spectrum)
        return spectrum, concentration

    def __len__(self):
        return len(self.label_list)


def csv_to_tensor(file_path):
    df = pd.read_csv(file_path, header=None)
    array = np.array(df)
    tensor = torch.tensor(array, dtype=torch.float64)
    return tensor

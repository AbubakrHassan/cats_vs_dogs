import torch
from torch.utils.data import Dataset
import os
from skimage import io


class CvDDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.files[idx]
        img_name = os.path.join(self.root_dir, file_name)
        if file_name.startswith("dog"):
            label = 1
        else:
            label = 0

        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image, label

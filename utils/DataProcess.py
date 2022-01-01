import cv2
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import numpy as np

import torchvision

class DCE_MRI_2d(Dataset):
    def __init__(self):

        self.data = []
        self.mask = []
        self.len = len(self.data)
        self.read_data()

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

    def __len__(self):
        return self.len

    def read_data(self):
        # 读取D:/GroupProjectDataSet/QIN Breast DCE-MRI/dicom/下的所有文件夹
        list_dataset = glob.glob("D:/GroupProjectDataSet/QIN Breast DCE-MRI/niigz/DCE-MRI-*/data/*")
        for name in list_dataset:
            img = nib.load(name).get_data()
            img = np.array(img)
            z_len = img.shape[2]
            for i in range(z_len):
                self.data.append(img[:, :, i])

        list_dataset = glob.glob("D:/GroupProjectDataSet/QIN Breast DCE-MRI/niigz/DCE-MRI-*/mask/*")
        for name in list_dataset:
            img = nib.load(name).get_data()
            img = np.array(img)
            z_len = img.shape[2]
            for i in range(z_len):
                self.mask.append(img[:, :, i])
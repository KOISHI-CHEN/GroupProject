import cv2
import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import numpy as np
import SimpleITK as sitk

class DCE_MRI_2d(Dataset):
    def __init__(self):

        self.data = []
        self.mask = []
        self.len = len(self.data)
        self.read_data()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.from_numpy(self.mask[index])

    def __len__(self):
        return self.len

    def read_data(self):
        # 读取D:/GroupProjectDataSet/QIN Breast DCE-MRI/dicom/下的所有文件夹
        list_dataset = glob.glob("D:/GroupProjectDataSet/QIN_Breast_DCE-MRI/niigz/DCE-MRI-*/data/*")
        print(list_dataset)
        for name in list_dataset:
            itk_img = sitk.ReadImage(name)
            img = sitk.GetArrayFromImage(itk_img)
            z_len = img.shape[0]
            for i in range(z_len):
                self.data.append(img[i, :, :])

        list_dataset = glob.glob("D:/GroupProjectDataSet/QIN_Breast_DCE-MRI/niigz/DCE-MRI-*/mask/*")
        print(list_dataset)
        for name in list_dataset:
            itk_img = sitk.ReadImage(name)
            img = sitk.GetArrayFromImage(itk_img)
            z_len = img.shape[0]
            for i in range(z_len):
                self.mask.append(img[i, :, :])


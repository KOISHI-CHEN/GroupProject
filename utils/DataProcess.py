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

        self.Data = []
        self.Label = []
        self.read_data()
        self.len = len(self.Data)

    def __getitem__(self, index):
        # return self.Data[index], self.Label[index]
        return {'image': self.Data[index], 'mask' : self.Label[index]}

    def __len__(self):
        return self.len

    def read_data(self):
        list_dataset = glob.glob("../niigz/DCE-MRI-*/data/*")
        print(list_dataset)
        for name in list_dataset:
            itk_img = sitk.ReadImage(name)
            img = sitk.GetArrayFromImage(itk_img)
            z_len = img.shape[0]
            for i in range(50, 70):
                print([img[i,:,:]])
                self.Data.append((torch.from_numpy(img[i, :, :] / 1.0)).type(torch.cuda.FloatTensor))

        list_dataset = glob.glob("../niigz/DCE-MRI-*/mask/*")
        print(list_dataset)
        for name in list_dataset:
            itk_img = sitk.ReadImage(name)
            img = sitk.GetArrayFromImage(itk_img)
            z_len = img.shape[0]
            for i in range(50, 70):
                self.Label.append((torch.from_numpy(img[i, :, :]/ 1.0)).type(torch.cuda.FloatTensor))

if __name__ == "__main__":
    dataset = DCE_MRI_2d()
    train_loader = DataLoader(dataset, batch_size=2)
    toimage = transforms.ToPILImage()

    for batch in train_loader:
        imag = batch['image']
        imag = imag[0].unsqueeze(0)
        imag = toimage(imag)
        imag.save('output.jpg')
        print(imag)
        break


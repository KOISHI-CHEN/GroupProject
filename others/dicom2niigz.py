import dicom2nifti
import os
import cv2
import glob



if __name__ == "__main__":
    output_folder = "D:/GroupProjectDataSet/QIN Breast DCE-MRI/niigz/DCE-MRI-"
    list = glob.glob("D:/GroupProjectDataSet/QIN Breast DCE-MRI/dicom/DCE-MRI*/data/57*")
    i = 1
    for folder in list:
        out_name = "D:/GroupProjectDataSet/QIN Breast DCE-MRI/niigz/DCE-MRI-{0}/data/57".format(i)
        dicom2nifti.dicom_series_to_nifti(folder, out_name, reorient_nifti=True)
        print(folder)
        i = i + 1
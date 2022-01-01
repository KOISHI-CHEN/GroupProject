import dicom2nifti
import os
import cv2
import glob



if __name__ == "__main__":
    output_folder = "D:/GroupProjectDataSet/QIN_Breast_DCE-MRI/niigz/DCE-MRI-"
    list = glob.glob("D:/GroupProjectDataSet/QIN_Breast_DCE-MRI/dicom/DCE-MRI*/data/57*")
    i = 1
    for folder in list:
        index = folder[56:58]
        if folder[57:58] == '\\':
            index = folder[56:57]
        print(index)
        out_name = "D:/GroupProjectDataSet/QIN_Breast_DCE-MRI/niigz/DCE-MRI-{0}/data/DCE_MRI-{1}-57".format(index, index)
        dicom2nifti.dicom_series_to_nifti(folder, out_name, reorient_nifti=True)
        print(folder)
        i = i + 1
import src.images_processing as PrImg
import os
import cv2 as cv
from matplotlib import pyplot as plt

ROOT_DIR = '/home/legokna/Proyects/python/lsm/images/raw'
PROS_DIR = '/home/legokna/Proyects/python/lsm/images/processed'
CLASSES = ['A', 'B', 'C', 'D','E','F','G','H','I','L','M','N','O','P','R','S','T','U','V','W','Y']
aux_CLASSES = ['A', 'B', 'C','D']
length = len(aux_CLASSES)
prs_img = PrImg

def create_directories():
    for i in range (length):
        new_dir1 = ROOT_DIR + '/'+CLASSES[i]
        new_dir2 = PROS_DIR + '/'+CLASSES[i]
        os.mkdir(new_dir1)
        os.mkdir(new_dir2)

def load_data_training(path_class):            
    dir = ROOT_DIR + path_class
    for filename in os.listdir(dir):                
        current_img_name = '/'+filename
        img = prs_img.processing_image(dir,current_img_name)      
        if img is not None:
            pr_path = PROS_DIR +path_class+current_img_name
            prs_img.save_img(img,pr_path)                    

def check_data():   
    for i in range (length):
        path_class = '/'+CLASSES[i]
        path_dir = PROS_DIR + path_class
        if not os.listdir(path_dir):
            load_data_training(path_class)
        else:
            print ("not empty")




#create_directories()
check_data()

"""
prs_img = PrImg
current_img_name = '/test.jpg'
prs_img.processing_image(ROOT_DIR,current_img_name)
"""
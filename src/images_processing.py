from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

SIZE = (32, 32)
ROOT_DIR = '/home/legokna/Proyects/python/lms/images/raw'
CLASSES = ['A', 'B', 'C', 'D']

#Funcion para mostrar el proceso de el tratamiento de imagenes
def show_imgs(titles,images):    
    for i in range(6):
        plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    cv.waitKey(0)
    cv.destroyAllWindows()

#Funcion para invertir colores de una imagen
def negative_img(img_src):
    not_img = cv.bitwise_not(img_src)
    return not_img

##Funcion para obtener los bordes de una imagen usando operadores Sobel
def sobel_edge(img_src):
    grad_x = cv.Sobel(img_src,cv.CV_64F,1,0,ksize=3)
    grad_y = cv.Sobel(img_src,cv.CV_64F,0,1,ksize=3)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)
    grad = negative_img(grad)
    return grad

#Funcion para el tratamiento de la imagen
def processing_image(name):
    #Lectura de la imagen en escala de grises
    img_name = ROOT_DIR+name
    img_src = cv.imread(img_name)
    img_src = cv.cvtColor(img_src,cv.COLOR_BGR2RGB)
    img = cv.imread(img_name, 0)
    #Conversion de la imagen en una imagen Binaria
    ret, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    img_th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv.THRESH_BINARY, 11, 2)
    #Contono de la imagen
    img_edged = sobel_edge(img_bin)
    #Redimensionar la imagen
    img_resized = cv.resize(img_edged, SIZE, interpolation=cv.INTER_AREA)
    #Mostrar los resultados
    titles = ['Source Image','GrayScale', 'Binary', 'TH3','EDGY','Resized']
    images = [img_src,img,img_bin, img_th3, img_edged,img_resized]
    show_imgs(titles,images)
    #Guardar la imagen procesada
    img_final_path = '/home/legokna/Proyects/python/lms/images/processed'+name
    cv.imwrite(img_final_path,img_resized)
    # plt.imshow(img_bin,cmap="gray")
    

current_img_name = '/test.jpg'
processing_image(current_img_name)

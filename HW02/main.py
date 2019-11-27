import numpy as np
import cv2
from util import *

def get_Gaussian_kernel(size=5,sigma=1):
    kernel=np.zeros((size,size))
    c=size//2
    for i in range(size):
        for j in range(size):
            ss=(i-c)**2+(j-c)**2
            kernel[i,j]=np.exp(-ss/(2*sigma**2))
    kernel_sum=np.sum(kernel)
    return kernel/kernel_sum

def get_Laplasian_kernel():
    kernel=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=float)
    return kernel

def LoG(image,thresh,gaussian=True):
    # Gaussian
    if gaussian:
        image = conv2D(image, get_Gaussian_kernel())
    # Laplasian
    image_after = conv2D(image, get_Laplasian_kernel())
    print('max: ', np.max(image_after), ' min:', np.min(image_after))
    # thresholding
    image_after[image_after > thresh] = 255
    image_after[image_after <= thresh] = 0
    image_after = image_after.astype(np.uint8)
    return image_after

def prewitt(image):
    px=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=float)
    py=np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=float)
    x=conv2D(image,px)
    y=conv2D(image,py)
    res=np.abs(x)+np.abs(y)/2
    res[res>255]=255
    res=res.astype(np.uint8)
    return res


if __name__=='__main__':
    npic=input('which pic to choose(1~6)?  ')
    image=cv2.imread('Prog1_images/p1im{}.bmp'.format(npic),0)
    image_after=LoG(image,thresh=8,gaussian=False)
    cv2.imshow('Laplacian',image_after)
    image_after = LoG(image, thresh=8, gaussian=True)
    cv2.imshow('LOG', image_after)
    image_after=prewitt(image)
    cv2.imshow('prewitt',image_after)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

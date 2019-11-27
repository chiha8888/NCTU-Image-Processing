import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
from main import get_Gaussian_kernel,get_Laplasian_kernel,LoG

def HTLine(image,stepTheta=1,stepRho=1):
    '''
    :param image: 0,255 binary image
    :param stepTheta:
    :param stepRho: accumulator,accDict:what point in accumlator[i,j]
    :return:
    '''
    row,cols=image.shape
    L=int(np.sqrt(row**2+cols**2)+1)
    numTheta=180//stepTheta
    numRho=2*L//stepRho+1

    accumulator=np.zeros((numRho,numTheta))
    accDict={}
    for i in range(numRho):
        for j in range(numTheta):
            accDict[(i,j)]=[]

    for i in range(row):
        for j in range(cols):
            if image[i,j]==255:
                for k in range(numTheta):
                    rho=int(j*math.cos(k*stepTheta*math.pi/180)+i*math.sin(k*stepTheta*math.pi/180))
                    accumulator[(rho+L)//stepRho,k]+=1
                    accDict[(rho+L)//stepRho,k].append((j,i))
    return accumulator,accDict



if __name__=='__main__':
    npic = input('which pic to choose (1~6)? ')
    image_origin = cv2.imread('Prog1_images/p1im{}.bmp'.format(npic))
    image_after=cv2.cvtColor(image_origin,cv2.COLOR_BGR2GRAY)
    image_after=LoG(image_after,thresh=8)
    cv2.imshow('LOG', image_after)

    accumulator,accDict=HTLine(image_after)
    print('accumulator max={:.0f}'.format(np.max(accumulator)))
    thresh=100
    for i in range(accumulator.shape[0]):
        for j in range(accumulator.shape[1]):
            if accumulator[i,j]>thresh:
                points=accDict[(i,j)]
                cv2.line(image_origin,points[0],points[-1],(0,255,0),1)
    cv2.imshow('image+line',image_origin)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
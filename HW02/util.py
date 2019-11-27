import numpy as np

def cross_correlation(m1,m2):
    return np.sum(m1*m2)

def conv2D(image,kernel):
    H,W=image.shape
    fh,fw=kernel.shape
    image_pad=np.pad(image,((fh//2,fh//2),((fw//2,fw//2))),'symmetric')
    #print('image shape:',image.shape,' image_padding shape:',image_pad.shape)
    res=np.zeros(image.shape,float)
    for i in range(H):
        for j in range(W):
            res[i,j]=cross_correlation(image_pad[i:i+fh,j:j+fw],kernel)
    return res

#!/usr/bin/python

import os.path as osp
import os, glob, re
import random
import time
from PIL import Image
import numpy as np
#from scipy.misc import imsave
import ntpath
#import skimage
from skimage.transform import resize
from skimage.io import imread

images = './cancer_data/'



def next_batch(batch, train=True):
    output = []
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    
    choice = ['neg','pos']

    for _ in range(batch):
        type_image = random.choice(choice)
        
        if train:
            image = random.choice(glob.glob(images +'inputs/' + type_image + '_train' + '*.png' ))
        else:
            image = random.choice(glob.glob(images +'inputs/' + type_image + '_test' + '*.png' ))
    
        #print image

        img_in = np.array(Image.open(image).resize((512/2,512/2)))
        img_out = imread(images+'outputs/'+ntpath.basename(image))#Image.open(images+'outputs/'+ntpath.basename(image)).resize((512,512))
        img_out = resize(img_out, (512/2,512/2))
        

        img_out =  np.stack((img_out.astype(int), (np.invert(img_out.astype(int)) + 2) ),axis=2)
        
        
        output[0].append(img_in.tolist())
        output[1].append(img_out.tolist())

    return output


if __name__ == '__main__':
    num = 10
    b = next_batch(num)
    print len(b[1]), len(b[1][0]), len(b[1][0][0]), len(b[1][0][0][0])
    #for i in range(num):
    #    im = np.array(b[1][i])
    #    im = im.reshape([512,512])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

#!/usr/bin/python

import os.path as osp
import os, glob, re
import random
import time
from PIL import Image
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import ntpath


images = './cancer_data/'



def next_batch(batch):
    output = []
    output.append([])
    output.append([])
    output.append([])
    output.append([])
    
    choice = ['neg','pos']

    for _ in range(batch):
        type_image = random.choice(choice)
        #print type_image
        #print images + type_image + '*.png'
        #print glob.glob(images i+'/inputs/'+ type_image + '*.png')
        image = random.choice(glob.glob(images +'inputs/' + type_image + '*.png' ))
        #print image
        #print ntpath.basename(image)
    

        img_in = np.array(Image.open(image).resize((512,512)))
        img_out = np.array(Image.open(images+'outputs/'+ntpath.basename(image)).resize((512,512)))

        #print img_in.shape
        #print img_out.shape

        output[0].append(img_in.tolist())
        output[1].append(img_out.tolist())

    return output


if __name__ == '__main__':
    num = 10
    b = next_batch(num)
    #print len(b[0]), len(b[1])
    #for i in range(num):
    #    im = np.array(b[0][i])
    #    im = im.reshape([100,100])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

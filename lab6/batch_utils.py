#!/usr/bin/python

import os.path as osp
import os, glob, re
import random
import time
from PIL import Image
import numpy as np
#from scipy.misc import imread, imresize, imsave, imshow
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
        #print type_image
        #print images + type_image + '*.png'
        #print glob.glob(images i+'/inputs/'+ type_image + '*.png')
        if train:
            image = random.choice(glob.glob(images +'inputs/' + type_image + '_train' + '*.png' ))
        else:
            image = random.choice(glob.glob(images +'inputs/' + type_image + '_test' + '*.png' ))
        #print image
        #print ntpath.basename(image)
    

        img_in = imread(image)#np.array(Image.open(image).resize((512,512)))
        img_out = imread(images+'outputs/'+ntpath.basename(image))#Image.open(images+'outputs/'+ntpath.basename(image)).resize((512,512))
        #img_out = img_out.convert('L')
        #img_out = np.array(img_out)
        img_out = resize(img_out, (512,512))
        

        #print img_in.tolist()
        print np.concimg_out.astype(int)
        print (np.invert(img_out.astype(int)) + 2).tolist()[0]


        
        output[0].append(img_in.tolist())
        output[1].append(img_out.tolist())

    return output


if __name__ == '__main__':
    num = 1
    b = next_batch(num)
    #print len(b[0]), len(b[1])
    #for i in range(num):
    #    im = np.array(b[0][i])
    #    im = im.reshape([100,100])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

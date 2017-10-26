#!/usr/bin/python

import os, glob, re
import random
import time
from PIL import Image
import numpy as np


images = './img_align_celeba/'



def next_batch(batch, train=True):
    output = []
    output.append([])
    


    for _ in range(batch):
        
        
        image = random.choice(glob.glob(images+'/*'))
    
        #print image
        # img_in = np.array(Image.open(image).resize((16,16)))
        img_in = Image.open(image)

        new_width = 150
        new_height = 150
        width, height = img_in.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        img_in = img_in.crop((left, top, right, bottom)).resize((16,16))
        img_in.show()
        img_in = np.array(img_in)
       	img_in = np.ndarray.flatten(img_in)
	 
        output[0].append(img_in.tolist())

    return output


if __name__ == '__main__':
    num = 10
    b = next_batch(num)
    
    #for i in range(num):
    #    im = np.array(b[1][i])
    #    im = im.reshape([512,512])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

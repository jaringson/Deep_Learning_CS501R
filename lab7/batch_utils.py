#!/usr/bin/python

import os, glob, re
import random
import time
from PIL import Image
import numpy as np


images = './img_align_celeba/'



def next_batch(batch,input_i):
    output = []
    output.append([])
    


    for i in range(batch):
        
       	input_i = 0 
        #image = random.choice(glob.glob(images+'/*'))
    	image = ""
	if input_i*batch + i+1 >= 10:
	    image = images+"0000"+str(input_i* batch+i+1)+".jpg"
	else:
	    image = images+"00000"+str(input_i*batch+i+1)+".jpg" 
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

        img_in = img_in.crop((left, top, right, bottom)).resize((32, 32))
        #img_in.show()
        img_in = np.array(img_in)
       	img_in = np.ndarray.flatten(img_in)
	img_in = img_in/255.0 
        output[0].append(img_in.tolist())

    return output


if __name__ == '__main__':
    num = 10
    b = next_batch(num)
    print b[0] 
    #for i in range(num):
    #    im = np.array(b[1][i])
    #    im = im.reshape([512,512])
    #    imsave(str(i)+'.png', im)
    #    #imshow(im)

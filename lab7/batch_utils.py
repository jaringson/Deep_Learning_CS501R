#!/usr/bin/python

import os, glob, re
import random
import time
from PIL import Image
import numpy as np


images = './img_align_celeba/'



def next_batch(batch):
    output = []
    output.append([])
    

    N = 202496

    for i in range(batch):
        
	place = random.randrange(0, N)
	img_in = None
	while img_in is None:
	    try:
		image = ""
		if place  >= 100000:
		    image = images+str(place)+".jpg"
		elif place >= 10000:
		    image = images+"0"+str(place)+".jpg"
		elif place >= 1000:
		    image = images+"00"+str(place)+".jpg"
		elif place >= 100:
		    image = images+"000"+str(place)+".jpg"
		elif place  >= 10:
		    image = images+"0000"+str(place)+".jpg"

		else:
		    image = images+"00000"+str(place)+".jpg" 
		img_in = Image.open(image)
	    except:
		place = (place + 1) % N
		

        new_width = 150
        new_height = 150
        width, height = img_in.size   # Get dimensions

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        img_in = img_in.crop((left, top, right, bottom)).resize((64, 64))
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

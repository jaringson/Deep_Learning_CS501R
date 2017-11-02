from PIL import Image
import numpy as np
from scipy.misc import imsave, imshow
import glob

#all_images  = glob.glob('./q/*.png')
#print all_images
d = './p/'
out = 'samples_2'
'''
pic = []
steps = 5
for i in range(10):
    pic_sub = []
    for j in range(steps):
        print d+str(j*steps+i)+'.png'
        if j ==0:
            pic_sub = Image.open(d+str(j*steps+i)+'.png')
        else:
            img = Image.open(d+str(j*steps+i)+'.png')
            pic_sub = np.concatenate((pic_sub,img), 1)
    if i == 0:
        pic = pic_sub
    else:
        pic = np.concatenate((pic, pic_sub), 0)

'''

img1 = Image.open('samples.png')
img2 = Image.open('samples_.png')
pic = np.concatenate((img1,img2),1)

imsave(out+'.png',pic)



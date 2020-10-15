#import library
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

#import images
img = cv2.imread('lena.jpg',0)

#create 9x9 filter from p62 from lecture 3
gaus = np.array([[ 0,  1,  2,  1,  0],
                 [ 1,  3,  5,  3,  1],
                 [ 2,  5,  9,  5,  2],
                 [ 1,  3,  5,  3,  1],
                 [ 0,  1,  2,  1,  0]])
#use np.sum function to get the weight of filter
filter1 = np.true_divide(gaus, np.sum(gaus))

#self design a 3x5 filter
gaus2 = np.array([[ 5,  0,  2, 0, -10],
                  [ 0,  6,  1, 6, 0],
                  [ -10,  0,  2, 0, 5]])
#use np.sum function to get the weight of filter
filter2 = np.true_divide(gaus2, np.sum(gaus2))

#use filter2d to apply designed filter
#-1 is the default of center 
#start from center pixel of img
output1 = cv2.filter2D(img,-1,filter1)
output2 = cv2.filter2D(img,-1,filter2)
output3 = gaussian_filter(img,sigma =2)

#--------save the img to file--------#
# cv2.imwrite('gaus9x9.jpg',output1)
# cv2.imwrite('gaus3x5.jpg',output2)
# cv2.imwrite('sigma2.jpg',output3)
#--------save the img to file--------#

#-------------------display everything into one plot-------------------#
#give name for plot called total_imgs, set subplot as 3x2 matrix
total_imgs, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 11))
#0 column display 9x9 filter show on p62 from Lecture 3
axes[0,0].set_title('Original',fontsize =25)
axes[0,0].imshow(img,cmap = 'gray')
axes[0,0].set_xticks([])
axes[0,0].set_yticks([])
axes[0,1].set_title('filter in p62',fontsize =25)
axes[0,1].imshow(output1,cmap = 'gray')
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])

#1 column display self created 3x5 filter
axes[1,0].set_title('Original',fontsize =25)
axes[1,0].imshow(img,cmap = 'gray')
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
axes[1,1].set_title('3x5 filter',fontsize =25)
axes[1,1].imshow(output2,cmap = 'gray')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])

#2 column display gaussian filter, sigma = 2
axes[2,0].set_title('Original',fontsize =25)
axes[2,0].imshow(img,cmap = 'gray')
axes[2,0].set_xticks([])
axes[2,0].set_yticks([])
axes[2,1].set_title('Sigma(σ)=2',fontsize =25)
axes[2,1].imshow(output3,cmap = 'gray')
axes[2,1].set_xticks([])
axes[2,1].set_yticks([])

plt.tight_layout()
plt.show() 

#-------------save the figure to file--------------#
# total_imgs.savefig('P2_Gaussian_output.jpg')
# plt.close(total_imgs) 
#-------------save the figure to file--------------#

#-------------------display everything into one plot-------------------#
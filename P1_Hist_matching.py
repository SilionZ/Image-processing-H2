#import library
import cv2
import os
from skimage import exposure 
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt

#import images
ref = os.path.basename('reference.jpg')
sour = os.path.basename('source.jpg')

#opencv read images with gray scale
ref_img = cv2.imread(ref,0)
sour_img = cv2.imread(sour,0)

#print out resulotion of two images
ref_size = ref_img.shape
sour_size = sour_img.shape
print("Reference size:{}\nSource size: {}".format(ref_size,sour_size))

#blending two images together
combin = cv2.addWeighted(ref_img,0.6,sour_img,0.4,0)
#---------display img-----------#
# cv2.imshow('Combin',combin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#---------display img-----------#

#----------save the image to file----------#
# cv2.imwrite('Match.jpg',combin)
#----------save the image to file----------#

#use match_histograms function to let combin histogram
#match to reference image
match = match_histograms(combin,ref_img,multichannel=True)

#apply cumulative distribution function to plot 
#out cumulative plot,bins value was defaulted as 256
#for match img
match_cdf,bins = exposure.cumulative_distribution(match)
#for source img
sour_cdf,bins = exposure.cumulative_distribution(sour_img)
#for reference img
ref_cdf,bins = exposure.cumulative_distribution(ref_img)



#-------------------display everything into one plot-------------------#
#give name for plot called total_imgs, set subplot as 3x3 matrix
total_imgs, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 11))
#0 column source img, histogram and cumulative distribution of source img
axes[0,0].set_title('Source',fontsize =30)
axes[0,0].imshow(sour_img,cmap = 'gray')
axes[1,0].hist(sour_img.ravel(),256,[0,256])
axes[2,0].plot(bins,sour_cdf,color = 'r')

#1 column ref img, histogram and cumulative distribution of ref img
axes[0,1].set_title('Reference',fontsize =30)
axes[0,1].imshow(ref_img,cmap = 'gray')
axes[1,1].hist(ref_img.ravel(),256,[0,256])
axes[2,1].plot(bins,ref_cdf,color = 'r')

#2 column match img, histogram and cumulative distribution of match img
axes[0,2].set_title('Matched',fontsize =30)
axes[0,2].imshow(combin,cmap = 'gray')
axes[1,2].hist(match.ravel(),256,[0,256])
axes[2,2].plot(bins,match_cdf,color = 'r')
plt.tight_layout()
plt.show() 

#-------------save the figure to file--------------#
# total_imgs.savefig('P1_hist_match_output.jpg')
# plt.close(total_imgs) 
#-------------save the figure to file--------------#

#-------------------display everything into one plot-------------------#


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:04:37 2021

@author: narge
"""

# imports
import glob
import cv2
from skimage import exposure
import matplotlib.pyplot as plt
import random


def show_image(img_name, img):
    while (1):
        cv2.imshow(img_name, img)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


def read_paths(folder_address):
    return glob.glob(folder_address)


def crop(img, x=250, y=250, h=800, w=1200):
    crop_img = img[x:x+h, y:y+w,:] # TODO: we need to crop img 
    return crop_img


def hist_match(src , ref , jj, base_folder, show=True):
    # =============================================================================
    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    #print("[INFO] performing histogram matching...")
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
   
    # show the output images
    # cv2.namedWindow('Source', cv2.WINDOW_NORMAL)
    # show_image('Source', src)
    # cv2.namedWindow('Reference', cv2.WINDOW_NORMAL)
    # show_image('Reference', ref)
    # cv2.namedWindow('Matched', cv2.WINDOW_NORMAL)
    # show_image('Matched', matched)]
  

    # construct a figure to display the histogram plots for each channel
    # before and after histogram matching was applied
    (fig, axs) =  plt.subplots(nrows=3, ncols=3, figsize=(8, 8))
    # loop over our source image, reference image, and output matched
    # image
    for (i, image) in enumerate((src, ref, matched)):
    	# convert the image from BGR to RGB channel ordering
    	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    	# loop over the names of the channels in RGB order
    	for (j, color) in enumerate(("red", "green", "blue")):
    		# compute a histogram for the current channel and plot it
    		(hist, bins) = exposure.histogram(image[..., j] , nbins=41, 
    			source_range="dtype")
    		axs[j, i].plot(bins, hist / hist.max())
    		# compute the cumulative distribution function for the
    		# current channel and plot it
    		(cdf, bins) = exposure.cumulative_distribution(image[..., j] , nbins=41)
    		axs[j, i].plot(bins, cdf)
    		# set the y-axis label of the current plot to be the name
    		# of the current color channel
    		axs[j, 0].set_ylabel(color)
            
    # set the axes titles
    axs[0, 0].set_title("Source")
    axs[0, 1].set_title("Reference")
    axs[0, 2].set_title("Matched")
    # display the output plots
    plt.tight_layout()
    plt.savefig('{}/src_ref_match_hist_fig/{:04d}.png'.format(base_folder, jj))
    if show:
        plt.show()
    
    return matched

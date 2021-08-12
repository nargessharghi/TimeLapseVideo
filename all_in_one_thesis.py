# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 20:02:08 2021

@author: narge
"""
#imports
import cv2
import numpy as np
from all_functions_in_one_thesis import read_paths, show_image, crop, hist_match
from color_adjusting import enhance_frame
import random
import os
from os.path import isfile, join


# Parameters
DO_HOMOGRAPHY= False
DO_CROPPING = False
DO_COLOR_CORRECTION = False
WRITE_VIDEO = True
MOTION_DETECTION = False


ENHANCE = True
enhance_type = "equ" # equ or CLAHE

CENTERING = True
USE_ONE_REF = True # if CENTERING is True
NUM_RANDOM_IMGS = 10 # if CENTERING is True and USE_ONE_REF is False

feature_detector_name = "SIFT" # SIFT or ORB
base_addr = "./Dataset/400_images"
raw_images_folder_addr = base_addr + "/raw_images/*.JPG"


if ENHANCE:
    extra_name = f'_enhance_{enhance_type}'
else:
    extra_name = ""


folder_addr_for_homography_imgs = f'{base_addr}/{feature_detector_name}{extra_name}'

UPDATE_REF = False
UPDATE_REF_NUM = 50
# =================== Start of Code ====================

if DO_HOMOGRAPHY:
    print("Compute Homigraphy section started ........ ")
    all_paths = read_paths(raw_images_folder_addr)
    
    ref_img = cv2.imread(all_paths[0])
    if ENHANCE:
        ref_img_enhance = enhance_frame(ref_img, type= enhance_type)
    
    for j in range(len(all_paths)-1):
        if UPDATE_REF and j%UPDATE_REF_NUM == 0:
            ref_img = cv2.imread(all_paths[j])
            ref_img_enhance = enhance_frame(ref_img, type= enhance_type)
            
        source_img = cv2.imread(all_paths[j+1])
        if ENHANCE:
            source_img_enhance = enhance_frame(source_img, type= enhance_type)  
            ref_img_gray = cv2.cvtColor(ref_img_enhance ,cv2.COLOR_BGR2GRAY)
            source_img_gray = cv2.cvtColor(source_img_enhance , cv2.COLOR_BGR2GRAY)
        else:
            ref_img_gray = cv2.cvtColor(ref_img ,cv2.COLOR_BGR2GRAY)
            source_img_gray = cv2.cvtColor(source_img , cv2.COLOR_BGR2GRAY)
            
            
        if feature_detector_name == "SIFT":
            #initate SIFT
            sift = cv2.SIFT_create(25000)
            kp1, des1 = sift.detectAndCompute(source_img_gray,None)
            kp2, des2 = sift.detectAndCompute(ref_img_gray,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]
            
        else:
            #initate ORB
            orb = cv2.ORB_create(3000) # 3000 at first retval	=	cv.ORB_create(	[, nfeatures[, scaleFactor[, nlevels[, edgeThreshold[, firstLevel[, WTA_K[, scoreType[, patchSize[, fastThreshold]]]]]]]]]	)
            kp1, des1 = orb.detectAndCompute(source_img_gray, None)
            kp2, des2 = orb.detectAndCompute(ref_img_gray, None)
            
            matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
            # match descriptor
            matches = matcher.match(des1, des2, None)
            good_matches = sorted(matches, key=lambda x: x.distance)
        
        points1 = np.zeros((len(good_matches), 2), dtype= np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype= np.float32)
        
        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
    
        
        h, mask = cv2.findHomography(points1, points2, method=cv2.RANSAC )
        
        
        im_out = cv2.warpPerspective(source_img, h, (source_img.shape[1], source_img.shape[0]))
        
        img3 = cv2.drawMatches(source_img, kp1, ref_img, kp2 , good_matches[:50] , None)
        
        
        # cv2.namedWindow("Warped Source Image" , cv2.WINDOW_NORMAL)
        #show_image("Warped Source Image", im_out)
        cv2.imwrite(f'{folder_addr_for_homography_imgs}' + "/{:04d}_Hmgfy_images_{}_{}.JPG".format(j, j+1, j+2), im_out)
        
    
        blended_img = cv2.addWeighted(source_img, 0.5, im_out, 0.5, 0.0)
        # cv2.namedWindow("Blended_image", cv2.WINDOW_NORMAL)
        #show_image("Blended_image", img_avg)
        cv2.imwrite(f'./Dataset/blended_images/Blended_image_{j}_{j+1}.JPG', blended_img)
    
    
# Section for Cropping =========
if DO_CROPPING:
    print("Cropping section started ........ ")
    all_paths = read_paths(f'{folder_addr_for_homography_imgs}/*.JPG')
    
    for j in range(len(all_paths)):
        image1 = cv2.imread(all_paths[j])
        crop_img = crop(image1, x = 550, y = 300, h = 2000, w = 2600)
        cv2.imwrite("{}/cropped/{:04d}_cropped.JPG".format(folder_addr_for_homography_imgs, j), crop_img)
    


# Section for Centering and Spreading =========
if DO_COLOR_CORRECTION:
    print("Centering and Spreading section started ........ ")
    all_paths = read_paths(f'{folder_addr_for_homography_imgs}/cropped/*.JPG')
    
    if CENTERING:
        # centering method
        if USE_ONE_REF:
            # use one image as reference
            ref = cv2.imread(all_paths[0])
            extra_name = ""
            
        else:
            # use bunch of random images and blend them with weght of 0.1 for each to use the result
            # as reference for centering
            random_indexes = []
    
            for i in range (NUM_RANDOM_IMGS):
                x = random.randint(1,len(all_paths))
                random_indexes.append(x)
                
            random_images = []
            for i,path in enumerate(all_paths):
                if i in random_indexes:
                    img = cv2.imread(path)
                    random_images.append(img)
            
            
            blend_images = np.zeros(random_images[0].shape)         
            for img in random_images:
                blend_images = blend_images + (img * 0.1)
            
            blend_images2 = np.array(blend_images, dtype=random_images[0].dtype)
            ref = blend_images2   
            extra_name = "_randoms"
            
        # centering method part
        base_folder = f'{folder_addr_for_homography_imgs}/centering{extra_name}'
        for j in range(len(all_paths)-1):
            src = cv2.imread(all_paths[j+1])    
            matched = hist_match(src, ref, j, base_folder, show=False)
            
            cv2.imwrite("{}/{:04d}.JPG".format(base_folder, j), matched)
            
    else:
        # spreading method part
        base_folder = f'{folder_addr_for_homography_imgs}/spreading'
        ref = cv2.imread(all_paths[0])
        for j in range(len(all_paths)-1):
            src = cv2.imread(all_paths[j+1])
            matched = hist_match(src , ref , j, base_folder, show=False)
            
            cv2.imwrite("{}/{:04d}.JPG".format(base_folder, j), matched)
            
            ref = matched



import numpy as np
import cv2
import time
from color_adjusting import enhance_frame

if CENTERING:
    # centering method
    if USE_ONE_REF:
        extra_name = ""
    else:
        extra_name = "_randoms"
    base_folder = f'{folder_addr_for_homography_imgs}/centering{extra_name}'
else:
    base_folder = f'{folder_addr_for_homography_imgs}/spreading'

pathIn= base_folder + "/*.JPG"
pathOut = base_folder + '/video_out.avi'
   

from cv2 import VideoWriter, VideoWriter_fourcc
WRITE_RAW_VIDEO = True
if WRITE_VIDEO:
    if WRITE_RAW_VIDEO == True:
        base_folder = base_addr + "/raw_images"
    else:
        if CENTERING:
            # centering method
            if USE_ONE_REF:
                extra_name = ""
            else:
                extra_name = "_randoms"
            base_folder = f'{folder_addr_for_homography_imgs}/centering{extra_name}'
        else:
            base_folder = f'{folder_addr_for_homography_imgs}/spreading'
        
    FPS = 15
    pathIn= base_folder + "/*.JPG"
    pathOut = base_folder + '/video_out.avi'
    

    size = (1200 , 960)
    frame_array = []    
    print(f'pathIn is {pathIn}')    
    all_paths = read_paths(pathIn)
    frame_array = []
    for i in range(len(all_paths)):
        img = cv2.imread(all_paths[i])
        img = cv2.resize(img , size)        
        #inserting the frames into an image array
        frame_array.append(img)
    
    fourcc = VideoWriter_fourcc(*'MP42')
    video = VideoWriter(pathOut, fourcc, float(FPS), size)
    
    for img in frame_array:
        # writing to a image array
        video.write(img)
    video.release()
    

if MOTION_DETECTION:    
    path_vid = pathOut
    cap = cv2.VideoCapture(path_vid)
    
    ret , frame1 = cap.read()
    ret , frame2 = cap.read()
    
    referenceFrame = frame1
    
    changedFrame = frame2
    j=0
    adjust_color = False
    while cap.isOpened():
        j+=1
        print(f'j = {j}')
        time.sleep(2)
        if adjust_color:
            enh_frame1 = enhance_frame(frame1, type= 'equ')
            enh_frame2 = enhance_frame(frame2, type= 'equ')
        else:
            enh_frame1 = frame1
            enh_frame2 = frame2
            
        diff = cv2.absdiff(enh_frame1 , enh_frame2)
        gray = cv2.cvtColor(diff , cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25,25) , 0)
        _,thresh = cv2.threshold(blur , 60 , 255 , cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh , None , iterations = 3)
        contours , _ = cv2.findContours(dilated , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            boundRect = cv2.boundingRect(contour)
            (x, y , w , h) = boundRect
            
            counterArea = cv2.contourArea(contour)
            if  counterArea < 2000 or counterArea > 35000:  # counterArea < 2000 or counterArea > 20000
                continue
            
            # if  x,w < 2000 or y,h > 35000:
            #     continue
            
            
            cv2.rectangle(changedFrame ,  (x , y) , (x+w , y+h) , (0 ,255 , 0 ) , 2)
            
            changedFrame[y:y+h,x:x+w] = referenceFrame[y:y+h, x:x+w]
            
            
        #cv2.drawContours(frame1 , contours , -1 ,(0 , 255 ,0) , 2)
        
        #cv2.namedWindow('feed', cv2.WINDOW_NORMAL)
        # cv2.imshow('feed', changedFrame)
        cv2.imwrite(base_folder + "/mothion_detection/{:04d}.JPG".format(j), changedFrame)
        #break
        frame1 = frame2
        ret , frame2 = cap.read()
        changedFrame = frame2
        
        if cv2.waitKey(40) == 27:
            break
    
    cv2.destroyAllWindows()    
    cap.release()
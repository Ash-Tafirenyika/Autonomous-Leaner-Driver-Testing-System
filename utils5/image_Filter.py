# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 04:49:06 2020

@author: Dr~Newt
"""
import numpy as np
import cv2
import math
import glob
from utils5.camera_calib import Camera_Calib

class ImageFilters():
    # Initialize ImageFilter
    def __init__(self, cam_Calib, debug=False):
        self.current_Frame = None
        # returns a copy of the camera calibration data
        self.mtx, self.dist, self.img_size = cam_Calib.get()
        # normal image size
        self.x, self.y = self.img_size
        self.mid = self.mid = int(self.y / 2)
        # current Image RGB - undistorted
        self.current_Image = np.zeros((self.y, self.x, 3), dtype=np.float32)
        # current Image Top half RGB
        self.current_SkyRGB = np.zeros((self.mid, self.x, 3), dtype=np.float32)
        # current Image Bottom half RGB
        self.current_RoadRGB = np.zeros((self.mid, self.x, 3), dtype=np.float32)
        # current Sky Luma Image
        self.current_SkyL = np.zeros((self.mid, self.x), dtype=np.float32)
        # current Road Luma Image
        self.current_RoadL = np.zeros((self.mid, self.x), dtype=np.float32)
        
        # current Edge (Left Only)
        self.current_Road_L_Edge = np.zeros((self.mid, self.x), dtype=np.uint8)
        self.curRoadLEdgeProjected = np.zeros((self.y, self.x, 3), dtype=np.uint8)
        
        self.debug = debug
        
        self.skylrgb = np.zeros((4), dtype=np.float32)
        self.roadlrgb = np.zeros((4), dtype=np.float32)
        self.roadbalance = 0.0
        self.horizonFound = False
        self.roadhorizon = 0
        self.visibility = 0
        

        # Textural Image Info
        self.skyText = 'NOIMAGE'
        self.skyImageQ = 'NOIMAGE'
        self.roadText = 'NOIMAGE'
        self.roadImageQ = 'NOIMAGE'
        
    def dir_sobel(self, gray_img, kernel_size=3, thres=(0, np.pi/2)):
            """
            Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
            and applies a threshold value to only set pixels within the specified range
            """
            sx_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
            sy_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))
            
            dir_sxy = np.arctan2(sx_abs, sy_abs)
        
            binary_output = np.zeros_like(dir_sxy)
            binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1
            
            return binary_output
        
        
    def abs_sobel(self,gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
        """
        Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
        The function also computes the asbolute value of the resulting matrix and applies a binary threshold
        """
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) 
        sobel_abs = np.absolute(sobel)
        sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))
        
        gradient_mask = np.zeros_like(sobel_scaled)
        gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
        return gradient_mask
        
        
    def to_hls(self,img):
        #Returns the same image in HLS format
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    def compute_hls_white_yellow_binary(self,rgb_img):
        #Returns a binary thresholded image produced retaining only white and yellow elements 
        hls_img = self.to_hls(rgb_img)
        
        # Compute a binary thresholded image where yellow is isolated from HLS components
        img_hls_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_yellow_bin[((hls_img[:,:,0] >= 15) & (hls_img[:,:,0] <= 35))
                      & ((hls_img[:,:,1] >= 30) & (hls_img[:,:,1] <= 204))
                      & ((hls_img[:,:,2] >= 115) & (hls_img[:,:,2] <= 255))                
                    ] = 1
        
        # Compute a binary thresholded image where white is isolated from HLS components
        img_hls_white_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_bin[((hls_img[:,:,0] >= 0) & (hls_img[:,:,0] <= 255))
                      & ((hls_img[:,:,1] >= 200) & (hls_img[:,:,1] <= 255))
                      & ((hls_img[:,:,2] >= 0) & (hls_img[:,:,2] <= 255))                
                    ] = 1
        
        # Now combine both
        img_hls_white_yellow_bin = np.zeros_like(hls_img[:,:,0])
        img_hls_white_yellow_bin[(img_hls_yellow_bin == 1) | (img_hls_white_bin == 1)] = 1
    
        return img_hls_white_yellow_bin
    
    def mag_sobel(self,gray_img, kernel_size=3, thres=(0, 255)):
        #Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
        #and applies a threshold value to only set pixels within the specified range
        sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        sxy = np.sqrt(np.square(sx) + np.square(sy))
        scaled_sxy = np.uint8(255 * sxy / np.max(sxy))
        
        sxy_binary = np.zeros_like(scaled_sxy)
        sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1
        
        return sxy_binary
    
    
    
    def combined_sobels(self,sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):

        sxy_direction_binary = self.dir_sobel(gray_img, kernel_size=kernel_size, thres=angle_thres)

        combined = np.zeros_like(sxy_direction_binary)
        # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
        combined[(sx_binary == 1) | ((sy_binary == 1) & (sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1
        
        return combined
    
    def ToLab(self, img):#Returns the same image in LAB format
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    def GreenFilter(self):
        
        undist_test_img_gray = self.ToLab(self.current_RoadRGB)[:,:,0]
        
        undistorted_yellow_white_hls_img_bin = self.compute_hls_white_yellow_binary(self.current_RoadRGB)
        
        sobx_best = self.abs_sobel(undist_test_img_gray, kernel_size=15, thres=(20, 120))
        
        # Saving our best sobel y result
        soby_best = self.abs_sobel(undist_test_img_gray, x_dir=False, kernel_size=15, thres=(20, 120))
        sobxy_best = self.mag_sobel(undist_test_img_gray, kernel_size=15, thres=(80, 200))
        
        sobel_combined_best = self.combined_sobels(sobx_best, soby_best, sobxy_best, undist_test_img_gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))                                                                            

        color_binary = np.dstack((np.zeros_like(sobel_combined_best), sobel_combined_best, undistorted_yellow_white_hls_img_bin)) * 255
        color_binary = color_binary.astype(np.uint8)
        
        combined_binary = np.zeros_like(undistorted_yellow_white_hls_img_bin)
        combined_binary[(sobel_combined_best == 1) | (undistorted_yellow_white_hls_img_bin == 1)] = 1
        
        return color_binary,combined_binary


    def image_Quality(self, img):
        
        self.current_Image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        self.yuv = cv2.cvtColor(self.current_Image, cv2.COLOR_RGB2YUV)
        
        # Computes stats for the sky image
        
        self.current_SkyL = self.yuv[0:self.mid, :, 0]
        self.current_SkyRGB[0:self.mid, : ] = self.current_Image[0:self.mid, : ]
        self.skylrgb[0] = np.average(self.current_SkyL[0:self.mid, :])
        self.skylrgb[1] = np.average(self.current_SkyRGB[0:self.mid, :, 0])
        self.skylrgb[2] = np.average(self.current_SkyRGB[0:self.mid, :, 1])
        self.skylrgb[3] = np.average(self.current_SkyRGB[0:self.mid, :, 2])
        
        # Computes stats for the road image
        self.current_RoadL = self.yuv[self.mid:self.y, :, 0]
        self.current_RoadRGB[:,:] = self.current_Image[self.mid:self.y, :]
        self.roadlrgb[0] = np.average(self.current_RoadL[0:self.mid, :])
        self.roadlrgb[1] = np.average(self.current_RoadRGB[0:self.mid, :, 0])
        self.roadlrgb[2] = np.average(self.current_RoadRGB[0:self.mid, :, 1])
        self.roadlrgb[3] = np.average(self.current_RoadRGB[0:self.mid, :, 2])
        # cv2.imshow("testing ", self.current_RoadRGB)
        # cv2.waitKey(0)
        # Sky image condition
        if self.skylrgb[0] > 160:
            self.skyImageQ = 'The Sky is : overexposed'
        elif self.skylrgb[0] < 50:
            self.skyImageQ = 'The Sky is : underexposed'
        elif self.skylrgb[0] > 143:
            self.skyImageQ = 'The Sky is : normal bright'
        elif self.skylrgb[0] < 113:
            self.skyImageQ = 'The Sky is : normal dark'
        else:
            self.skyImageQ = 'The Sky is : normal'

        # Sky detected weather or lighting conditions
        if self.skylrgb[0] > 128:
            if self.skylrgb[3] > self.skylrgb[0]:
                if self.skylrgb[1] > 120 and self.skylrgb[2] > 120:
                    if (self.skylrgb[2] - self.skylrgb[1]) > 20.0:
                        self.skyText = 'Sky Condition: tree shaded'
                    else:
                        self.skyText = 'Sky Condition: cloudy'
                else:
                    self.skyText = 'Sky Condition: clear'
            else:
                self.skyText = 'Sky Condition: UNKNOWN SKYL>128'
        else:
            if self.skylrgb[2] > self.skylrgb[3]:
                self.skyText = 'Sky Condition: surrounded by trees'
                self.visibility = -80
            elif self.skylrgb[3] > self.skylrgb[0]:
                if (self.skylrgb[2] - self.skylrgb[1]) > 10.0:
                    self.skyText = 'Sky Condition: tree shaded'
                else:
                    self.skyText = \
                        'Sky Condition: very cloudy or under overpass'
            else:
                self.skyText = 'Sky Condition: UNKNOWN!'

        self.roadbalance = self.roadlrgb[0] / 10.0

        # Detemines the conditions of the road 
        if self.roadlrgb[0] > 160:
            self.roadImageQ = 'Road Image: overexposed'
        elif self.roadlrgb[0] < 50:
            self.roadImageQ = 'Road Image: underexposed'
        elif self.roadlrgb[0] > 143:
            self.roadImageQ = 'Road Image: normal bright'
        elif self.roadlrgb[0] < 113:
            self.roadImageQ = 'Road Image: normal dark'
        else:
            self.roadImageQ = 'Road Image: normal'

    def Background_sub(self, imgs):
        img = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img ,(self.x, self.y), None)
        res = cv2.createBackgroundSubtractorMOG2()
        return cv2.bitwise_or(res.apply(img) , img)
    
    ## Define a function to masks out yellow lane lines
    def image_only_yellow_white(self, img):
        # setup inRange to mask off everything except white and yellow
        lower_yellow_white = np.array([140, 140, 64])
        upper_yellow_white = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_yellow_white, upper_yellow_white)
        self.all_yellow = cv2.bitwise_and(img, img, mask=mask)

    

    # Define a function that applies Sobel x or y, then takes an absolute value and applies a threshold.
    def abs_sobel_thresh(self, img, orient='x', thresh=(40, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            abs_sobel = np.absolute(sobelx)
        if orient == 'y':
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            abs_sobel = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))# Rescale back to 8 bit integer
        # Create a copy and apply the threshold
        ret, binary_output = cv2.threshold(scaled_sobel, thresh[0], thresh[1], cv2.THRESH_BINARY)
        return binary_output

    # Define a function that applies Sobel x and y,then computes the magnitude of the gradient and applies a threshold
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(40, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        ret, mag_binary = cv2.threshold(gradmag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)
        return mag_binary

    # Define a function that applies Sobel x and y, then computes the direction of the gradient and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        with np.errstate(divide='ignore', invalid='ignore'):
            dirout = np.absolute(np.arctan(sobely/sobelx))
            # 5) Create a binary mask where direction thresholds are met
            dir_binary =  np.zeros_like(dirout).astype(np.float32)
            dir_binary[(dirout > thresh[0]) & (dirout < thresh[1])] = 1
            # 6) Return this mask as your binary_output image
        # update nan to number
        np.nan_to_num(dir_binary)
        # make it fit
        dir_binary[(dir_binary>0)|(dir_binary<0)] = 128
        return dir_binary.astype(np.uint8)

    # Define a function that thresholds the S-channel of HLS
    def hls_s(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:,:,2]
        retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return s_binary

    # Define a function that thresholds the H-channel of HLS
    def hls_h(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = hls[:,:,0]
        retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return h_binary
   
    
    def thresholding(self,img):
        
        img_to_gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)#makes image gray
        kernel = np.ones((5,5))
        img_g_blur = cv2.GaussianBlur(img_to_gray, (5, 5),0)
        img_current_RoadEdge = cv2.Canny(img_g_blur, 50, 150 )
        white_yellow ,white,yellow = self.Color_filter(img)
        img_current_RoadEdge_yellow = cv2.Canny(white_yellow, 50, 150 )
        #dial and erode are irrelevant but improves the results a little
        img_dial = cv2.dilate(img_current_RoadEdge,kernel,iterations=1)
        img_erode = cv2.erode(img_dial,kernel,iterations=1)
        to_procees = cv2.Canny(white_yellow, 50, 100)
        combined_img = cv2.bitwise_or(to_procees, img_erode)
    
        return combined_img,img_current_RoadEdge_yellow,white_yellow
    
    def Color_filter(self, img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
      
        # Range for lower red
        red_lower = np.array([0,120,70])
        red_upper = np.array([10,255,255])
        mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
        # Range for upper range
        red_lower = np.array([170,120,70])
        red_upper = np.array([180,255,255])
        mask_red2 = cv2.inRange(hsv, red_lower, red_upper)
        mask_red = mask_red1 + mask_red2
       
        
        # Range for upper range
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        White_lower = np.array([0, 0, 200])
        White_upper = np.array([255, 255, 255])
        
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        mask_white = cv2.inRange(hsv,White_lower,White_upper)
        
        img_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
        img_white = cv2.bitwise_and(img, img, mask=mask_white)
        red_output = cv2.bitwise_and(img, img, mask=mask_red)
        
        red_ratio=(cv2.countNonZero(mask_red))/(img.size/3)
        yellow_ratio =(cv2.countNonZero(mask_yellow))/(img.size/3)
        white_ratio = (cv2.countNonZero(mask_white))/(img.size/3)
        
        Yellowinimage = np.round(yellow_ratio*100, 2)
        whiteinimage = np.round(white_ratio*100, 2)
        Redinimage = np.round(red_ratio*100, 2)
        
        white_yellow = cv2.bitwise_or(img_white, img_yellow)
        
        return white_yellow,img_white,img_yellow
    
    
    ################################## STILL TO TEST ############################
     # A function to cut the image in half horizontally
    def Half_Img(self, image, half=0):
        if half == 0:
            if len(image.shape) < 3:
                newimage = np.copy(image[self.mid:self.y, :])
            else:
                newimage = np.copy(image[self.mid:self.y, :, :])
        else:
            if len(image.shape) < 3:
                newimage = np.copy(image[0:self.mid, :])
            else:
                newimage = np.copy(image[0:self.mid, :, :])
        return newimage
    
    def miximg(self, img1, img2, α=0.8, β=1., λ=0.):
        return cv2.addWeighted(img1.astype(np.uint8),
                               α, img2.astype(np.uint8), β, λ)
    
    
    def horizonDetect(self, debug=False, thresh=50):
        if not self.horizonFound:
            img = np.copy(self.current_RoadRGB).astype(np.uint8)
            magch = self.mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 150))
            horizonLine = 50
            while not self.horizonFound and horizonLine < int(self.y / 2):
                magchlinesum = np.sum(magch[horizonLine:(horizonLine + 1), :]).astype(np.float32)
                
                if magchlinesum > (self.x * thresh):
                    self.horizonFound = True
                    self.roadhorizon = horizonLine + int(self.y / 2)
                else:
                    horizonLine += 1

    def drawHorizon(self, image):
        horizonLine = self.roadhorizon
        image[horizonLine:(horizonLine + 1), :, 0] = 255
        image[horizonLine:(horizonLine + 1), :, 1] = 255
        image[horizonLine:(horizonLine + 1), :, 2] = 0

    def balanceEx(self):
        # separate each of the RGB color channels
        r = self.current_RoadRGB[:, :, 0]
        g = self.current_RoadRGB[:, :, 1]
        b = self.current_RoadRGB[:, :, 2]
        # Get the Y channel (Luma) from the YUV color space
        # and make two copies
        yo = np.copy(self.current_RoadL[:, :]).astype(np.float32)
        yc = np.copy(self.current_RoadL[:, :]).astype(np.float32)
        # use the balance factor calculated previously to calculate the
        # corrected Y
        yc = (yc / self.roadbalance) * 8.0
        # make a copy and threshold it to maximum value 255.
        lymask = np.copy(yc)
        lymask[(lymask > 255.0)] = 255.0
        # create another mask that attempts to masks yellow road markings.
        uymask = np.copy(yc) * 0
        # subtract the thresholded mask from the corrected Y.
        # Now we just have peaks.
        yc -= lymask
        # If we are dealing with an over exposed image
        # cap its corrected Y to 242.
        if self.roadlrgb[0] > 160:
            yc[(b > 254) & (g > 254) & (r > 254)] = 242.0
        # If we are dealing with a darker image
        # try to pickup faint blue and cap them to 242.
        elif self.roadlrgb[0] < 128:
            yc[(b > self.roadlrgb[3]) & (yo > 160 + (self.roadbalance * 20))] = 242.0
        else:
            yc[(b > self.roadlrgb[3]) & (yo > 210 + (self.roadbalance * 10))] = 242.0
        # attempt to mask yellow lane lines
        uymask[(b < self.roadlrgb[0]) & (r > self.roadlrgb[0]) & (g > self.roadlrgb[0])] = 242.0
        # combined the corrected road luma and the masked yellow
        yc = self.miximg(yc, uymask, 1.0, 1.0)
        # mix it back to the original luma.
        yc = self.miximg(yc, yo, 1.0, 0.8)
        # resize the image in an attempt to get the lane lines to the bottom.
        yc[int((self.y / 72) * 70):self.y, :] = 0
        self.yuv[self.mid:self.y, :, 0] = yc.astype(np.uint8)
        self.yuv[(self.y - 40):self.y, :, 0] = \
            yo[(self.mid - 40):self.mid, :].astype(np.uint8)
        # convert back to RGB.
        self.current_RoadRGB = cv2.cvtColor(self.yuv[self.mid:self.y, :, :], cv2.COLOR_YUV2RGB)
        
    # def yellow_colorDetection(self , image):
        
    #     hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #     '''Red'''
    #     # Range for lower red
    #     red_lower = np.array([0,120,70])
    #     red_upper = np.array([10,255,255])
    #     mask_red1 = cv2.inRange(hsv, red_lower, red_upper)
    #     # Range for upper range
    #     red_lower = np.array([170,120,70])
    #     red_upper = np.array([180,255,255])
    #     mask_red2 = cv2.inRange(hsv, red_lower, red_upper)
    #     mask_red = mask_red1 + mask_red2
    #     red_output = cv2.bitwise_and(image, image, mask=mask_red)
    #     red_ratio=(cv2.countNonZero(mask_red))/(image.size/3)
    #     print("Red in image", np.round(red_ratio*100, 2))
    #     '''yellow'''
    #     # Range for upper range
    #     yellow_lower = np.array([20, 100, 100])
    #     yellow_upper = np.array([30, 255, 255])
    #     mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    #     yellow_output = cv2.bitwise_and(image, image, mask=mask_yellow)
    #     yellow_ratio =(cv2.countNonZero(mask_yellow))/(image.size/3)
    #     print("Yellow in image", np.round(yellow_ratio*100, 2))
        
    #     return yellow_output
    
    # A function to cut the image in half horizontally
    def Half_Img(self, image, half=0):
        if half == 0:
            if len(image.shape) < 3:
                newimage = np.copy(image[self.mid:self.y, :])
            else:
                newimage = np.copy(image[self.mid:self.y, :, :])
        else:
            if len(image.shape) < 3:
                newimage = np.copy(image[0:self.mid, :])
            else:
                newimage = np.copy(image[0:self.mid, :, :])
        return newimage
            
            
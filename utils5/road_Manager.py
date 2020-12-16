# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 08:32:41 2020

@author: Dr~Newt
"""

import numpy as np
import cv2
from utils5.image_Filter import ImageFilters
from utils5.projection_Manager import Projection_Mgr
from utils5.lane import Lane
from utils5.line import Line
import sys 

class RoadManager():
    # Initialize roadManager

    def __init__(self, camCal):
        self.current_Frame = None
        self.mtx, self.dist, self.img_size = camCal.get()
        self.x, self.y = self.img_size
        self.mid = int(self.y / 2)
        self.left_a, self.left_b, self.left_c = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []
        self.im_filt = ImageFilters(camCal)
        self.Final_proc_img = None
    #*******************************************************************************************#
    #                          TRACK BACK INTERFACE DESIGN                                      #
    #*******************************************************************************************#    
    def nothing(x):# call back function for tackbar : does nothing
        pass
    
    def initializeTrackbars(self, init_Tracbar_Vals):#track bar interface
        
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 360, 240)
        cv2.createTrackbar("Width Top", "Trackbars", init_Tracbar_Vals[0],50, self.nothing)
        cv2.createTrackbar("Height Top", "Trackbars", init_Tracbar_Vals[1], 100, self.nothing)
        cv2.createTrackbar("Width Bottom", "Trackbars", init_Tracbar_Vals[2], 50, self.nothing)
        cv2.createTrackbar("Height Bottom", "Trackbars", init_Tracbar_Vals[3], 100, self.nothing)
    
    def valTrackbars(self):#Tracks the track bar position
        
        width_Top = cv2.getTrackbarPos("Width Top", "Trackbars")
        height_Top = cv2.getTrackbarPos("Height Top", "Trackbars")
        width_Bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
        height_Bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    
        src_points = np.float32([(width_Top/100,height_Top/100), (1-(width_Top/100), height_Top/100),
                          (width_Bottom/100, height_Bottom/100), (1-(width_Bottom/100), height_Bottom/100)])
        return src_points
    
    def drawPoints(self, img,src_points):#Draws points on image to define prespective warp
        
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src_points = src_points * img_size
        for x in range( 0,4):
            cv2.circle(img,(int(src_points[x][0]),int(src_points[x][1])),15,(0,0,255),cv2.FILLED)
        return img
    
    def textDisplay(self, img,curvature, frame):
        # num_Vehicle = Projection_Mgr()
        font_Type = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img, "Curvature : " +str(curvature), ((img.shape[1]//2)+100, 35), font_Type, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Frame Number : " + str(frame), ((img.shape[1]//2)+100, 15), font_Type, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Vehicle Count : " + str("num_Vehicle.Num_Vehicle"), ((img.shape[1]//2)+100, 55), font_Type, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Pedestrian : " + str(0), ((img.shape[1]//2)+100, 75), font_Type, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 20), (200, 0),(255,255,255), 30 )
        direction_Text = "No lanes Detected"
        
        if curvature > 10: 
            direction_Text = "Turn Right"
        elif curvature < -10:
            direction_Text = "Turn Left"
        elif curvature >= 4 and curvature < 10 or curvature >= -4 and curvature < -10 :
            direction_Text = "Go Straight"
        elif curvature == -1000000:
            cv2.putText(img, "WARNING !!!", ((img.shape[1]//2)-35, 205 ), font_Type, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            direction_Text = "No lanes Detected"
        
        cv2.putText(img,direction_Text, ((img.shape[1]//2)-35,205 ), font_Type, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    #Returns the birds eye view of the image(frame)                
    def perspective_warp(self, img, dst_size=(1280, 720),
                         src_points = np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                         dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
        
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src_points = src_points* img_size
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped
    
    #returns an image 
    def inv_perspective_warp(self, img, dst_size=(1280,720),
                         src_points=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                         dest_points=np.float32([(0.43,0.65),(0.48,0.65),(0.1,1),(1,1)])):
        
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src_points = src_points* img_size
        dest_points = dest_points * np.float32(dst_size)
        presp_mtx = cv2.getPerspectiveTransform(src_points, dest_points)#calculate the perspective transform matrix
        inv_warped = cv2.warpPerspective(img, presp_mtx, dst_size)
        return inv_warped
    
    
    def get_hist(self, img):
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        return hist
    
    def sliding_window(self, img, nwindows=15, margin=60, minpix=1, draw_windows=True):
        
        #for storing a,b and c after computation
        current_left_fit = np.empty(3)
        current_right_fit = np.empty(3)
        # out_img = img
        out_img = np.dstack((img,img,img))*255
        histogram = self.get_hist(img)
        # find peaks of left and right halves from hist
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # for Identifying the x and y positions of all non zero pixels in the binary image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows or rectangles on the birds eye view binary image
            if draw_windows == True:
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),(0, 0, 255), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),(255, 0, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        if leftx.size and rightx.size:

            # Fit a second order polynomial to to left and right lane pixel values
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            #captures the calculated values of a,b and c for right and left
            self.left_a.append(left_fit[0])
            self.left_b.append(left_fit[1])
            self.left_c.append(left_fit[2])
            self.right_a.append(right_fit[0])
            self.right_b.append(right_fit[1])
            self.right_c.append(right_fit[2])
    
            current_left_fit[0] = np.mean(self.left_a[-10:])
            current_left_fit[1] = np.mean(self.left_b[-10:])
            current_left_fit[2] = np.mean(self.left_c[-10:])
    
            current_right_fit[0] = np.mean(self.right_a[-10:])
            current_right_fit[1] = np.mean(self.right_b[-10:])
            current_right_fit[2] = np.mean(self.right_c[-10:])
    
            # Generate x and y values for plotting
            ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
            
            left_fitx_eq = current_left_fit[0] * ploty ** 2 + current_left_fit[1] * ploty + current_left_fit[2]
            right_fitx_eq = current_right_fit[0] * ploty ** 2 + current_right_fit[1] * ploty + current_right_fit[2]
    
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
            return out_img, (left_fitx_eq, right_fitx_eq), (current_left_fit, current_right_fit), ploty
        else:
            return img,(0,0),(0,0),0
     
    def get_curve(self, img, leftx, rightx):
        
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 1 / img.shape[0]  # meters per pixel in y dimension
        xm_per_pix = 0.1 / img.shape[0]  # meters per pixel in x dimension
    
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
    
        car_pos = img.shape[1] / 2
        l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
    
        return (l_fit_x_int, r_fit_x_int, center)
    
    
    
    def draw_lanes(self, img, left_fit, right_fit,frameWidth,frameHeight,src):
        
        ploty = np.linspace(0, img.shape[0]-1 , img.shape[0])
        color_img = np.zeros_like(img)
    
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right + 15))
    
        cv2.fillPoly(color_img, np.int_(points), (0, 150, 0))
        inv_perspective = self.inv_perspective_warp(color_img,(frameWidth,frameHeight),dest_points = src)
        inv_perspective = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)
        return inv_perspective
    
    def Hough_Lines_Draw(self, img):
        
        lines = cv2.HoughLinesP(img ,2, np.pi/180, 100 , np.array([]), minLineLength=40, maxLineGap = 5)
        mask = np.zeros((self.y, self.x, 3), dtype=np.float32)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(mask, (x1,y1), (x2,y2), (255,255,255),10)
            return mask
        return None
    

    def drawLines(self, img,lane_curve):#Draws vertical lines
        myWidth = img.shape[1]
        myHeight = img.shape[0]
        for x in range(-30, 30):
            w = myWidth // 20
            cv2.line(img, (w * x + int(lane_curve // 100), myHeight - 30),
                     (w * x + int(lane_curve // 100), myHeight), (0, 0, 255), 2)
        cv2.line(img, (int(lane_curve // 100) + myWidth // 2, myHeight - 30),
                 (int(lane_curve // 100) + myWidth // 2, myHeight), (0, 255, 0), 3)
        cv2.line(img, (myWidth // 2, myHeight - 50), (myWidth // 2, myHeight), (255, 0, 0), 2)
    
        return img
    
    #Combines images and shows the main pipeline
    def stackImages(self, scale,imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range ( 0, rows):
                for y in range(0, cols):
                    
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor= np.hstack(imgArray)
            ver = hor
        return ver
    

    

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 01:24:00 2020

@author: Dr~Newt
"""

import cv2
import numpy as np
from utils5.image_Filter import ImageFilters
from utils5.Tasks_tested import MYTASKS
from keras.models import model_from_json
from keras.models import load_model as f
from astropy.time import Time
import time

class Projection_Mgr():

    def __init__(self, camCal, model):
        
        self.mtx, self.dist, self.img_size = camCal.get()
        self.x, self.y = self.img_size
        self.mid = int(self.y / 2)
        self.Num_Vehicle = 0
        self.current_RoadRGB = np.zeros((self.mid, self.x, 3), dtype=np.float32)
        self.current_Sign = np.zeros((self.mid, self.x, 3), dtype=np.float32)
        self.frameWidth= 640       # CAMERA RESOLUTION
        self.frameHeight = 480
        self.loaded_model = model
        self.brightness = 180
        self.threshold = 0.75         # PROBABLITY THRESHOLD
        self.font = cv2.FONT_HERSHEY_SIMPLEX   
        self.mySpeed = 0.0
        self.task = MYTASKS()
        self.time_now = time.time()
        self.time_last = time.localtime()
        
        
    def timeInSec(self,t):
        
        mytime = Time(t,format='gps')
        mytime = Time(mytime, format='iso')
        T = str(mytime).split(":")
        tSec = float(T[-1])
        print("seconds: ",tSec)
        return tSec
        
    def SensorOutput(self,sensor_raw_data):
        
        description = sensor_raw_data['desc']
        units = sensor_raw_data['unit']
        accel_data = sensor_raw_data['data']
        time = accel_data[0][0]
        
        zAverageSpeed = 0.0
        for i in range(len(accel_data)):
            x = accel_data[-1][1][0]
            y = accel_data[-1][1][1]
            z = accel_data[i][1][-1]
            X,Y,Z = self.calibXYZ(x,y,z)
            try:
                gpsTime = accel_data[i+1][0] - accel_data[i][0]
                # print(accel_data[i+1][0], " - ",accel_data[i][0], " = ", gpsTime)
                zSpeedCalc = Z * gpsTime/60
                # zDistance = Z * (gpsTime)^2
                zAverageSpeed += zSpeedCalc
        
            except:
                pass
            
        self.mySpeed = zAverageSpeed#Current Speed
        
        if zAverageSpeed> 7:
            print("[ Moving Forward ] Current speed : ", zAverageSpeed)
        
        elif zAverageSpeed<-7:
            print("[ Moving Backwards ] Current speed : ", zAverageSpeed)
        else:
            print("[ Stationery ] Current speed : ", zAverageSpeed)    
       
    def calibXYZ(self,x,y,z):
    
        yActual = y + 0.011901006
        xActual = x + 0.025333405
        zActual = z + 0.003943324
        
        return xActual,yActual,zActual
        
    def Detect_Objs(self, img,img1, detect_objs=0):
        print("Obj active")
        if detect_objs == 0:
            self.ojb_state = "NOT ENABLED"
            return img
        
        thres = 0.45 # Threshold to detect object
        nms_threshold = 0.2
        classNames= []
        classFile = 'coco.names'
        
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')
    
        configPath = 'Models_and_Data/Obj_My_model.pbtxt'
        weightsPath = 'Models_and_Data/Obj_My_weights.pb'
        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        
        indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)
        
            
        for i in indices:
            i = i[0]
            box = bbox[i]
            x,y,w,h = box[0],box[1],box[2],box[3]
            cv2.rectangle(img1, (x,y),(x+w,h+y), color=(0, 0, 255), thickness=2)
            cv2.putText(img1,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,255),2)
            
            
            if classNames[classIds[i][0]-1].upper() == "TRAFFIC LIGHT":
                print("[ Traffic lights detected.].........................................")
                self.TrafficLights(img, detect_traffic_light=1)
                
            if classNames[classIds[i][0]-1].upper() == "STOP SIGN":
                #self.current_Sign[x:x+w,y:y+h] = img[x:x+w,y:y+h]
                print("[ Stop sign Detected... ].........................................")
                self.StopSigns(img, detect_Stop_signs=1)
                
            if classNames[classIds[i][0]-1].upper() == "CLOCK":
                print("[ Speed Sign detected.].........................................")
                self.SpeedLimit(img, detect_Stop_signs=1)
     
    def grayscale(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    
    def equalize(self,img):
        img =cv2.equalizeHist(img)
        return img
    
    def preprocessing(self,img):
        img = self.grayscale(img)
        img = self.equalize(img)
        img = img/255
        return img  
        
    def getCalssName(self,classNo):
        MyClasses = {0:'Speed Limit 20 km/h', 1:'Speed Limit 30 km/h',
        2:'Speed Limit 50 km/h',3:'Speed Limit 60 km/h', 4:'Speed Limit 70 km/h',
        5:'Speed Limit 80 km/h',6:'End of Speed Limit 80 km/h',7:'Speed Limit 100 km/h',
        8:'Speed Limit 120 km/h',9:'No passing',10:'No passing for vechiles over 3.5 metric tons',
        11:'Right-of-way at the next intersection',12:'Priority road',13:'Yield',14:'Stop',
        15:'No vechiles',16:'Vechiles over 3.5 metric tons prohibited',17:'No entry',18:'General caution',
        19:'Dangerous curve to the left',20:'Dangerous curve to the right',21:'Double curve',22:'Bumpy road',
        23:'Slippery road',24:'Road narrows on the right',25:'Road work',
        26:'Traffic signals',27:'Pedestrians',28:'Children crossing',29:'Bicycles crossing',
        30:'Beware of ice/snow',31:'Wild animals crossing',32:'End of all speed and passing limits',
        33:'Turn right ahead',34:'Turn left ahead',35:'Ahead only',36:'Go straight or right',
        37:'Go straight or left',38:'Keep right',39:'Keep left',40:'Roundabout mandatory',
        41:'End of no passing',42:'End of no passing by vechiles over 3.5 metric tons'}
        
        return MyClasses[classNo]
        
    def Traffic_Sign_Rec(self,imgs, active=0): #uses a trained model to detect and classify speed and stop signs
        
    
        if active == 0:
            return imgs
        # PROCESS IMAGE
        img = np.asarray(imgs)
        img = cv2.resize(img, (32, 32))
        img = self.preprocessing(img)
       
        img = img.reshape(1, 32, 32, 1)
        cv2.putText(imgs, "CLASS : " , (20, 35), self.font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgs, "ACCURACY : ", (20, 75), self.font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # PREDICT IMAGE
        predictions = self.loaded_model.predict(img)
        classIndex = self.loaded_model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        if probabilityValue > self.threshold:
            traffic_sign_class = (self.getCalssName(int(classIndex)))
            
            cv2.putText(imgs,str(classIndex)+" "+str(traffic_sign_class), (120, 35), self.font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgs, str(round(probabilityValue*100,2) )+"%", (180, 75), self.font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            #print("The following class : " ,str(classIndex)+" "+str(traffic_sign_class))
            if traffic_sign_class == 'STOP':#Activates test sign task
                print("processing stop")
                self.task.StopSign(self.mySpeed)#run as a thread

            return imgs
        return imgs
    
    def TrafficLights(self,img,detect_traffic_light=0):#detects traffic lights signals
        
        if detect_traffic_light == 0:
            print("traffic light detection not enabled")
            return img
        
        traffic_sign_xml = "xml_classifier/traffic_light.xml"
 
        # except:
        #     print("traffic light detection and identification files not Found!!")
        
        # img_process = img[x:y,w:h]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        traffic_light_cascade = cv2.CascadeClassifier(traffic_sign_xml)

        # minimum value to proceed traffic light state validation
        threshold = 150

        # detection
        cascade_obj = traffic_light_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))

        # draw a rectangle around the objects
        for (x_pos, y_pos, width, height) in cascade_obj:
            cv2.rectangle(img, (x_pos + 5, y_pos + 5), (x_pos + width - 5, y_pos + height - 5), (255, 255, 255), 2)
            
            if width / height == 1:
                return img
            # traffic lights

            roi = gray[y_pos + 10:y_pos + height - 10, x_pos + 10:x_pos + width - 10]
            mask = cv2.GaussianBlur(roi, (25, 25), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

            # check if light is on
            if maxVal - minVal > threshold:
                cv2.circle(roi, maxLoc, 5, (255, 0, 0), 2)

                # Red light
                if 1.0 / 8 * (height - 30) < maxLoc[1] < 4.0 / 8 * (height - 30):
                    print("<Red Traffic light Detected>")
                    cv2.putText(img, 'Red', (x_pos + 5, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    self.red_light = True
                    self.lightstate = "RED"
                    self.task.TrafficLights(self.mySpeed,self.lightstate)
                    
                # Green light
                elif 5.5 / 8 * (height - 30) < maxLoc[1] < height - 30:
                    print("<Green Traffic light Detected>")
                    cv2.putText(img, 'Green', (x_pos + 5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
                    self.green_light = True
                    self.lightstate = "GREEN"
                    self.task.TrafficLights(self.mySpeed,self.lightstate)
                    
                # yellow light
                elif 4.0/8*(height-30) < maxLoc[1] < 5.5/8*(height-30):
                    print("<Orange Traffic light Detected> ")
                    cv2.putText(img, 'Yellow', (x_pos+5, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    self.yellow_light = True
                    self.lightstate = "YELLOW"
                    self.task.TrafficLights(self.mySpeed,self.lightstate)
                    
                #run as a thread

        
        
    def SpeedLimit(self, img,detect_speed_sign=0):#detects speed limit signs
        
        if detect_speed_sign == 0:
            print("Speed sign light detection not enabled")
            return img
        
        traffic_img = self.Traffic_Sign_Rec(img,active=1)
        return traffic_img
    
        
    def StopSigns(self,img,detect_Stop_signs=0):#detects stopsigns
        
        if detect_Stop_signs == 0:
            print("Stop signs detection not enabled")
            return img
        
        stopSign_img = self.Traffic_Sign_Rec(img, active=1)
        return stopSign_img    
    
    
    #more accurate but slow
    # def Detect_Objs_Yolo(self, imgs, detect_objs = 0):#uses yolov3 for object detection
        
    #     if detect_objs == 0:
    #         self.ojb_state = "NOT ENABLED"
    #         return None
        
    #     #self.current_RoadRGB[self.mid : self.y, :] = img1[self.y :self.mid, :]
    #     #cnn network setup
    #     net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    #     classes = []
    #     #extracts labels and populates them in classes
    #     with open("coco.names", "r") as f:
    #         classes = [line.strip() for line in f.readlines()]
    #     #get layers from cnn     
    #     layer_names = net.getLayerNames()
    #     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
    #     #generates random colors to be applied to object detected
    #     colors = np.random.uniform(0, 255, size=(len(classes), 3))
    #     height, width, channels = imgs.shape
    
    #     # Detecting objects
    #     blob = cv2.dnn.blobFromImage(imgs, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
    #     net.setInput(blob)
    #     outs = net.forward(output_layers)
        
    #     # Showing informations on the screen
    #     class_ids = []
    #     for car in class_ids:
    #         self.Num_Vehicle += 1
    #     confidences = []
    #     boxes = []
    #     for out in outs:
    #         for detection in out:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
    #             if confidence > 0.5:
    #                 # Object detected
    #                 center_x = int(detection[0] * width)
    #                 center_y = int(detection[1] * height)
    #                 w = int(detection[2] * width)
    #                 h = int(detection[3] * height)
        
    #                 # Rectangle coordinates
    #                 x = int(center_x - w / 2)
    #                 y = int(center_y - h / 2)
        
    #                 boxes.append([x, y, w, h])
    #                 confidences.append(float(confidence))
    #                 class_ids.append(class_id)
        
    #     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #     print(indexes)
    #     font = cv2.FONT_HERSHEY_COMPLEX
    #     for i in range(len(boxes)):
    #         if i in indexes:
    #             x, y, w, h = boxes[i]
    #             label = str(classes[class_ids[i]])
    #             color = colors[class_ids[i]]
    #             cv2.rectangle(imgs, (x, y), (x + w, y + h), color, 2)
    #             cv2.rectangle(imgs, (x + int(x/17), y - int(y/30)),(x, y), color, 8)
    #             cv2.putText(imgs, label, (x, y - int(y/45)), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

class PLOT_GRAPHS():

    def __init__(self):
        pass

    def plotDistance(self,distanceData):
        pass

    def plotAcceleration(self, accelData):
        pass

    def plotSpeed(self, speedData):
        pass

    
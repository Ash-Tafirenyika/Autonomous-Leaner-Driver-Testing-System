# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:38:30 2020

@author: Dr~Newt
"""
"""
    CAR LANE DETECTION SYSTEM USING OPENCV
    BY ASHTON TAFIRENYIKA 
    DONE IN FULFILMENT OF THE BSC HONS DEGREE PART 3 POJECT
"""

import argparse
import cv2
import numpy as np
import requests
from utils5.camera_calib import Camera_Calib
from utils5.image_Filter import ImageFilters
from utils5.road_Manager import RoadManager
from utils5.projection_Manager import Projection_Mgr
from keras.models import model_from_json
from keras.models import load_model as f
import json
from PySide2.QtCore import Signal, Qt, Slot, QRunnable, QObject, QThread
from PySide2.QtGui import QImage
import traceback, sys


# class WorkerSignals(QObject):
#     '''
#     Defines the signals available from a running worker thread.
#     Supported signals are:
#     finished
#         No data
#     error
#         `tuple` (exctype, value, traceback.format_exc() )
#     result
#         `object` data returned from processing, anything
#     progress
#         `int` indicating % progress
#     '''
#     finished = Signal()
#     error = Signal(tuple)
#     result = Signal(object)
#     progress = Signal(int)


class Thread(QThread):

    changePixmap = Signal(QImage)

    #def __init__(self, *args):  # , fn, , **kwargs
        # super(MyWorker, self).__init__()
        # # Store constructor arguments (re-used for processing)
        # self.fn = fn
        # self.args = args
        # self.kwargs = kwargs
        # self.signals = WorkerSignals()
        # # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress
    testInprogress = True
    calib_dir = "./camera_cal/calibrationdata.p"  # load or perform camera calibrations
    camCal = Camera_Calib('camera_cal', calib_dir)
    img_Filters = ImageFilters(camCal)
    road_mgr = RoadManager(camCal)
    frame_Width = 640
    frame_Height = 480
    intialTracbarVals = [43, 64, 0, 100]  # wT,hT,wB,hB
    url_video = "http://192.168.1.117:8080/shot.jpg"  # "http://192.168.43.1:8080/video"
    url_sensor = "http://192.168.1.117:8080/sensors.json"
    # Variable intialisation
    noOfArrayValues = 10
    arrayCounter = 0
    arrayCurve = np.zeros([noOfArrayValues])
    json_file = open("Models_and_Data/speed_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Models_and_Data/speed_model_weights.h5")
    loaded_model.save("Models_and_Data/speed_model_weights.hdf5")
    loaded_model = f("Models_and_Data/speed_model_weights.hdf5")
    prog_mgr = Projection_Mgr(camCal, loaded_model)

    def run(self):  # Perfoms processing of frames and display
        print("In Thread>>>>>>>>>>>>>>>>>")
        Frame = 0
        # vid_output = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*'MJPG'), 10,
        #                              (self.frame_Width, self.frame_Height))
        self.road_mgr.initializeTrackbars(self.intialTracbarVals)
        # continuos loop for generating frames detect lanes and display them
        while True:
            print("Thread Started>>>>>>>>>>>>>>>>>")
            # gets the image
            if self.testInprogress == True:

                img_raw_data = self.get_videpo(self.url_video)
                image_array = np.array(bytearray(img_raw_data.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, -1)
                Frame += 1
                img = cv2.resize(img, (self.frame_Width, self.frame_Height), None)

            else:
                img = self.get_videpo(self.url_video)
                img = cv2.resize(img, (self.frame_Width, self.frame_Height), None)
                # self.convert_Image(img)

            imgWarpPoints = img.copy()
            imgFinal = img.copy()
            imgCanny = img.copy()

            # gets the sensor data
            try:
                sensor_raw_data = self.get_sensorData(self.url_sensor)
            except:
                print("[Error Sense] : Unable to obtain Sensors data system will restarting.")
                continue
            # pocesses images
            Image_Quality = self.img_Filters.image_Quality(img)
            imgUndis = self.img_Filters.current_Image
            skytext = self.img_Filters.skyText
            skyimgq = self.img_Filters.skyImageQ
            roadimgq = self.img_Filters.roadImageQ

            self.img_Filters.balanceEx()
            self.img_Filters.horizonDetect()
            self.img_Filters.drawHorizon(imgFinal)
            imgThres, imgCanny, imgColor = self.img_Filters.thresholding(imgUndis)

            color_binary, combined_binary = self.img_Filters.GreenFilter()
            # thresh_to_process = img_Filters.Half_Img(imgThres)
            src = self.road_mgr.valTrackbars()
            imgWarp = self.road_mgr.perspective_warp(imgThres, dst_size=(self.frame_Width, self.frame_Height),
                                                     src_points=src)
            img_inv_wap = self.road_mgr.inv_perspective_warp(imgThres)
            imgWarpPoints = self.road_mgr.drawPoints(imgWarpPoints, src)
            imgSliding, curves, lanes, ploty = self.road_mgr.sliding_window(imgWarp, draw_windows=True)

            try:
                curverad = self.road_mgr.get_curve(imgFinal, curves[0], curves[1])
                lane_curve = np.mean([curverad[0], curverad[1]])
                imgFinal = self.road_mgr.draw_lanes(img, curves[0], curves[1], self.frame_Width, self.frame_Height, src)
                # ## Average
                currentCurve = lane_curve // 50
                if int(np.sum(self.arrayCurve)) == 0:
                    averageCurve = currentCurve
                else:
                    averageCurve = np.sum(self.arrayCurve) // self.arrayCurve.shape[0]
                if abs(averageCurve - currentCurve) > 200:
                    self.arrayCurve[arrayCounter] = averageCurve
                else:
                    self.arrayCurve[arrayCounter] = currentCurve
                arrayCounter += 1
                if arrayCounter >= self.noOfArrayValues:
                    arrayCounter = 0
                cv2.putText(imgFinal, "Mean Curve Radi : " + str(int(averageCurve)), (self.frame_Width // 2 - 70, 180),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            except:
                lane_curve = 00
                pass

            self.road_mgr.textDisplay(imgFinal, lane_curve, Frame)
            imgFinal = self.road_mgr.drawLines(imgFinal, lane_curve)
            self.prog_mgr.SensorOutput(sensor_raw_data)
            self.prog_mgr.Detect_Objs(img, imgFinal, detect_objs=0)  # Detects objects, people an cars and draws a box on final img
            # traffic = prog_mgr.Traffic_Sign_Rec(img, active=1)
            # Resize imgs for displaying
            # imgUndis = cv2.resize(imgUndis, None, fx=0.9, fy=0.7)
            # imgWarpPoints = cv2.resize(imgWarpPoints, None, fx=0.9, fy=0.7)
            # imgColor = cv2.resize(imgColor, None, fx=0.9, fy=0.7)
            # imgCanny = cv2.resize(imgCanny, None, fx=0.9, fy=0.7)
            # imgWarp = cv2.resize(imgWarp, None, fx=0.9, fy=0.7)
            # imgSliding = cv2.resize(imgSliding, None, fx=0.9, fy=0.7)
            # img = cv2.resize(img, None, fx=0.9, fy=0.7)
            # img_inv_wap = cv2.resize(img_inv_wap, None, fx=0.9, fy=0.7)

            # vid_output.write(imgFinal)  # Saves video for future reference

            imgThres = cv2.cvtColor(imgThres, cv2.COLOR_GRAY2BGR)
            # labels = [["Distorted Input Image", "Undistorted Input Image", "Warp Points"],
            #           ["Filtered Image", "Canny Image", "Threshold Image"],
            #           ["Warped Image", "Sliding Window", "Inverted Warp"]]
            #
            # imgStacked = self.road_mgr.stackImages(0.7, ([img, imgUndis, imgWarpPoints],
            #                                              [imgColor, color_binary, imgThres],
            #                                              [imgWarp, imgSliding, imgFinal],
            #                                              ))
            # labels for the images.
            # cv2.putText(imgStacked, labels[0][0], ((imgStacked.shape[1] // 2) - 500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[0][1], ((imgStacked.shape[1] // 2) - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[0][2], ((imgStacked.shape[1] // 2) + 350, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             (255, 255, 255), 1, cv2.LINE_AA)
            #
            # cv2.putText(imgStacked, labels[1][0], ((imgStacked.shape[1] // 2) - 500, 260), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[1][1], ((imgStacked.shape[1] // 2) - 100, 260), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[1][2], ((imgStacked.shape[1] // 2) + 350, 260), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            #
            # cv2.putText(imgStacked, labels[2][0], ((imgStacked.shape[1] // 2) - 500, 490), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[2][1], ((imgStacked.shape[1] // 2) - 100, 490), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgStacked, labels[2][2], ((imgStacked.shape[1] // 2) + 350, 490), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
            #
            # cv2.putText(imgFinal, skytext, ((imgFinal.shape[1] // 2) - 300, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgFinal, skyimgq, ((imgFinal.shape[1] // 2) - 300, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (0, 255, 255), 1, cv2.LINE_AA)
            # cv2.putText(imgFinal, roadimgq, ((imgFinal.shape[1] // 2) - 300, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (0, 255, 255), 1, cv2.LINE_AA)

            rgbImage = cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(self.frame_Width, self.frame_Height, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
            # cv2.imshow("PipeLine Process", imgStacked)
            # cv2.imshow("Main video", imgFinal)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # cv2.destroyAllWindows()

    def get_videpo(self, url):  # Perfoms processing of frames and display
        try:
            raw_data = requests.get(url, verify=False)
            self.testInprogress = True
        except:
            raw_data = np.zeros((340, 240, 3), dtype=np.float32)
            print('waiting to connect to camera!!.......')

        print(".........................................NEXT................................................")
        print('connected to camera successfull , [Test in progress %s]!!.......' % self.testInprogress)

        return raw_data

    def get_sensorData(self, url):

        try:
            raw_sensor_data = requests.get(url, verify=False)
            data = (raw_sensor_data.content).decode("utf8").replace("'", '"')
            theJson = json.loads(data)
            final = json.dumps(theJson, indent=4, sort_keys=True)
            # print("start : ", theJson, " end \n\n\n\n")
            accel = theJson['accel']  #: {'desc': ['Ax', 'Ay', 'Az'], 'unit': 'm/s²', 'data'
            gyro = theJson['gyro']  #:{'desc': ['GYRx', 'GYRy', 'GYRz'], 'unit': 'rad/s', 'data'
            lin_accel = theJson['lin_accel']  #: {'desc': ['LAx', 'LAy', 'LAz'], 'unit', 'ms^2','data'
            proximity = theJson['proximity']  # =>{'unit': 'cm', 'data
            rot_vector = theJson[
                'rot_vector']  #: {'desc': ['x*sin(θ/2)', 'y*sin(θ/2)', 'z*sin(θ/2)', 'cos(θ/2)', 'Accuracy'], 'unit'
            gravity = theJson['gravity']  # => desc': ['Gx', 'Gy', 'Gz'], 'unit': 'm/s²', 'data'
            pressure = theJson['pressure']  #: {'unit': 'mbar', 'data'
            light = theJson['light']  #: {'unit': 'lx', 'data'

        except:
            print("[Warning : Unable to obtain senor data.]")
            pass
        return lin_accel

    def convert_Image(self, img):
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImg.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImg.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.keepAspectRatio)
        self.changePixmap.emit(p)

# if __name__ == '__main__':
#     start = Thread()
#     start.main()

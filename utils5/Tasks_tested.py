# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:36:50 2020

@author: Dr~Newt
"""
import time

class MYTASKS():
    
    def __init__(self):
        self.test_progress = True
        self.driverPoints = 1000
    
    def LaneKeeping(self,position, activateTask=False):
        if not activateTask:
            return 0
        
        if position > 9:
            self.laneKeeping = "Failed"
            self.driverPoints -= 100
    
    def StopSign(self,speed):
        
        #Allows a delay of 5s for drivers response and action
        
        if speed <= 7:
            self.stopsignTest = "Failed Moved Too early"
            self.driverPoints -= 100
        else:
            # Count = 5
            # while Count > 0:#Measures a delay of 5 seconds before driver continues 
            #     if speed <=7 :
            #         self.driverPoints -= 50
            #         self.stopsignTest = "Failed Moved Too early"#if drivers moves before the 5s delay has ended
            #         break
            #     time.sleep(1)
            #     Count -= 1
            #     self.stopsignTest = "Passed"
            if speed <= 7:
                self.stopsignTest = "Passed"
        
        print(self.stopsignTest, "####### Driver points #############", self.driverPoints)
    
    def TrafficLights(self,speed,lightstate):
         
         if lightstate == "RED" and speed <= 7:
             print("Driver stopped.......>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
             self.trafficlightObidience = "Pass"
             
         elif lightstate == "RED" and speed > 7:
            self.trafficlightObidience = "Failed"
            self.driverPoints -= 300
            self.disqualified = True
            print("<<<<<<<<<<<<<<<<<<..Failed to obey traffic laws, Test failed..>>>>>>>>>>>>>>>>>")
        
         elif lightstate == "GREEN" and speed == 0:
            
            self.driverPoints -= 100
            print("<<<<<<<<<<<<<<<<<<..Failed waited too long at traffic light..>>>>>>>>>>>>>>>>>")
         
         print(self.trafficlightObidience,"####### Driver points #############", self.driverPoints)
            
             
        
        
            
        
            
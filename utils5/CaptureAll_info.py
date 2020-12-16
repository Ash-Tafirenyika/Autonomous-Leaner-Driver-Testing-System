# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:20:42 2020

@author: Dr~Newt
"""
import mysql.connector as mc
from mysql.connector import Error
from PySide2.QtWidgets import *

class MyInfo():
    
    def __init__(self):
        pass
    
    def NewDriver(self, fname, lname, gender, dob, phoneNum, email, address, pic):
        try:
            connection = mc.connect(host='localhost',
                                    database='loginregister',
                                    user='root',
                                    password='')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO driverinfo (fname, lname, gender, dob, phoneNum, email, address, pic) 
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) """

            recordTuple = (fname, lname, gender, dob, phoneNum, email, address, pic)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            msgBox = QMessageBox()
            msgBox.setText("Record inserted successfully into DRIVERS table")
            msgBox.exec_()

        except mc.connect.Error as error:
            msgBox = QMessageBox()
            msgBox.setText("[Error DB] {} connetion failed.".format(error))
            msgBox.exec_()

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    def Newbooking(self,driverID, bookDateTime, LicenceClass):
        try:
            connection = mc.connect(host="localhost",
                                    user="root",
                                    password="",
                                    database="loginregister")
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO bookings (driverID, bookDateTime, LicenceClass) 
                                    VALUES (%s, %s, %s) """

            recordTuple = (driverID, bookDateTime, LicenceClass)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            msgBox = QMessageBox()
            msgBox.setText("Record inserted successfully into Bookings table")
            msgBox.exec_()


        except mc.Error as error:
            msgBox = QMessageBox()
            msgBox.setText("[Error DB] {} connetion failed.".format(error))
            msgBox.exec_()

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    def DriverTesResult(self,driverID, TaskID, Score, Decision):
        try:
            connection = mc.connect(host='localhost',
                                    database='loginregister',
                                    user='root',
                                    password='')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO results (self,driverID, TaskID, Score, Decision) 
                                    VALUES (%s, %s, %s, %s) """

            recordTuple = (self,driverID, TaskID, Score, Decision)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            msgBox = QMessageBox()
            msgBox.setText("Record inserted successfully into results table")
            msgBox.exec_()

        except mc.Error as error:
            msgBox = QMessageBox()
            msgBox.setText("[Error DB] {} connetion failed.".format(error))
            msgBox.exec_()

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    def NewUser(self, fname, gender, uname, passw, email):
        try:
            connection = mc.connect(host='localhost',
                                    user='root',
                                    password='',
                                    database='loginregister')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO users (fullname, gender, username, password, email) 
                                    VALUES (%s, %s, %s, %s, %s) """

            recordTuple = (fname, gender, uname, passw, email)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            msgBox = QMessageBox()
            msgBox.setText("Record inserted successfully into users table")
            msgBox.exec_()

        except mc.Error as error:
            msgBox = QMessageBox()
            msgBox.setText("[Error DB] {} connetion failed.".format(error))
            msgBox.exec_()

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    
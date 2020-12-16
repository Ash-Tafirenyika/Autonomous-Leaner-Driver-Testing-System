# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:20:42 2020

@author: Dr~Newt
"""
import mysql.connector
from mysql.connector import Error

class DriverInfo():
    
    def __init__(self):
        pass
    
    
    def insertVariblesIntoTable(self.fname, lname, gender, dob,pic)
        try:
            connection = mysql.connector.connect(host='localhost',
                                                database='loginregister',
                                                user='root',
                                                password='')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO driverinfo (fname, lname, gender,D.O.B) 
                                    VALUES (%s, %s, %s, %s, %s) """

            recordTuple = (fname, lname, gender, dob, pic)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            print("Record inserted successfully into DRIVER table")

        except mysql.connector.Error as error:
            print("Failed to insert into MySQL table {}".format(error))

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    def booking(self,driverID, bookDateTime, LicenceClass)
        try:
            connection = mysql.connector.connect(host='localhost',
                                                database='loginregister',
                                                user='root',
                                                password='')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO bookings (driverID, bookDateTime, LicenceClass) 
                                    VALUES (%s, %s, %s) """

            recordTuple = (driverID, bookDateTime, LicenceClass)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            print("Record inserted successfully into DRIVER table")

        except mysql.connector.Error as error:
            print("Failed to insert into MySQL table {}".format(error))

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    def DriverTesResult(self,driverID, TaskID, Score, Decision)
        try:
            connection = mysql.connector.connect(host='localhost',
                                                database='loginregister',
                                                user='root',
                                                password='')
            cursor = connection.cursor()
            mySql_insert_query = """INSERT INTO results (self,driverID, TaskID, Score, Decision) 
                                    VALUES (%s, %s, %s, %s) """

            recordTuple = (self,driverID, TaskID, Score, Decision)
            cursor.execute(mySql_insert_query, recordTuple)
            connection.commit()
            print("Record inserted successfully into results table")

        except mysql.connector.Error as error:
            print("Failed to insert into MySQL table {}".format(error))

        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")

    
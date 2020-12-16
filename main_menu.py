
import sys
import os
import platform
from random import randint

import cv2
import mysql.connector as mc
import qimage2ndarray
from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QTimer, QDateTime, QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt, QEvent, QThreadPool, Slot)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence,
                           QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient, QImage)
from PySide2.QtWidgets import *
from driver_main_window import *
# from  utils5.CaptureAll_info import MyInfo
from original_main import Thread
from utils5.CaptureAll_info import MyInfo

import pyqtgraph as pg

WINDOW_SIZE = 0# Global value for the windows status determine if the window is minimized or maximized

# Main class
class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # Remove window tlttle bar
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        # Set main background to transparent
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        # Apply shadow effect
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 92, 157, 150))
        # Appy shadow to central widget
        self.ui.centralwidget.setGraphicsEffect(self.shadow)
        # Button click events to our top bar buttons
        ####################################THREAD VIDEO##########################################
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        ##########################################################################################
        # Minimize window
        self.ui.minimizeButton.clicked.connect(lambda: self.showMinimized())
        # Close window
        self.ui.closeButton.clicked.connect(lambda: self.close())
        # Restore/Maximize window
        self.ui.restoreButton.clicked.connect(lambda: self.restore_or_maximize_window())

        # ###############################################

        # ###############################################
        # Move window on mouse drag event on the tittle bar
        # ###############################################
        def moveWindow(e):
            # Detect if the window is  normal size
            # ###############################################
            if self.isMaximized() == False:  # Not maximized
                # Move window only when window is normal size
                # ###############################################
                # if left mouse button is clicked (Only accept left mouse button clicks)
                if e.buttons() == Qt.LeftButton:
                    # Move window
                    self.move(self.pos() + e.globalPos() - self.clickPosition)
                    self.clickPosition = e.globalPos()
                    e.accept()

        # ###############################################
        # Add click event/Mouse move event/drag event to the top header to move the window
        # ###############################################
        self.ui.main_header.mouseMoveEvent = moveWindow
        # ###############################################
        # Add click event/Mouse move event/drag event to the top header to move the window
        # ###############################################
        self.ui.main_header.mouseMoveEvent = moveWindow
        # SLIDABLE LEFT MENU/////////////////
        # Left Menu toggle button
        self.ui.left_menu_toggle_btn.clicked.connect(lambda: self.slideLeftMenu())
        # STACKED PAGES (DEFAUT /CURRENT PAGE)/////////////////
        # Set the page that will be visible by default when the app is opened
        self.ui.stackedWidget.setCurrentWidget(self.ui.home_page)
        # STACKED PAGES NAVIGATION/////////////////
        # Using side menu buttons
        # navigate to Home page
        self.ui.home_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.home_page))
        # navigate to Accounts page
        self.ui.accounts_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.accounts_page))
        # navigate to Settings page
        self.ui.settings_button.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.settings_page))
        # navigate to add driver
        self.ui.btnaddNewDriver.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.NewDriver_page))
        # navigate to preview window
        self.ui.btnPriewWin.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.prieviewWin_page))
        # navigate to new user
        self.ui.btnAddNewUser.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.addNewUser_page))
        # navigate to add new booking
        self.ui.btnAddNewBooking.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.NewBooking_page))
        # navigate to exit
        self.ui.btnShutdown.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.close()))
        # ############################################
        #Form buttons lOGIC
        self.ui.btnLogIn.clicked.connect(self.Login)#Handles login
        
        self.ui.btnView.clicked.connect(self.disTestImg)#refreshes prieview window
        self.ui.btnPipelineView_2.clicked.connect(lambda: self.stackedWidget.setCurrentWidget(self.ui.PipelineView_page))#moves to pipeline window
        self.ui.btnSaveNewUser.clicked.connect(self.AdduserFields)#for saving new user redord
        self.ui.btnSaveDriver.clicked.connect(self.AddDriverFields)#saves driver info
        self.ui.btnSaveBooking.clicked.connect(self.AddBookingFields)#saves new bookind

        self.plot1()
        self.plot2()
        # Show window
        self.show()
        # ###############################################
    # ###############################################
    # Add mouse events to the window
    # ###############################################
    def mousePressEvent(self, event):
        # ###############################################
        # Get the current position of the mouse
        self.clickPosition = event.globalPos()
        # We will use this value to move the window
        # ###############################################

    # Restore or maximize your window
    def restore_or_maximize_window(self):
        # Global windows state
        global WINDOW_SIZE  # The default value is zero to show that the size is not maximized
        win_status = WINDOW_SIZE

        if win_status == 0:
            # If the window is not maximized
            WINDOW_SIZE = 1  # Update value to show that the window has been maxmized
            self.showMaximized()
            # Update button icon  when window is maximized
            self.ui.restoreButton.setIcon(QtGui.QIcon(u":/icons/icons/cil-window-restore.png"))  # Show minized icon
        else:
            # If the window is on its default size
            WINDOW_SIZE = 0  # Update value to show that the window has been minimized/set to normal size (which is 800 by 400)
            self.showNormal()
            # Update button icon when window is minimized
            self.ui.restoreButton.setIcon(QtGui.QIcon(u":/icons/icons/cil-window-maximize.png"))  # Show maximize icon
    ########################################################################
    # Slide left menu
    ########################################################################
    def slideLeftMenu(self):
        # Get current left menu width
        width = self.ui.left_side_menu.width()
        # If minimized
        if width == 50:
            # Expand menu
            newWidth = 150
        # If maximized
        else:
            # Restore menu
            newWidth = 50
        # Animate the transition
        self.animation = QPropertyAnimation(self.ui.left_side_menu, b"minimumWidth")  # Animate minimumWidht
        self.animation.setDuration(250)
        self.animation.setStartValue(width)  # Start value is the current menu width
        self.animation.setEndValue(newWidth)  # end value is the new menu width
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.start()


    ##################################################################################
    Slot(QImage)
    def setImage(self, image):
        self.ui.pipelineVideoView_2.setPixmap(QPixmap.fromImage(image))

    def Login(self):
        USERNAME = self.ui.uname_2.text()
        PASSWORD = self.ui.password_2.text()

        try:

            mydb = mc.connect(
                host="localhost",
                user="root",
                password="",
                database="loginregister"

            )

            mycursor = mydb.cursor()
            mycursor.execute(
                "SELECT username,password from users where username like '" + USERNAME + "'and password like '" + PASSWORD + "'")
            result = mycursor.fetchone()

            if result == None:
                msgBox = QMessageBox()
                msgBox.setText("Incorrect Email & Password")
                msgBox.exec_()
            else:
                msgBox = QMessageBox()
                msgBox.setText("login successful, welcome %s"%USERNAME)
                msgBox.exec_()
                self.ui.uname_2.setText("")
                self.ui.password_2.setText("")
                self.ui.settings_button.setEnabled(True)
                self.ui.btnaddNewDriver.setEnabled(True)
                self.ui.btnAddNewBooking.setEnabled(True)
                self.ui.btnAddNewUser.setEnabled(True)
                self.ui.btnPriewWin.setEnabled(True)
                mc.close()
        except mc.Error as e:
            msgBox = QMessageBox()
            msgBox.setText("[Error DB] connetion failed.")
            msgBox.exec_()


    def AdduserFields(self):
        FULLNAWE = self.ui.fullname.text(" ")
        GENDER = self.ui.gender.text(" ")
        USERNAME = self.ui.uname.text(" ")
        PASSWORD = self.ui.password.text(" ")
        EMAIL = self.ui.useremail.text(" ")

        if FULLNAWE == "" or GENDER == "" or USERNAME == "" or PASSWORD == "" or EMAIL == "":
            msgBox = QMessageBox()
            msgBox.setText("All fields are required!!!")
            msgBox.exec_()
        else:
            infoToDb = MyInfo()
            infoToDb.NewUser(FULLNAWE, GENDER, USERNAME, PASSWORD, EMAIL)
            self.ui.fullname.setText(" ")
            self.ui.gender.setText(" ")
            self.ui.uname.setText(" ")
            self.ui.password.setText(" ")
            self.ui.useremail.setText(" ")

    def AddDriverFields(self):
        FNAME, LNAME = (self.ui.fullname.text()).split(" ")
        GENDER = self.ui.gender.text()
        DOB = self.ui.dateEdit.text()
        PHONENUM = self.ui.driverphoneNum.text()
        EMAIL = self.ui.driverEmail.text()
        ADDRESS = self.ui.driverAddress.text()
        IMAGE = self.ui.driverEmail.text()

        if FNAME == "" or LNAME == "" or GENDER == "" or DOB == "" or PHONENUM == "" or EMAIL == "" or ADDRESS == "" or IMAGE == "":
            msgBox = QMessageBox()
            msgBox.setText("All fields are required!!!")
            msgBox.exec_()
        else:
            infoToDb = MyInfo()
            infoToDb.NewUser(FNAME, LNAME, GENDER, DOB, PHONENUM, EMAIL, ADDRESS, IMAGE)
            self.ui.fullname.setText(" ")
            self.ui.gender.setText(" ")
            self.ui.dateEdit.setText(" ")
            self.ui.driverphoneNum.setText(" ")
            self.ui.driverEmail.setText(" ")
            self.ui.driverAddress.setText(" ")
            self.ui.driverEmail.setText(" ")

    def AddBookingFields(self):
        DRIVERID = self.ui.DriverId.text()
        BOOKDATETIME = self.ui.bookDate.text()
        LICENCECLASS = self.ui.licenceClass.text()

        if DRIVERID == "" or BOOKDATETIME == "" or LICENCECLASS == "":
            msgBox = QMessageBox()
            msgBox.setText("All fields are required!!!")
            msgBox.exec_()
        else:
            infoToDb = MyInfo()
            infoToDb.Newbooking(DRIVERID, BOOKDATETIME, LICENCECLASS)
            self.ui.DriverId.setText(" ")
            self.ui.bookDate.setText(" ")
            self.ui.licenceClass.setText(" ")

    def plot1(self):    #****************************************************************************************************
        self.x1 = list(range(100))  # 100 time points
        self.y1 = [randint(0, 100) for _ in range(100)]  # 100 data points
        pen1 = pg.mkPen(color=(255, 0, 0))
        self.data_line1 = self.ui.LinearAccelerationPlot_2.plot(self.x1, self.y1, pen=pen1)
        self.data_line1 = self.ui.distancePlot_2.plot(self.x1, self.y1, pen=pen1)
        self.timer1 = QTimer()
        self.timer1.setInterval(50)
        self.timer1.timeout.connect(self.update_plot_data1)
        self.timer1.start()

    def plot2(self):    #****************************************************************************************************
        self.x2 = list(range(100))  # 100 time points
        self.y2 = [randint(0, 100) for _ in range(100)]  # 100 data points
        pen = pg.mkPen(color=(255, 0, 0))
        self.data_line2 = self.ui.distancePlot_2.plot(self.x2, self.y2, pen=pen)
        self.timer2 = QTimer()
        self.timer2.setInterval(50)
        self.timer2.timeout.connect(self.update_plot_data2)
        self.timer2.start()

    def update_plot_data1(self):
        self.x1 = self.x1[1:]  # Remove the first y element.
        self.x1.append(self.x1[-1] + 1)  # Add a new value 1 higher than the last.
        self.y1 = self.y1[1:]  # Remove the first
        self.y1.append(randint(0, 100))  # Add a new random value.
        self.data_line1.setData(self.x1, self.y1)  # Update the data.

    def update_plot_data2(self):
        self.x2 = self.x2[1:]  # Remove the first y element.
        self.x2.append(self.x2[-1] + 1)  # Add a new value 1 higher than the last.
        self.y2 = self.y2[1:]  # Remove the first
        self.y2.append(randint(0, 100))  # Add a new random value.
        self.data_line2.setData(self.x2, self.y2)  # Update the data.

    def toggleMouse(self, state):
        if state == Qt.Checked:
            enabled = True
        else:
            enabled = False
        self.ui.distancePlot_2.setMouseEnabled(x=enabled, y=enabled)
        #*****************************************************************************************************

    def disTestImg(self):
        img = "defaultIMG.png"
        tes = cv2.imread(img)
        tes = cv2.resize(tes, None, fx=0.9, fy=0.9)
        videoframe = cv2.cvtColor(tes, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(videoframe)
        self.ui.pipelineVideoView_2.setPixmap(QPixmap.fromImage(image))

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     sys.exit(app.exec_())
# else:
#     print(__name__, "hh")

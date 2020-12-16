import numpy as np
import cv2
import glob
import re
import os
import pickle


class Camera_Calib():

    #perfoms calibration if no pickle file exists
    # or just load the pickle file if existing.
    def __init__(self, calibration_dir, pickle_file):
        # Initialize cameraCalib
        self.mtx = None
        self.dist = None
        self.img_size = None
        self.objpoints = []# objpoints = []  # 3d points in real world space
        self.imgpoints = []# imgpoints = []  # 2d points in image plane.
        self.frame_Width= 640
        self.frame_Height = 480
        
        if not os.path.isfile(pickle_file):
            # prepare object points: (0,0,0), (1,0,0), (2,0,0) .., (6,5,0)
            # The images may have different detected checker board dimensions!
            # Currently, possible dimension combinations are: (9,6), (8,6),
            # (9,5), (9,4) and (7,6)
            objp1 = np.zeros((6 * 9, 3), np.float32)
            objp1[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
            objp2 = np.zeros((6 * 8, 3), np.float32)
            objp2[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
            objp3 = np.zeros((5 * 9, 3), np.float32)
            objp3[:, :2] = np.mgrid[0:9, 0:5].T.reshape(-1, 2)
            objp4 = np.zeros((4 * 9, 3), np.float32)
            objp4[:, :2] = np.mgrid[0:9, 0:4].T.reshape(-1, 2)
            objp5 = np.zeros((6 * 7, 3), np.float32)
            objp5[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
            objp6 = np.zeros((6 * 5, 3), np.float32)
            objp6[:, :2] = np.mgrid[0:5, 0:6].T.reshape(-1, 2)

            text = 'Performing camera calibrations relative to chessboard images: '
            print('{}"./{}/calibration*.jpg"...'.format(text, calibration_dir))
            # geneates a list of calibration images
            images = glob.glob(calibration_dir + '/calibration*.jpg')

            # Step through the image list and search for chessboard corners
            for idx, fname in enumerate(images):
                img = cv2.imread(fname)
                img = cv2.resize(img, (self.frame_Width, self.frame_Height), None)
                img2 = np.copy(img)
                self.img_size = (img.shape[1], img.shape[0])
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Finds the chessboard corners using possible combinations of dimensions.
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                objp = objp1
                if not ret:#if ret == false
                    ret, corners = cv2.findChessboardCorners(
                        gray, (8, 6), None)
                    objp = objp2
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (9, 5), None)
                    objp = objp3
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (9, 4), None)
                    objp = objp4
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (7, 6), None)
                    objp = objp5
                if not ret:
                    ret, corners = cv2.findChessboardCorners(
                        gray, (5, 6), None)
                    objp = objp6
                # print("corners: ", corners.shape, "\n", corners)

                # If corners are found, add object points, image points
                if ret:
                    self.objpoints.append(objp)
                    self.imgpoints.append(corners)
                    cv2.drawChessboardCorners(img2, (corners.shape[1],corners.shape[0]),corners, ret)
                    ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                            self.img_size, None, None)

            # now time to save the results into a pickle file for later use without additional calculations saving computational time.
            try:
                with open(pickle_file, 'w+b') as pfile1:
                    text = 'Saving data to pickle file'
                    print('{}: {} ...'.format(text, pickle_file))
                    pickle.dump({'img_size': self.img_size,
                                 'mtx': self.mtx,
                                 'dist': self.dist,
                                 'rvecs': self.rvecs,
                                 'tvecs': self.tvecs},
                                pfile1, pickle.HIGHEST_PROTOCOL)
                    print("The Camera Calibration Data saved to", pickle_file)
            except Exception as e:
                print('The system was unable to save data to', pickle_file, ':', e)
                raise

        # previously saved pickle file of the distortion correction data
        # has been found.  go ahead and revive it.
        else:
            try:
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                    self.img_size = pickle_data['img_size']
                    self.mtx = pickle_data['mtx']
                    self.dist = pickle_data['dist']
                    self.rvecs = pickle_data['rvecs']
                    self.tvecs = pickle_data['tvecs']
                    del pickle_data
                    print("The Camera Calibration data restored from", pickle_file)
            except Exception as e:
                print('The system unable to restore camera calibration data from',
                      pickle_file, ':', e)
                raise

    # if the source image is now smaller than the original calibration image
    # just set it
    def setImageSize(self, img_shape):
        self.img_size = (img_shape[1], img_shape[0])

    # returns subset of the camera calibration result that
    def get(self):
        return self.mtx, self.dist, self.img_size

    # returns all of the camera calibration result that
    def getall(self):
        return self.mtx, self.dist, self.img_size, self.rvecs, self.tvecs
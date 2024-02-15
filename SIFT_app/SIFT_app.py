#!/usr/bin/env python3

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import numpy as np
import sys

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)
        
        self._cam_id = 0
        self._cam_fps = 2
        self._is_cam_enabled = False
        self._is_template_loaded = False
        self._show_homography = False  # Variable to control homography display

        # Set the initial size of the application window
        self.resize(800, 600)

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)
        self.homography_checkbox.stateChanged.connect(self.SLOT_toggle_homography)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        self.sift = cv2.SIFT_create()
        self.kp_image, self.desc_image = None, None

        # Feature matching
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

        # Create QLabel to display matches
        self.matches_label = QtWidgets.QLabel(self)
        self.matches_label.setGeometry(10, 400, 780, 180)
        self.matches_label.setAlignment(QtCore.Qt.AlignCenter)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        self.selected_template_img = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)
        print("Loaded template image file: " + self.template_path)

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                        bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.selected_template_img is not None:
            # Feature detection for the template image
            kp_template, desc_template = self.sift.detectAndCompute(self.selected_template_img, None)

            # Feature detection for the current frame
            kp_grayframe, desc_grayframe = self.sift.detectAndCompute(grayframe, None)

            # Feature matching
            matches = self.flann.knnMatch(desc_template, desc_grayframe, k=2)
            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:
                    good_points.append(m)

            # Homography and Perspective Transform
            if len(good_points) > 10 and self._show_homography:
                query_pts = np.float32([kp_template[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
                train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
                matches_mask = mask.ravel().tolist()

                # Perspective transform
                h, w = grayframe.shape
                pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, matrix)
                homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
                cv2.imshow("Homography", homography)

            # Draw matches on the images
            img_matches = cv2.drawMatches(self.selected_template_img, kp_template, grayframe, kp_grayframe, good_points, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Convert to QPixmap and display in QLabel
            matches_pixmap = self.convert_cv_to_pixmap(img_matches)
            self.matches_label.setPixmap(matches_pixmap)
            
        else:
            # If no template, display the original frame in the existing window
            pixmap = self.convert_cv_to_pixmap(grayframe)
            self.live_image_label.setPixmap(pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

    def SLOT_toggle_homography(self, state):
        # Toggle the display of homography based on checkbox state
        self._show_homography = state == QtCore.Qt.Checked

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())

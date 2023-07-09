'''
    File name         : detectors2.py
    File Description  : Detect objects in video frame
    Author            : Karl Gardner
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
import cv2
import copy



class Detectors(object):
    """Detectors class to detect objects in video frame
    Attributes:
        None
    """

    def __init__(self):
        """Initialize variables used by Detectors class
        Args:
            None
        Return:
            None
        """
        #self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.counter = 0

    def Detect(self, frame,frame_count):
        """Detect objects in video frame using following pipeline
            - Convert captured frame from BGR to GRAY
            - Perform Background Subtraction
            - Detect edges using Canny Edge Detection
              http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
            - Retain only edges within the threshold
            - Find contours
            - Find centroids for each valid contours
        Args:
            frame: single video frame
        Return:
            centers: vector of object centroids in a frame
        """
        ctlower = (46, 49, 18)
        ctupper = (116, 197, 67)
        #blacked = frame
        orig_frame = copy.copy(frame)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ctlower, ctupper)

        # Use Closing Method: Dilation followed by Erosion
        dilation = cv2.dilate(mask, None, iterations=3)
        erosion = cv2.erode(dilation, None, iterations=2)
        mask2 = np.ones_like(erosion)
        if frame_count >= 31380 and frame_count <= 32133:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-31380-32133.npy")
        if frame_count >= 32851 and frame_count <= 36484:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-32851-36484.npy")
        if frame_count >= 37146 and frame_count <= 41191:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-37146-41191.npy")
        if frame_count >= 37423 and frame_count <= 41191:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-37423-41191.npy")
        if frame_count >= 38344 and frame_count <= 38349:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-38344-38349.npy")
        if frame_count >= 38646 and frame_count <= 41191:
            mask2 = mask2 & np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/mask-38646-41191.npy")

        blacked = mask2*erosion

        # Find contours
        contours, hierarchy = cv2.findContours(blacked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Used for saving images
        """
        #if frame_count >= 33326 and frame_count <= 33377:
        #if frame_count >= 33996 and frame_count <= 34093:
        if frame_count >= 39000 and frame_count <= 39070:
            if frame_count % 3 == 0:
                im_out = cv2.imread("Resources/hexagonaltrajectories/superimposed2.png")
                zeros = np.zeros_like(blacked)
                cv2.drawContours(zeros,contours,-1, (255, 255, 255),-1)
                mask_crop = cv2.erode(zeros[320:551, 620:1151], None, iterations=4)
                #print("src.shape: ", posts.shape)
                #print("mask_crop.shape: ", mask_crop.shape)
                print("orig_frame[320:551, 620:1151].shape: ", orig_frame[320:551, 620:1151].shape)
                img1_bg = cv2.bitwise_and(im_out,im_out,mask=cv2.bitwise_not(mask_crop))
                img2_fg = cv2.bitwise_and(orig_frame[320:551, 620:1151],orig_frame[320:551, 620:1151],mask=mask_crop)
                im_out = cv2.add(img1_bg,img2_fg)
                cv2.imshow("Show",im_out)
                cv2.imwrite("Resources/hexagonaltrajectories/superimposed2.png", im_out)
                # cv2.waitKey(0)
        """

        centers = []  # vector of object centroids in a frame
        sizes = []
        # we only care about centroids with size of bug in this example
        # recommended to be tuned based on expected object size for
        # improved performance
        blob_min_radius = 4
        # Find centroid for each valid contours
        for cnt in contours:
            try:
                # Calculate and draw circle
                (x, y), r = cv2.minEnclosingCircle(cnt)
                radius = int(round(r))
                if radius>blob_min_radius:
                    cv2.circle(frame, (int(round(x)), int(round(y))), radius, (255, 255, 255), 2)
                    b = np.array([[x], [y]])
                    centers.append(b)
                    sizes.append(r)
            except ZeroDivisionError:
                pass
        # show contours of tracking objects
        # cv2.imshow('Track Bugs', frame)

        return centers, sizes

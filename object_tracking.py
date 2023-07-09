'''
    File name         : object_tracking.py
    File Description  : Multi Object Tracker Using Kalman Filter and Hungarian Algorithm
    Author            : Karl Gardner
    Date created      : 04/02/2021
    Date last modified: 04/02/2021
    Python Version    : 3.7
'''

# Import python libraries
import cv2
import copy
from detector import Detectors
from tracker import Tracker
import numpy as np
import matplotlib.pyplot as plt
import cellfun
from matplotlib import cm

"""Main function for multi object tracking
Usage:
    $ python2.7 objectTracking.py
Pre-requisite:
    - Python2.7
    - Numpy
    - SciPy
    - Opencv 3.0 for Python
Args:
    None
Return:
    None
"""

d_post = 100  # Diameter of posts in micrometers
circle_path = "../../MachineLearning/CellVideos2/6-11-2021-movie2-CEM-point4mLperhour.mp4"
time = [4,40]
#micronperpix = 1883.952/1312.5
micronperpix = 14*149/1388.0
centerxy = [989.0,570.5]


# Call drawcircles() function to get image of circles and circles vector
circle_img, videocircles,differences = cellfun.trackingcircles(circle_path,micronperpix,centerxy,time)


# Show the circles
"""
cv2.namedWindow("Circle Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Circle Image", 600, 400)
cv2.imshow("Circle Image", circle_img)
cv2.waitKey(0)
"""

# Create opencv video capture object
cap = cv2.VideoCapture("../../MachineLearning/CellVideos2/6-11-2021-movie1-CEM-point4mLperhour.mp4")
size = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
initfps = cap.get(cv2.CAP_PROP_FPS)
dt = 1/initfps


#badtracks = np.array([1,2,3,52,55,135,136,137])
badtracks = np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/badtracks.npy")
print("badtracks: ",badtracks)
#np.save("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/badtracks.npy",badtracks)


combine = [[54,58],[189,191]]

#badframes = np.array([31690,31920,32134,32850,32998,33029,33777,33957,34235,34300,34375,34486,34677,34820,35014,35452,35532,36404,36485,37145,37707,37735,37960,38050,38583,38645,38730,38940,39572,39590,39660,40048,40551,40820,40935,41043,41192,42390])
badframes = np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/badframes.npy")
print("badframes: ",badframes)
#np.save("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/badframes.npy",badframes)
# Minimum x length to plot in microns
minxlength = 900

totalframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Size of Video (W x H: ", size)
print("Initial frames/sec in video: ",initfps)
print("Total frames in video: ",totalframes)
# Create Object Detector
detector = Detectors()

flatten = []
for m in range(len(combine)):
    for n in range(len(combine[m])):
        flatten.append(combine[m][n])
# Create Object Tracker

tracker = Tracker(80, 8, 1000, 0,flatten)


# Variables initialization
track_colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,127,255),(127,0,255),(127,0,127)]
pause = False

# For saving the video to a file
#video_out = cv2.VideoWriter("Resources/edited7.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (int(size[0]), int(size[1])), True)



# Initialize more stuff
#starttime = [20,24]
#startframe = int(round((starttime[0]*60+starttime[1])*initfps))
#endframe = int(round((endtime[0]*60+endtime[1])*initfps))

#startframe = 31380
startframe = 39034

endframe = 41191

cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
frame_num = startframe
timer = cv2.getTickCount()
# Infinite loop to process video frames
while True:
    if np.any(frame_num == badframes):
        index2 = np.nonzero(frame_num == badframes)
        frame_num = badframes[index2[0][0]+1]+1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        continue

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    timer = cv2.getTickCount()

    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        continue
    if frame_num == endframe:
        break
    # Make copy of original frame
    orig_frame = copy.copy(frame)

    # Detect and return centroids of the objects in the frame
    centers, sizes = detector.Detect(frame,frame_num)
    # If centroids are detected then track them
    if len(centers) > 0 or len(tracker.assignment) > 0:

        # Track object using Kalman Filter
        tracker.Update(centers, sizes,frame_num,endframe-1,dt,videocircles,differences,micronperpix,centerxy,size[1],minxlength,badtracks,combine)

        # For identified object tracks draw tracking line with various colors to indicate different track_id
        for i in range(len(tracker.tracks)):
            if len(tracker.tracks[i].trace_x) > 1:
                for j in range(len(tracker.tracks[i].trace_x)-1):
                    # Draw trace line
                    x1 = tracker.tracks[i].trace_x[j]
                    y1 = tracker.tracks[i].trace_y[j]
                    x2 = tracker.tracks[i].trace_x[j+1]
                    y2 = tracker.tracks[i].trace_y[j+1]
                    clr = tracker.tracks[i].track_id % 9
                    cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), track_colors[clr], 2)
            # Display the track ID above the current (last) CTC position
            cv2.putText(frame,str(tracker.tracks[i].track_id), (int(round(tracker.tracks[i].trace_x[-1]))+20,int(round(tracker.tracks[i].trace_y[-1])+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #cv2.rectangle(frame,(720,620),(1000,790),(255,255,255),2)
    # Display the resulting tracking frame
    cv2.putText(frame, "Tracking Video", (75, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(frame, "Objective Lens: 10x", (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Flow Type: Negative Pressure", (75, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "CTC Type: PC3", (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Video Captured At: " + str(round(initfps, 1)) + " FPS", (75, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    cv2.putText(frame, "Playing At: "+str(round(fps,1))+" FPS", (75, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2)
    cv2.putText(frame, "Frame num: " + str(frame_num), (75, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),2)
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.resizeWindow("Tracking", 1000,700)
    cv2.imshow('Tracking', frame)

    # Display the original frame and save video of tracking frames
    cv2.putText(orig_frame, "Original Video", (75, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.putText(orig_frame, "Objective Lens: 10x", (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(orig_frame, "Flow Type: Pressure Difference", (75, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(orig_frame, "CTC Type: PC3", (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(orig_frame, "Playing and Taken At: " + str(round(initfps,2)) + " FPS", (75, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    #cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("Original", 600, 500)
    #cv2.imshow('Original', orig_frame)
    #video_out.write(frame)



    #Slow the FPS
    #cv2.waitKey(100)

    # Check for key strokes
    k = cv2.waitKey(50) & 0xff
    if k == 27:  # 'esc' key has been pressed, exit program.
        break
    if k == 112:  # 'p' has been pressed. this will pause/resume the code.
        pause = not pause
        if (pause is True):
            print("Code is paused. Press 'p' to resume..")
            # This is to append the bad frames from pausing
            """
            if counter % 2 == 0:
                badframes.append(frame_num)
                print("Frame " + str(frame_num)+" added to badframes")
            else:
                badframes.append(int(round(frame_num-15)))
                print("Frame " + str(frame_num-15) + " added to badframes")
            counter += 1
            """
            while (pause is True):
                # stay in this loop until
                key = cv2.waitKey(30) & 0xff
                if key == 112:
                    pause = False
                    print("Resume code..!!")
                    break
    frame_num += 1
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


fig, ax = plt.subplots()

cellfun.drawcircles(ax,circlevecvid=videocircles)
savepath = "Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/trajectorydata.csv"
cellfun.drawtraj(ax,fig,tracker.dumptrackid,tracker.trajectorydata,color="b")

ax.set_aspect('equal')
#plt.tight_layout()
#plt.savefig('Resources/test.png', bbox_inches='tight',pad_inches = 0)
plt.show()
cv2.waitKey(0)

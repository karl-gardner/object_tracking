import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import cellfun

path = "Resources/edited4.mp4"
cap = cv2.VideoCapture(path)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
initfps = int(cap.get(cv2.CAP_PROP_FPS))
totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Size of Video (W x H: ", size)
print("Initial frames/sec in video: ",initfps)
print("Total frames in video: ",totalframes)

# Initialize tracker, bounding box coordinate, x position, and y position
trackers = cv2.MultiTracker_create()
bboxarray = np.array([[0,0,0,0]])
x_pos = np.array([0])
y_pos = np.array([0])
box = None

counter = 0
# For saving the video to a file
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#video_out = cv2.VideoWriter("Resources/tracking.mp4", fourcc, 30, size, True)
while True:
    timer = cv2.getTickCount()
    #time.sleep(0.08)
    success, img = cap.read()
    if not success:
        break

    if box is not None:

        success, boxes = trackers.update(img)
        if success:
            img, xy_pos = cellfun.imdraw2(img, boxes)
            for m in range(counter):
                mm = str(m)
                exec('x_pos%s = np.append(x_pos%s, %d)' % (mm, mm, xy_pos[m, 0]))
                exec('y_pos%s = np.append(y_pos%s, %d)' % (mm, mm, xy_pos[m, 1]))
                exec('traj_len = x_pos%s.shape[0]' % mm)
                for i in range(traj_len):
                    exec('img[int(round(y_pos%s[i])), int(round(x_pos%s[i])), :] = [0, 0, 255]' % (mm, mm))
        else:
            cv2.putText(img, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img, "Playing at "+str(int(fps))+" FPS", (850, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Initial FPS: "+str(int(initfps))+" FPS", (850, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        box = cv2.selectROI("Bounding Box Selection", img)
        tracker = cv2.TrackerCSRT_create()
        trackers.add(tracker, img, box)
        img, xy_pos2 = cellfun.imdraw2(img, np.array([[box[0], box[1], box[2], box[3]]]))
        exec('x_pos%s = %d' % (counter, xy_pos2[0, 0]))
        exec('y_pos%s = %d' % (counter, xy_pos2[0, 1]))
        counter += 1
    elif key == ord("q"):
        break

    cv2.imshow("Video", img)

    # For saving the video to a file
    #video_out.write(img)

cap.release()
#video_out.release()
cv2.destroyAllWindows()

fig, ax = plt.subplots()
d_post = 100  # Diameter of posts in micrometers

# Call imfindcircle() function to get image of circles and circles vector
ax, circle_img, circles = cellfun.drawcircles(ax, path, 30, 43, size[0], size[1], d_post)

# Call drawtraj to display trajectories on axes with circles
for m in range(counter):
    mm = str(m)
    exec('ax = cellfun.drawtraj(ax, circles, size[0], size[1], x_pos%s, y_pos%s, d_post)' % (mm, mm))
#ax = cellfun.drawtraj(ax, circles, size[0], size[1], x_pos0, y_pos0, d_post)
plt.show()

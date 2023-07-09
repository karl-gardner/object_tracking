import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
import copy
import csv
from matplotlib import patches
import sys
import h5py


# Function to find circles from an mp4 video input file
def trackingcircles(filename,micronperpix,centerxy,timegrab):
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((timegrab[0]*60+timegrab[1])*fps)))
    success, image = cap.read()


    """
    image2 = image.copy()
    cv2.namedWindow("Select Circle", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Select Circle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    bbox1 = cv2.selectROI("Select Circle", image2)
    cv2.circle(image2, (int(round(bbox1[0]+bbox1[2]/2)), int(round(bbox1[1]+bbox1[3]/2))), int(round((bbox1[2]+bbox1[3])/4)), (0, 0, 255), 2)
    bbox2 = cv2.selectROI("Select Circle", image2)
    cv2.circle(image2, (int(round(bbox2[0]+bbox2[2]/2)),int(round(bbox2[1]+bbox2[3]/2))),int(round((bbox2[2]+bbox2[3])/4)),(0,0,255),2)
    bbox3 = cv2.selectROI("Select Circle", image2)
    cv2.circle(image2, (int(round(bbox3[0]+bbox3[2]/2)),int(round(bbox3[1]+bbox3[3]/2))),int(round((bbox3[2]+bbox3[3])/4)),(0,0,255),2)
    cv2.imshow("Select Circle",image2)
    circlevec = np.array([[bbox1[0]+bbox1[2]/2,bbox1[1]+bbox1[3]/2,(bbox1[2]+bbox1[3])/4],[bbox2[0]+bbox2[2]/2,bbox2[1]+bbox2[3]/2,(bbox2[2]+bbox2[3])/4],[bbox3[0]+bbox3[2]/2,bbox3[1]+bbox3[3]/2,(bbox3[2]+bbox3[3])/4]])
    np.save("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-0-/circlesvid.npy", circlevec)
    print("pixels: ",bbox2[0]+bbox2[2]/2-(bbox3[0]+bbox3[2]/2))
    print("center circle: ",bbox1[0]+bbox1[2]/2,bbox1[1]+bbox1[3]/2)
    

    # Continue with the rest of the circles
    circlevec = np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-0-/circlesvid.npy")
    while True:
        for (x, y, r) in circlevec:
            cv2.circle(image, (int(round(x)), int(round(y))), int(round(r)), (0, 0, 255), 2)
        cv2.namedWindow("Select Circle", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Select Circle", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        bbox = cv2.selectROI("Select Circle", image)
        if bbox[2] < 5:
            break
        circlevec = np.append(circlevec,[[bbox[0]+bbox[2]/2,bbox[1]+bbox[3]/2,(bbox[2]+bbox[3])/4]],axis=0)
    np.save("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-0-/circlesvid.npy", circlevec)
    """



    # Draw the actual circles on the figure
    circlevec = []
    #with open("Resources/overlaidtrajectories/postpositions_actual.csv", newline='') as csvfile:
    with open("Resources/overlaidtrajectories/postpositions_actual.csv", newline='') as csvfile:
        circlereader = csv.reader(csvfile)
        counter = 0
        for row in circlereader:
            counter += 1
            if counter == 1:
                continue
            circlevec.append([float(row[0]),float(row[1]),float(50)])
    circlevecactual = np.array(circlevec)


    # Load the post positions from the video geometry
    circlevecvid = np.load("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/circlesvid.npy")
    #np.savetxt("Resources/hexagonaltrajectories/CEMtrajectories/6-11-2021-movie1-31380-41191/circlepositions.txt",circlevecvid[:,:2])
    circlevecvid[:, 0] = circlevecvid[:, 0]*micronperpix#+(3262.33343116-centerxy[0]*micronperpix)
    circlevecvid[:, 1] = (1080-circlevecvid[:, 1])*micronperpix#+(1683.58621655-(1080-centerxy[1])*micronperpix)
    differences = np.zeros_like(circlevecvid)
    for i in range(circlevecvid.shape[0]):
        counter = 0
        mindist = 10000
        for j in range(circlevecactual.shape[0]):
            dist = math.sqrt((circlevecvid[i,0]-circlevecactual[j,0])**2+(circlevecvid[i,1]-circlevecactual[j,1])**2)
            if dist < mindist:
                mindist = dist
                index = counter
            counter += 1
        differences[i,:] = circlevecactual[index,:]-circlevecvid[i,:]

    """
    # Run for loop over all posts and find three closest distances for Ding

    with open("Resources/highlowExpression/quadpoints_database.csv", 'w', newline='') as csvfile:
        database = csv.writer(csvfile)
        database.writerow(["post x position","post y position","x1","y1","x2","y2","x3","y3","x4","y4"])

        # For loop over all posts to start
        #dx = np.zeros((circlevecactual.shape[0], circlevecactual.shape[0]))
        #dy = np.zeros((circlevecactual.shape[0], circlevecactual.shape[0]))
        box_len = 400
        quad_data = np.zeros((circlevecactual.shape[0],10))
        for i in range(circlevecactual.shape[0]):
            # One for loop to save numpy array of six closest posts from index of dist matrix
            dy_min1 = 10000
            dx_min2 = 10000
            dy_min3 = 10000
            dx_min4 = 10000
            idx1flag = True
            idx2flag = True
            idx3flag = True
            idx4flag = True
            for j in range(circlevecactual.shape[0]):
                dx = circlevecactual[j,0]-circlevecactual[i,0]
                dy = circlevecactual[j,1]-circlevecactual[i,1]
                if i == j:
                    dx = 10000
                    dy = 10000
                if dx > 0 and dx < box_len and abs(dy) < box_len:
                    idx1flag = False
                    if abs(dy) < dy_min1:
                        dy_min1 = abs(dy)
                        idx1 = j
                if dy < 0 and abs(dy) < box_len and abs(dx) < box_len:
                    idx2flag = False
                    if abs(dx) < dx_min2:
                        dx_min2 = abs(dx)
                        idx2 = j
                if dx < 0 and abs(dx) < box_len and abs(dy) < box_len:
                    idx3flag = False
                    if abs(dy) < dy_min3:
                        dy_min3 = abs(dy)
                        idx3 = j
                if dy > 0 and dy < box_len and abs(dx) < box_len:
                    idx4flag = False
                    if abs(dx) < dx_min4:
                        dx_min4 = abs(dx)
                        idx4 = j
            quad_data[i,0] = circlevecactual[i,0]
            quad_data[i,1] = circlevecactual[i,1]

            if idx1flag:
                quad_data[i,2:4] = np.nan
            else:
                quad_data[i,2] = circlevecactual[idx1,0]
                quad_data[i,3] = circlevecactual[idx1,1]
            if idx2flag:
                quad_data[i,4:6] = np.nan
            else:
                quad_data[i,4] = circlevecactual[idx2, 0]
                quad_data[i,5] = circlevecactual[idx2, 1]
            if idx3flag:
                quad_data[i,6:8] = np.nan
            else:
                quad_data[i,6] = circlevecactual[idx3,0]
                quad_data[i,7] = circlevecactual[idx3,1]
            if idx4flag:
                quad_data[i,8:10] = np.nan
            else:
                quad_data[i,8] = circlevecactual[idx4, 0]
                quad_data[i,9] = circlevecactual[idx4, 1]

            database.writerow(quad_data[i])
            #print("i: ",i)
    """
    return image,circlevecvid,differences


def drawcircles(ax,circlevecvid=False):

    # Draw the actual circles on the figure
    circlevec = []
    with open("Resources/overlaidtrajectories/postpositions_actual.csv", newline='') as csvfile:
    #with open("Resources/highlowExpression/circles_wholechip.csv", newline='') as csvfile:
        circlereader = csv.reader(csvfile)
        counter = 0
        for row in circlereader:
            counter += 1
            if counter == 1:
                continue
            circlevec.append([float(row[0]), float(row[1]), float(50)])

    circlevecactual = np.array(circlevec)

    #ax.set_xlim((-20, np.amax(circlevecactual[:, 0])+70))
    #ax.set_ylim((-20, np.amax(circlevecactual[:, 1])+70))
    ax.set_xlim((600, 1550))
    ax.set_ylim((370, 840))
    for i in range(circlevecactual.shape[0]):
        circle = plt.Circle((circlevecactual[i, 0], circlevecactual[i, 1]), 50, color='gray',alpha=0.4)
        #ax.add_artist(circle)

    # Draw the circles from the microscope video if the circlevecvid is passed
    if np.any(circlevecvid):
        for i in range(circlevecvid.shape[0]):
            circle = plt.Circle((circlevecvid[i, 0], circlevecvid[i, 1]),radius=78/2, color='gray')
            ax.add_artist(circle)


def drawtraj(ax, fig,trackids,data,ctclabel=False,color=False,cellvel=False,size=False,fluidvel=False,stream=False,save=False,images=False,sequence=False):
    if fluidvel:
        # This is to plot COMSOL velocities:
        fig, ax = plt.subplots()
        values = np.load("Resources/overlaidtrajectories/fluidvelocity.npy")
        ax.imshow(values,vmin=0,vmax=np.nanmax(values),extent=[0.6819762549592183,6819.0805733372235,0.3554905234788679,3554.549744265201],cmap="bone",alpha=0.75)
        cbar = fig.colorbar(cm.ScalarMappable(cmap="bone"),ax=ax,ticks=np.linspace(0, 1, 10),orientation="horizontal",pad=.05,aspect=80)
        cbar.ax.set_xticklabels([int(round(num)) for num in np.linspace(0, np.nanmax(values), 10)])
        #cbar.ax.set_ylabel("Fluid Velocities (COMSOL) [um]",rotation=270,labelpad=20)

    if stream:
        counter = 0
        with open("Resources/overlaidtrajectories/streamlines.csv",newline='') as csvfile:
            streamreader = csv.reader(csvfile)
            for row in streamreader:
                if counter%2 == 0:
                    x = np.array([])
                    for j in range(len(row)):
                        x = np.append(x,float(row[j]))
                if (counter+1)%2 == 0:
                    y = np.array([])
                    for j in range(len(row)):
                        y = np.append(y, float(row[j]))
                    ax.plot(x,y,c="grey", lw=.5)
                counter += 1

    if size:
        ax.scatter([50000], [50000], label="< "+str(round(size-2.5,2))+" um", color=color)
        ax.scatter([50000], [50000], label=str(round(size,2))+" um", color=color)
        ax.scatter([50000], [50000], label=str(round(size+2,2))+" um", color=color)
        ax.scatter([50000], [50000], label="> "+str(round(size+4,2))+" um", color=color)
        lgnd = ax.legend(loc="upper right")
        lgnd.legendHandles[0]._sizes = [0.5]
        lgnd.legendHandles[1]._sizes = [3]
        lgnd.legendHandles[2]._sizes = [5]
        lgnd.legendHandles[3]._sizes = [7]


    max_height = 0
    max_height_half = 0
    numtracks = len(trackids)
    X = []
    for i in range(numtracks):
        ID = trackids[i]
        if cellvel:
            if ID == 171 or ID == 168 or ID == 102 or ID == 191:
                #ax.plot(data["x"+str(ID)], data["y"+str(ID)], c="gray", lw=.05)
                #ax.scatter(data["x"+str(ID)][np.isnan(data["v_mag"+str(ID)])],data["y"+str(ID)][np.isnan(data["v_mag"+str(ID)])], s=3,c="gray")
                #ax.scatter(data["x"+str(ID)], data["y"+str(ID)], s=8,c=data["v_mag"+str(ID)], cmap='jet', vmin=0,vmax=data["maxvmag"])
                ax.scatter(data["x"+str(ID)],data["y"+str(ID)],s=70,c=data["v_mag"+str(ID)],cmap='jet',vmin=0,vmax=1400)

        if color:
            #ax.plot(data["x"+str(ID)], data["y"+str(ID)], c=color, lw=.05)
            if size:
                ax.scatter(data["x"+str(ID)],data["y"+str(ID)], s=data["diameter"+str(ID)], c=color)
            else:
                ax.scatter(data["x" + str(ID)], data["y" + str(ID)], s=5, c=color)
                #data["y"+str(ID)][0] >= 1700 and data["y"+str(ID)][0] <= 1800:
                #ax.text(data["x"+str(ID)][-1],data["y"+str(ID)][-1],str(ID))
                #ax.set_aspect('equal')
                #plt.show()

        if sequence:
            #plt.close()
            #fig, ax = plt.subplots()
            # Call drawcircles() function to get image of circles and circles vector
            #drawcircles(ax)

            # Only take data that have a velocity through boolean array indexing
            x_data_full = data["x"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            y_data_full = data["y"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            postdist1_full = data["postdist1"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            postdist2_full = data["postdist2"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            v_data_full = np.clip(data["v_mag"+str(ID)][~np.isnan(data["v_mag"+str(ID)])],a_min=None,a_max=2400)

            tot_length = x_data_full.shape[0]
            stride = 15
            window = 22
            num_examples = int((tot_length-window)/stride)
            # Initialize the trajectory images with the number of images as the first index

            # Loop over each training example and each trajectory point
            for n in range(num_examples):
                # Take all x_data and y_data less than range of W_p
                # Find the last "full trajectory" image then move to the end to get last image
                x_i = []

                x_data = x_data_full[n*stride:(n*stride+window)]
                y_data = y_data_full[n*stride:(n*stride+window)]
                v_data = v_data_full[n*stride:(n*stride+window)]
                postdist1 = postdist1_full[n*stride:(n*stride+window)]
                postdist2 = postdist2_full[n*stride:(n*stride+window)]



                for j in range(len(x_data)):
                    x_i.append([x_data[j], y_data[j],postdist1[j],postdist2[j],v_data[j]])

                X.append(x_i)


        if images:
            #plt.close()
            #fig, ax = plt.subplots()
            # Call drawcircles() function to get image of circles and circles vector
            drawcircles(ax)

            # Only take data that have a velocity through boolean array indexing
            x_data_full = data["x"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            y_data_full = data["y"+str(ID)][~np.isnan(data["v_mag"+str(ID)])]
            v_data_full = np.clip(data["v_mag"+str(ID)][~np.isnan(data["v_mag"+str(ID)])],a_min=None,a_max=2400)

            # First define the ranges of x and y in micrometers (try 700 and 200 microns) with minxlength in object_tracking = 900 mu
            W_mu = 700
            H_mu = 233
            AR = H_mu/ W_mu

            # Next define the ranges of the x and y in pixels of trajectory images but keep aspect ratio the same
            H_p = 40
            W_p = int(H_p/AR)
            print("W_p:", W_p)

            # Use CNN pixel equation: n_W^[l] = (n_W^[l-1]+2p-f)/s, to find total number of images
            stride = 250
            stride = 400
            traj_len = np.amax(x_data_full)-np.amin(x_data_full)
            #num_images = int(math.ceil((traj_len-W_mu)/stride) + 1)
            num_images = int((traj_len-W_mu)/stride)+1+1
            # Initialize the trajectory images with the number of images as the first index
            traj_image = np.zeros((num_images, H_p, W_p))



            # Loop over each image and each trajectory point
            for n in range(num_images):
                # Take all x_data and y_data less than range of W_p
                # Find the last "full trajectory" image then move to the end to get last image

                if n == num_images - 1:
                    # Add one pixel length here to ensure no index is -1
                    x_start =np.amax(x_data_full)-W_mu+W_mu/W_p
                    if np.amax(x_data_full)-x_end < 80:
                        continue
                    else:
                        x_end = np.amax(x_data_full)
                    bool = (x_data_full > x_start) & (x_data_full <= x_end)
                else:
                    x_start = np.amin(x_data_full)+n*stride
                    x_end = np.amin(x_data_full)+W_mu+n*stride
                    bool = (x_data_full >= x_start) & (x_data_full<x_end)

                x_data = x_data_full[bool]
                y_data = y_data_full[bool]
                v_data = v_data_full[bool]

                min_y = np.amin(y_data)
                max_y = np.amax(y_data)

                # Karl's Proposal: just center it with max and min in the y direction because the max_height is less than with centroid method
                # Make sure y values are in range of H_di
                if (np.amax(y_data) - np.amin(y_data)) >= H_mu:
                    #max_height = (np.amax(y_data) - np.amin(y_data))
                    sys.exit("Error: y_data is larger than H_di range")

                # If there are not more than a certain amount of velocity points then continue
                # In addition, if the spacing between any two points is greater than max_space then continue
                if x_data.shape[0] <= 15:
                    continue

                max_space = 120
                dontsave = False
                # If either of the two ends are more than max_space than continue
                if (x_data[0]-x_start) > max_space or (x_end-x_data[-1]) > max_space:
                    continue

                offset_y = (H_p-(max_y-min_y)*(H_p/H_mu))/2
                for j in range(len(x_data)):
                    if j != 0 and (x_data[j]-x_data[j-1]) > max_space:
                        dontsave = True
                        break
                    if n == num_images - 1:
                        x = int((x_data[j]-(x_end-W_mu))*(W_p/W_mu))-1
                    else:
                        x = int((x_data[j]-x_start)*(W_p/W_mu))
                    y = int((y_data[j]-min_y)*(H_p/H_mu)+offset_y)

                    # No if statement necessary because I throw an error before if it is not in the range
                    # Don't forget to flip the fucking image now
                    y = H_p - y - 1
                    traj_image[n, y, x] = v_data[j]



                # If there is too much space in between points, then continue with images
                if dontsave:
                    continue

                # Normalize the velocities and save images right after
                traj_image[n,:,:] *= 255/np.amax(traj_image[n,:,:])
                #cv2.imwrite(images+str(ID)+"_"+str(n)+".png",traj_image[n])

                if n == num_images - 1:
                    rect = patches.Rectangle((np.amax(x_data_full)-W_mu+W_mu/W_p,min_y-(H_mu-(max_y-min_y))/2),W_mu,H_mu,fill=False,color="black",lw=0.8)
                else:
                    rect = patches.Rectangle((np.amin(x_data_full)+n*stride,min_y-(H_mu-(max_y-min_y))/2),W_mu,H_mu,fill=False,color="black",lw=0.8)
                ax.add_patch(rect)

            # To plot the cell velocity colorbar on the bottom
            """
            cbar = fig.colorbar(cm.ScalarMappable(cmap='jet'),ax=ax,ticks=np.linspace(0, 1, 11),pad=0.07,fraction=0.08,orientation="horizontal",aspect=80)
            # cbar.ax.set_yticklabels([int(round(num)) for num in np.linspace(0, data["maxvmag"], 10)])
            labels = []
            for num in np.linspace(0, 2400 - 2400 / 10, 10):
                labels.append(str(int(num)))
            labels.append("> 2400")
            cbar.ax.set_xticklabels(labels)
            #cbar.ax.set_xlabel("Cell Velocities [um/s]", labelpad=10,fontweight="semibold")
            """


            ax.scatter(x_data_full, y_data_full, s=30, c=v_data_full,cmap='jet',vmin=0,vmax=np.amax(v_data_full))
            #plt.savefig(images+str(ID)+".png")
            #ax.set_xticks(range(2460,3200,35))
            #ax.set_yticks(range(2000,2400,30))
            #ax.grid(b=True, which="both", axis="both",color="black",alpha=0.8)
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.set_aspect('equal')
            plt.show()

        # To plot the velocity colorbar on the bottom

        if i == numtracks - 1:
            if cellvel:
                cbar = fig.colorbar(cm.ScalarMappable(cmap='jet'),ax=ax,ticks=np.linspace(0, 1, 11),pad=.12,fraction=0.04,orientation="horizontal",aspect=80)
                # cbar.ax.set_yticklabels([int(round(num)) for num in np.linspace(0, data["maxvmag"], 10)])
                labels = []
                for num in np.linspace(0, 1400 - 1400 / 10, 10):
                    labels.append(str(int(num)))
                labels.append("1400")
                cbar.ax.set_xticklabels(labels)
                # cbar.ax.set_xlabel("Cell Velocities [um/s]", labelpad=10,fontweight="semibold")


        # To plot a legend with labels
        """
        if ctclabel == "SKBR3 Cells":
            ax.scatter([50000], [50000], label="PC3 Cells", color="g")
            ax.scatter([50000], [50000], label=ctclabel, color=color)
            #lgnd = ax.legend(loc="upper right")
            #lgnd.legendHandles[0]._sizes = [30]
            #lgnd.legendHandles[1]._sizes = [30]
        """

        if save:
            # This is to write the trajectory data to a .csv file
            with open(save, 'a', newline='') as csvfile:
                trajectorywriter = csv.writer(csvfile)
                trajectorywriter.writerow(np.array([ID]))
                trajectorywriter.writerow(np.array([data["maxvmag"]]))
                trajectorywriter.writerow(np.array([data["diameter"+str(ID)]]))
                #trajectorywriter.writerow(data["postdist1"+str(ID)])
                #trajectorywriter.writerow(data["postdist2"+str(ID)])
                trajectorywriter.writerow(data["x"+str(ID)])
                trajectorywriter.writerow(data["y"+str(ID)])
                trajectorywriter.writerow(data["v_x"+str(ID)])
                trajectorywriter.writerow(data["v_y"+str(ID)])
                trajectorywriter.writerow(data["v_mag"+str(ID)])
                trajectorywriter.writerow('')
    #print("max_height: ", max_height)
    #print("len(X)",len(X))
    #with h5py.File("Resources/overlaidtrajectories/"+ctclabel+".h5", "w") as hdf:
        #hdf.create_dataset("examples", data=np.array(X))

    return ax

# Function to draw blacked out pixels
def blackout(frame,blackpoints):
    x_1 = blackpoints[0, :]
    y_1 = blackpoints[1, :]
    x_2 = blackpoints[2, :]
    y_2 = blackpoints[3, :]
    for i in range(len(x_1)):
        frame[y_1[i]:y_2[i], x_1[i]:x_2[i]] = 0

def drawline(count,cap):
    xpoints = []
    ypoints = []
    while True:
        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, frame = cap.read()
        cv2.namedWindow("Select Point", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Select Point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        bbox1 = cv2.selectROI("Select Point", frame)
        if bbox1[2] > 50:
            cv2.destroyWindow("Select Point")
            break
        xpoints.append(bbox1[0]+bbox1[2]/2)
        ypoints.append(bbox1[1]+bbox1[3]/2)
        count += 1
    drawframe = 255*np.ones_like(frame)
    for i in range(len(xpoints)):
        if i == 0:
            continue
        cv2.line(drawframe, (int(round(xpoints[i-1])), int(round(ypoints[i-1]))), (int(round(xpoints[i])), int(round(ypoints[i]))), (0,0,0), 2)
    return drawframe[:,:,0]

# Function to draw bounding box and trajectory for each image
def imdraw(img2, bbox, x_pos, y_pos):
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img2, (x,y), ((x+w), (y+h)), (255,0,0), 3, 1)
    for i in range(len(x_pos)):
        img2[int(round(y_pos[i])), int(round(x_pos[i])), :] = [0, 0, 255]
        #img2[int(round(y_pos[i]) + 1), int(round(x_pos[i])), :] = [0, 0, 255]
        #img2[int(round(y_pos[i]) - 1), int(round(x_pos[i])), :] = [0, 0, 255]
        #img2[int(round(y_pos[i])), int(round(x_pos[i]) + 1), :] = [0, 0, 255]
        #img2[int(round(y_pos[i])), int(round(x_pos[i]) - 1), :] = [0, 0, 255]
    cv2.putText(img2, "Tracking", (850, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img2

# Function to draw bounding boxes and trajectories for each image
def imdraw2(img2, bboxes):
    xyposition = np.zeros((bboxes.shape[0], 2))
    for m in range(bboxes.shape[0]):
        (x, y, w, h) = [bboxes[m,0], bboxes[m,1], bboxes[m,2], bboxes[m,3]]
        xyposition[m, :] = [(x+x+w)/2, (y+y+h)/2]
        cv2.rectangle(img2, (int(round(x)), int(round(y))), (int(round(x + w)), int(round(y + h))), (0, 255, 0), 2)
    return img2, xyposition

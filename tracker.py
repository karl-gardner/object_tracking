'''
    File name         : tracker.py
    File Description  : Tracker Using Kalman Filter & Hungarian Algorithm
    Author            : Srini Ananthakrishnan
    Date created      : 07/14/2017
    Date last modified: 07/16/2017
    Python Version    : 2.7
'''

# Import python libraries
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track(object):
    """Track class for every object to be tracked
    Attributes:
        None
    """

    def __init__(self, prediction, trackIdCount):
        """Initialize variables used by Track class
        Args:
            prediction: predicted centroids of object to be tracked
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.track_id = trackIdCount  # identification of each track object
        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace_x = []  # trace path
        self.trace_y = []
        self.radii = []
        self.frame = []  # Karl's addition for frame number of each detection
        self.dx = []  # Karl's addition for velocity calculations
        self.dy = []  # Karl's addition for velocity calculations
        self.dmag = []


class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """

    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length, trackIdCount,flatten):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold, track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = (0.5)*dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.tracks = []
        self.trackIdCount = trackIdCount
        self.max_trace_length = max_trace_length
        self.assignment = []
        self.dumptrackid = []
        self.trajectorydata = {}
        self.trajectorydata["maxvmag"] = 0
        self.flattened = np.array(flatten)


    def Update(self, detections, detectionsizes,frame_count,end_frame,dt,circles,micdiff,mperp,center_xy,height,minxlength,badtracks,combine):
        """Update tracks vector using following steps:
            - Create tracks if no tracks vector found
            - Calculate cost using sum of square distance
              between predicted vs detected centroids
            - Using Hungarian Algorithm assign the correct
              detected measurements to predicted tracks
              https://en.wikipedia.org/wiki/Hungarian_algorithm
            - Identify tracks with no assignment, if any
            - If tracks are not detected for long time, remove them
            - Now look for un_assigned detects
            - Start new tracks
            - Update KalmanFilter state, lastResults and tracks trace
        Args:
            detections: detected centroids of object to be tracked
        Return:
            None
        """
        # Create tracks if no tracks vector found
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
        # Calculate cost using sum of square distance between predicted vs detected centroids
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros(shape=(N, M))   # Cost matrix
        for i in range(len(self.tracks)):
            if M == 0:
                cost = np.array([[0]])
                break
            for j in range(M):
                # Try just using the previous trace value for the cost here: Fuck the Kalman Filter
                # diff = self.tracks[i].prediction - detections[j]
                if len(self.tracks[i].trace_x) == 0:
                    diff = [[0],[0]]
                else:
                    diff = np.array([[self.tracks[i].trace_x[-1]],[self.tracks[i].trace_y[-1]]])-detections[j]
                    #diff = self.tracks[i].trace[-1] - detections[j]
                distance = np.sqrt(diff[0][0]*diff[0][0]+diff[1][0]*diff[1][0])
                cost[i][j] = distance


        # Let's average the squared ERROR
        cost = cost*0.5
        # If all of the rows (current tracks) of a detection (column) is greater than the threshold
        # distance then take it out and start a new track with it right away
        boolean1 = np.all(cost > self.dist_thresh, axis=0)
        un_assigned_detects2 = []
        un_assigned_detects2sizes = []
        cost = cost[:,~boolean1]
        for i in reversed(range(len(boolean1))):
            if boolean1[i]:
                un_assigned_detects2.append(detections[i])
                un_assigned_detects2sizes.append(detectionsizes[i])
                del detections[i]
                del detectionsizes[i]

        if np.all(boolean1):
            cost = np.array([[0]])
        # Do the same thing with tracks except if an entire row is greater than
        # the threshold distance just set that row to the largest value of that row
        boolean2 = np.all(cost > self.dist_thresh, axis=1)
        for i in range(boolean2.shape[0]):
            if boolean2[i]:
                cost[i,:] = np.amax(cost[i,:])

        # Using Hungarian Algorithm assign the correct detected measurements to tracks
        self.assignment = []
        for _ in range(N):
            self.assignment.append(-1)
        if len(detections) != 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                self.assignment[row_ind[i]] = col_ind[i]
        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(self.assignment)):
            if self.assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost[i][self.assignment[i]] > self.dist_thresh):
                    self.assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1
        # If tracks are not detected for a long time, remove them
        del_tracks = []
        for i in range(len(self.tracks)):
            if (self.tracks[i].skipped_frames > self.max_frames_to_skip):
                del_tracks.append(i)

        for d in range(len(del_tracks)):
            del_tracks[d] = del_tracks[d]-d

        if len(del_tracks) > 0:  # only when skipped frame exceeds max
            for id in del_tracks:
                if id < len(self.tracks):
                    # Karl's Addition to save self.tracks array before every deletion
                    x_values = np.array(self.tracks[id].trace_x)*mperp#+(3262.33343116-center_xy[0]*mperp)
                    y_values = (height-np.array(self.tracks[id].trace_y))*mperp#+(1683.58621655-(height-center_xy[1])*mperp)

                    # Since the microscope distorts the image a little find the minimum post distance of all the points and
                    # subtract the differences from the posts
                    postdist_x = circles[:, 0][:, np.newaxis] - x_values
                    postdist_y = circles[:, 1][:, np.newaxis] - y_values
                    postdist_mag = np.sqrt(postdist_x**2+postdist_y**2)
                    minimums = np.argmin(postdist_mag, axis=0)

                    min_post_dists = np.zeros((2,x_values.shape[0]))

                    for i in range(x_values.shape[0]):
                        x_values[i] = x_values[i]# + micdiff[minimums[i], 0]
                        y_values[i] = y_values[i]# + micdiff[minimums[i], 1]
                        min_post_dists[0,i] = np.amin(postdist_mag[:,i])
                        postdist_mag[minimums[i],i] = 1000
                        min_post_dists[1,i] = np.amin(postdist_mag[:,i])

                    if (np.amax(x_values) - np.amin(x_values) > minxlength and np.all(badtracks != self.tracks[id].track_id)) or np.any(self.tracks[id].track_id == self.flattened):
                        self.dumptrackid.append(self.tracks[id].track_id)
                        self.trajectorydata["x"+str(self.tracks[id].track_id)] = x_values
                        self.trajectorydata["y"+str(self.tracks[id].track_id)] = y_values
                        self.tracks[id].dx.append(np.nan)
                        self.tracks[id].dy.append(np.nan)
                        self.tracks[id].dmag.append(np.nan)
                        self.trajectorydata["v_x"+str(self.tracks[id].track_id)] = np.array(self.tracks[id].dx)*mperp/dt
                        self.trajectorydata["v_y"+str(self.tracks[id].track_id)] = np.array(self.tracks[id].dy)*mperp/dt
                        self.trajectorydata["v_mag"+str(self.tracks[id].track_id)] = np.array(self.tracks[id].dmag)*mperp/dt
                        # Use the radius of the smallest velocity for the diameter of the cell but if
                        # all of the velocities are nan then use the smallest radius
                        if np.any(~np.isnan(np.array([self.tracks[id].dmag]))):
                            sizeindex = np.nanargmin(np.array([self.tracks[id].dmag]))
                        else:
                            sizeindex = np.argmin(np.array([self.tracks[id].radii]))
                        self.trajectorydata["diameter"+str(self.tracks[id].track_id)] = self.tracks[id].radii[sizeindex]*2*mperp
                        self.trajectorydata["postdist1"+str(self.tracks[id].track_id)] = min_post_dists[0,:]
                        self.trajectorydata["postdist2"+str(self.tracks[id].track_id)] = min_post_dists[1,:]

                        # Save the maximum velocity before trajectories are combined
                        maxv = np.nanmax(self.trajectorydata["v_mag" + str(self.tracks[id].track_id)])
                        if maxv > self.trajectorydata["maxvmag"]:
                            self.trajectorydata["maxvmag"] = maxv

                        # This is to combine trajectories in the combine list
                        for m in range(len(combine)):
                            for n in range(len(combine[m])):
                                if self.tracks[id].track_id == combine[m][n] and n != 0:
                                    self.trajectorydata["x"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["x"+str(combine[m][n-1])],self.trajectorydata["x"+str(self.tracks[id].track_id)])
                                    self.trajectorydata["y"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["y"+str(combine[m][n-1])],self.trajectorydata["y"+str(self.tracks[id].track_id)])
                                    self.trajectorydata["v_x"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["v_x"+str(combine[m][n-1])],self.trajectorydata["v_x"+str(self.tracks[id].track_id)])
                                    self.trajectorydata["v_y"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["v_y"+str(combine[m][n-1])],self.trajectorydata["v_y"+str(self.tracks[id].track_id)])
                                    self.trajectorydata["v_mag"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["v_mag"+str(combine[m][n-1])],self.trajectorydata["v_mag"+str(self.tracks[id].track_id)])
                                    # Add the or statement in case there are velocities in current self.tracks[id].track_id but no velocities in the combine[m][n-1]
                                    if np.nanmin(self.trajectorydata["v_mag"+str(self.tracks[id].track_id)]) < np.nanmin(self.trajectorydata["v_mag"+str(combine[m][n-1])]) or (np.any(~np.isnan(np.array([self.tracks[id].dmag]))) and np.all(np.isnan(self.trajectorydata["v_mag"+str(combine[m][n-1])]))):
                                        self.trajectorydata["diameter"+str(self.tracks[id].track_id)] = self.tracks[id].radii[np.nanargmin(np.array(self.tracks[id].dmag))]*2*mperp
                                    else:
                                        self.trajectorydata["diameter"+str(self.tracks[id].track_id)] = self.trajectorydata["diameter"+str(combine[m][n-1])]
                                    self.trajectorydata["postdist1"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["postdist1"+str(combine[m][n-1])],self.trajectorydata["postdist1"+str(self.tracks[id].track_id)])
                                    self.trajectorydata["postdist2"+str(self.tracks[id].track_id)] = np.append(self.trajectorydata["postdist2"+str(combine[m][n-1])],self.trajectorydata["postdist2"+str(self.tracks[id].track_id)])
                                    del self.trajectorydata["x"+str(combine[m][n-1])]
                                    del self.trajectorydata["y"+str(combine[m][n-1])]
                                    del self.trajectorydata["v_x"+str(combine[m][n-1])]
                                    del self.trajectorydata["v_y"+str(combine[m][n-1])]
                                    del self.trajectorydata["v_mag"+str(combine[m][n-1])]
                                    del self.trajectorydata["diameter"+str(combine[m][n-1])]
                                    del self.trajectorydata["postdist1"+str(combine[m][n-1])]
                                    del self.trajectorydata["postdist2"+str(combine[m][n-1])]
                                    # Try python .remove next time for simplicity!!
                                    index = np.nonzero(combine[m][n-1] == np.array(self.dumptrackid))[0][0]
                                    del self.dumptrackid[index]
                                    if self.tracks[id].track_id == combine[m][-1] and np.amax(self.trajectorydata["x"+str(self.tracks[id].track_id)])-np.amin(self.trajectorydata["x"+str(self.tracks[id].track_id)]) <= minxlength:
                                        del self.trajectorydata["x"+str(combine[m][n])]
                                        del self.trajectorydata["y"+str(combine[m][n])]
                                        del self.trajectorydata["v_x"+str(combine[m][n])]
                                        del self.trajectorydata["v_y"+str(combine[m][n])]
                                        del self.trajectorydata["v_mag"+str(combine[m][n])]
                                        del self.trajectorydata["diameter"+str(combine[m][n])]
                                        del self.trajectorydata["postdist1"+str(combine[m][n])]
                                        del self.trajectorydata["postdist2"+str(combine[m][n])]
                                        index2 = np.nonzero(combine[m][n] == np.array(self.dumptrackid))[0][0]
                                        del self.dumptrackid[index2]

                    del self.tracks[id]
                    del self.assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")
                    exit()

        # Now look for un_assigned detects
        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in self.assignment:
                un_assigned_detects.append(i)

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(detections[un_assigned_detects[i]], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                self.tracks[-1].trace_x.append(detections[un_assigned_detects[i]][0,0])
                self.tracks[-1].trace_y.append(detections[un_assigned_detects[i]][1,0])
                self.tracks[-1].radii.append(detectionsizes[un_assigned_detects[i]])
                self.tracks[-1].frame.append(frame_count)

        if len(un_assigned_detects2) != 0:
            for i in range(len(un_assigned_detects2)):
                track = Track(un_assigned_detects2[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)
                self.tracks[-1].trace_x.append(un_assigned_detects2[i][0,0])
                self.tracks[-1].trace_y.append(un_assigned_detects2[i][1,0])
                self.tracks[-1].radii.append(un_assigned_detects2sizes[i])
                self.tracks[-1].frame.append(frame_count)

        # Update KalmanFilter state, lastResults, and tracks trace
        counter = 0
        for i in range(len(self.assignment)):
            if(self.assignment[i] != -1):
                self.tracks[i].skipped_frames = 0

                # Append the actual detections instead of predictions and add frame number
                self.tracks[i].trace_x.append(detections[self.assignment[i]][0,0])
                self.tracks[i].trace_y.append(detections[self.assignment[i]][1,0])
                self.tracks[i].radii.append(detectionsizes[self.assignment[i]])
                self.tracks[i].frame.append(frame_count)
                if len(self.tracks[i].trace_x) > 1:
                    if self.tracks[i].frame[-1]-self.tracks[i].frame[-2] == 1:
                        self.tracks[i].dx.append(self.tracks[i].trace_x[-1]-self.tracks[i].trace_x[-2])
                        self.tracks[i].dy.append(self.tracks[i].trace_y[-1]-self.tracks[i].trace_y[-2])
                        self.tracks[i].dmag.append(np.sqrt(self.tracks[i].dx[-1]**2+self.tracks[i].dy[-1]**2))
                    else:
                        counter += 1
                        self.tracks[i].dx.append(np.nan)
                        self.tracks[i].dy.append(np.nan)
                        self.tracks[i].dmag.append(np.nan)
                        
            if (len(self.tracks[i].trace_x) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace_x) - self.max_trace_length):
                    del self.tracks[i].trace_x[j]
                    del self.tracks[i].trace_y[j]
                    del self.tracks[i].radii[j]

        # Save everything from current tracks
        if frame_count == end_frame:
            for i in range(len(self.tracks)):
                x_values = np.array(self.tracks[i].trace_x)*mperp#+(3262.33343116-center_xy[0]*mperp)
                y_values = (height-np.array(self.tracks[i].trace_y))*mperp#+(1683.58621655-(height-center_xy[1])*mperp)
                # Since the microscope distorts the image a little find the minimum post distance of all the points and
                # subtract the differences from the posts
                postdist_x = circles[:, 0][:, np.newaxis] - x_values
                postdist_y = circles[:, 1][:, np.newaxis] - y_values
                postdist_mag = np.sqrt(postdist_x**2+postdist_y**2)
                minimums = np.argmin(postdist_mag, axis=0)
                min_post_dists = np.zeros((2, x_values.shape[0]))
                for j in range(x_values.shape[0]):
                    x_values[j] = x_values[j]#+micdiff[minimums[j], 0]
                    y_values[j] = y_values[j]#+micdiff[minimums[j], 1]
                    min_post_dists[0,j] = np.amin(postdist_mag[:,j])
                    postdist_mag[minimums[j],j] = 500
                    min_post_dists[1,j] = np.amin(postdist_mag[:,j])

                if (np.amax(x_values)-np.amin(x_values) > minxlength and np.all(badtracks != self.tracks[i].track_id)) or np.any(self.tracks[i].track_id == self.flattened):
                    self.dumptrackid.append(self.tracks[i].track_id)
                    self.trajectorydata["x"+str(self.tracks[i].track_id)] = x_values
                    self.trajectorydata["y"+str(self.tracks[i].track_id)] = y_values
                    self.tracks[i].dx.append(np.nan)
                    self.tracks[i].dy.append(np.nan)
                    self.tracks[i].dmag.append(np.nan)
                    self.trajectorydata["v_x"+str(self.tracks[i].track_id)] = np.array(self.tracks[i].dx)*mperp/dt
                    self.trajectorydata["v_y"+str(self.tracks[i].track_id)] = np.array(self.tracks[i].dy)*mperp/dt
                    self.trajectorydata["v_mag"+str(self.tracks[i].track_id)] = np.array(self.tracks[i].dmag)*mperp/dt
                    # Use the radius of the smallest velocity for the diameter of the cell but if
                    # all of the velocities are nan then use the smallest radius
                    if np.any(~np.isnan(np.array([self.tracks[i].dmag]))):
                        sizeindex = np.nanargmin(np.array([self.tracks[i].dmag]))
                    else:
                        sizeindex = np.argmin(np.array([self.tracks[i].radii]))
                    self.trajectorydata["diameter"+str(self.tracks[i].track_id)] = self.tracks[i].radii[sizeindex]*2*mperp
                    self.trajectorydata["postdist1"+str(self.tracks[i].track_id)] = min_post_dists[0,:]
                    self.trajectorydata["postdist2"+str(self.tracks[i].track_id)] = min_post_dists[1,:]

                    maxv = np.nanmax(self.trajectorydata["v_mag"+str(self.tracks[i].track_id)])
                    if maxv > self.trajectorydata["maxvmag"]:
                        self.trajectorydata["maxvmag"] = maxv

                    # This is to combine trajectories in the combine list
                    for m in range(len(combine)):
                        for n in range(len(combine[m])):
                            if self.tracks[i].track_id == combine[m][n] and n != 0:
                                self.trajectorydata["x"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["x"+str(combine[m][n-1])],self.trajectorydata["x"+str(self.tracks[i].track_id)])
                                self.trajectorydata["y"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["y"+str(combine[m][n-1])],self.trajectorydata["y"+str(self.tracks[i].track_id)])
                                self.trajectorydata["v_x"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["v_x"+str(combine[m][n-1])],self.trajectorydata["v_x"+str(self.tracks[i].track_id)])
                                self.trajectorydata["v_y"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["v_y" + str(combine[m][n-1])],self.trajectorydata["v_y"+str(self.tracks[i].track_id)])
                                self.trajectorydata["v_mag" + str(self.tracks[i].track_id)] = np.append(self.trajectorydata["v_mag"+str(combine[m][n-1])],self.trajectorydata["v_mag"+str(self.tracks[i].track_id)])
                                if np.nanmin(self.trajectorydata["v_mag"+str(self.tracks[i].track_id)]) < np.nanmin(self.trajectorydata["v_mag"+str(combine[m][n-1])]) or (np.any(~np.isnan(np.array([self.tracks[i].dmag]))) and np.all(np.isnan(self.trajectorydata["v_mag"+str(combine[m][n-1])]))):
                                    self.trajectorydata["diameter"+str(self.tracks[i].track_id)] = self.tracks[i].radii[np.nanargmin(np.array(self.tracks[i].dmag))]*2*mperp
                                else:
                                    self.trajectorydata["diameter"+str(self.tracks[i].track_id)] = self.trajectorydata["diameter"+str(combine[m][n-1])]
                                self.trajectorydata["postdist1"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["postdist1"+str(combine[m][n-1])],self.trajectorydata["postdist1"+str(self.tracks[i].track_id)])
                                self.trajectorydata["postdist2"+str(self.tracks[i].track_id)] = np.append(self.trajectorydata["postdist2"+str(combine[m][n-1])],self.trajectorydata["postdist2" + str(self.tracks[i].track_id)])

                                del self.trajectorydata["x"+str(combine[m][n-1])]
                                del self.trajectorydata["y"+str(combine[m][n-1])]
                                del self.trajectorydata["v_x"+str(combine[m][n-1])]
                                del self.trajectorydata["v_y"+str(combine[m][n-1])]
                                del self.trajectorydata["v_mag"+str(combine[m][n-1])]
                                del self.trajectorydata["diameter"+str(combine[m][n-1])]
                                del self.trajectorydata["postdist1"+str(combine[m][n-1])]
                                del self.trajectorydata["postdist2"+str(combine[m][n-1])]
                                index = np.nonzero(combine[m][n-1] == np.array(self.dumptrackid))[0][0]
                                del self.dumptrackid[index]

                                if self.tracks[i].track_id == combine[m][-1] and np.amax(self.trajectorydata["x"+str(self.tracks[i].track_id)])-np.amin(self.trajectorydata["x"+str(self.tracks[i].track_id)]) <= minxlength:
                                    del self.trajectorydata["x"+str(combine[m][n])]
                                    del self.trajectorydata["y"+str(combine[m][n])]
                                    del self.trajectorydata["v_x"+str(combine[m][n])]
                                    del self.trajectorydata["v_y"+str(combine[m][n])]
                                    del self.trajectorydata["v_mag"+str(combine[m][n])]
                                    del self.trajectorydata["diameter"+str(combine[m][n])]
                                    del self.trajectorydata["postdist1"+str(combine[m][n-1])]
                                    del self.trajectorydata["postdist2"+str(combine[m][n-1])]
                                    index2 = np.nonzero(combine[m][n] == np.array(self.dumptrackid))[0][0]
                                    del self.dumptrackid[index2]

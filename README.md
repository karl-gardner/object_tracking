<h4 align="center">Author: Karl Gardner<br>PhD Candidate, Department of Chemical Engineering, Texas Tech University</h4>

<div align="center">
  <a href="https://www.depts.ttu.edu/che/research/li-lab/">
  <img src="https://user-images.githubusercontent.com/91646805/154190573-53e361f6-7c60-4062-b56b-7cbd11d39fc4.jpg"/></a><br><br>
  
  <a href="https://www.depts.ttu.edu/che/research/li-lab/">
  <img src="https://user-images.githubusercontent.com/91646805/156635015-0cdcb0bb-0482-4693-b096-04f2a78f6b8e.svg" height="32"/></a>
  
  <a href="https://vanapallilab.wixsite.com/microfluidics">
  <img src="https://user-images.githubusercontent.com/91646805/156635010-a1049d8a-a72e-4ed5-89ec-2ace11169d85.svg" height="32"/></a>
  
  <a href="https://www.depts.ttu.edu/che/">
  <img src="https://user-images.githubusercontent.com/91646805/156641068-be8f0336-89b5-43e9-aa64-39481ce37c94.svg" height="32"/></a>
  
  <a href="https://roboflow.com/">
  <img src="https://user-images.githubusercontent.com/91646805/156641388-c609a6aa-8fce-47f0-a111-abfde9c5da05.svg" height="32"/></a><br>
  
  <a href="https://www.rsc.org/journals-books-databases/about-journals/lab-on-a-chip/">
  <img src="https://user-images.githubusercontent.com/91646805/169677461-13cb1d50-e7cf-457e-8777-cc6df29ce0bd.svg" height="32"/></a>
  
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov3.ipynb">
  <img src="https://user-images.githubusercontent.com/91646805/156640198-51f0ef4c-21c1-4d0f-aebd-861561dede95.svg" height="32"/></a>
  
  <a href="https://colab.research.google.com/github/karl-gardner/droplet_detection/blob/master/yolov5.ipynb">
  <img src="https://user-images.githubusercontent.com/91646805/156640073-0a7ad496-7691-4e1c-822c-b78f3e7d070b.svg" height="32"/></a>
  
  <a href="https://github.com/ultralytics">
  <img src="https://user-images.githubusercontent.com/91646805/156641066-fbc3635b-f373-4cb7-b141-9bcaad21beff.svg" height="32"/></a>



# Multi-Cell Tracking with Hungarian Algorithm
The most suitable method to track multiple moving objects simultaneously is to solve the assignment problem with the Hungarian method. An example of this problem is assigning three workers to three separate jobs, but each worker demands different pay for the various tasks. While the Hungarian algorithm creates a cost matrix to solve this problem in polynomial time, the result is finding the lowest cost way to assign the jobs. Illustrated in the figure below, the object detection pipeline with this Hungarian method describes the detailed workflow starting with a frame-by-frame analysis of the fluorescent cells flowing to the right in a microfluidic device. While there are a total of three cells in each frame, track 1, track 2, and track 3 are assigned to the top, middle, and bottom cell respectively. Therefore, in the second frame (the cells that the red arrows are pointing to) the three cells must be assigned to the three tracks that were instantiated in the first frame. A cost matrix is created based on the distances between the cells in the first and second frame, e.g., d_12 as the distance between the first cell in frame one and second cell in the frame two. Then, the Hungarian method assigns the cells with the shortest distances to the appropriate tracks, i.e., the cell that d_11 points to track 1. The bottom of the figure summarizes the object tracking pipeline in full and therefore includes both cell detection (top) and cell tracking (bottom). This object detector uses color detection with dilation and erosion to ultimately obtain contour centroids for the middle of each cell. Furthermore, the contour centroid is used to create and assign the trackID for input into SciPy’s linear sum assignment function that employs the Hungarian algorithm to assign the cell to the correct trackID [6]. Finally, unassigned tracks and detections are handled, and the x and y positions of the contour centroids are dumped to a csv file or other file type to use in a specific task.
</div>

![project_workflow](https://github.com/karl-gardner/object_tracking/assets/91646805/7671b392-249d-4b68-b6ed-faeccec956e5)

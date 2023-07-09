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


<details>
<summary>Instructions (click to expand)</summary>
<br>

1) First create a folder in your google drive account called droplet_classification (This step is important in order to keep the directories in check)
2) Use this link <a href="https://drive.google.com/drive/folders/1Oo68HSdU-jzcBAEr0yeRuzuSxoprEP_D?usp=sharing">
  <img src="https://user-images.githubusercontent.com/91646805/156700933-5cc77dba-5df1-40c0-94c8-7459abb6402b.svg" height="18"/></a> to access the shared google drive folder
3) At the top there will be a dropdown arrow after the folder location (Shared with me > data_files): click on this dropdown arrow
4) Click on the "Add shortcut to Drive" button then navigate to inside your droplet_classification folder and click the blue "Add Shortcut" button.  This will add a shortcut to the shared google drive folder in your droplet_classification folder.
5) Open the yolov3 colab notebook from the colab badge provided, then click "Save a copy in Drive" under File > Save a copy in Drive.  Do the same for the provided yolov5 colab notebook.
6) This will save the two notebooks in the "Colab Notebooks" folder in your google drive.  Move these two notebooks to the droplet_classification folder and rename them yolov3.ipynb and yolov5.ipynb respectively in order for the directories to be correct.  The final droplet_classification folder should look like this:<img width="720" alt="image" src="https://user-images.githubusercontent.com/91646805/148874654-890a5d94-f9e9-4273-bcd8-318df44feca4.png">

7) Find the droplet model dataset here: <a href="https://universe.roboflow.com/karl-gardner-kmk9u/pc3dropletdetection2/dataset/8">
  <img src="https://user-images.githubusercontent.com/91646805/156698861-29c0ae55-eff3-4bfe-9dcc-fe06e5a1c6cd.svg" height="18"/></a> and you will see two datasets (No_Augmentation and final_dataset).  Start with the final_dataset and click on "Download" in the upper right corner.  Then, click "Sign in with Github" and follow the prompts to allow roboflow to sign in with github.  Or you may create a different account with roboflow.  Then, the download link will bring you to a pop up that says Export.  For the "Format" click on the YOLO v5 PyTorch and "show download code" on the bottom.  You will then see a link that you can use to enter in the colab notebook.  The final page should look like this but with your own link under the red stripe: <img width="925" alt="image" src="https://user-images.githubusercontent.com/91646805/149068681-5d5529b4-7d6f-41f5-8710-98f04c780654.png"> Then copy this link into the section of both notebooks (yolov3.ipynb and yolov5.ipynb) that says "Curl droplet data from roboflow > Data with Augmentation for Training > [ROBOFLOW-API-KEY]": ![image](https://user-images.githubusercontent.com/91646805/151044698-1d03e6c8-7d2b-401c-b632-b00d1fbe6821.png)  Copy your download link inside of the double quaotations as in the red box in the image provided.

8) Repeat step 7 for the droplet dataset with no augmentations (No_Augmentation): ![image](https://user-images.githubusercontent.com/91646805/151045660-a4fb9e26-a108-4369-aba9-63be2bb9efc1.png)

9) Repeat steps 7 and 8 with the cell dataset <a href="https://universe.roboflow.com/karl-gardner-kmk9u/cropped_drops2/dataset/3">
  <img src="https://user-images.githubusercontent.com/91646805/156698862-6591ba12-a90f-4495-8736-cab83f5cd237.svg" height="18"/></a>
10) You can now use both notebooks to perform more testing or contribute to the project.  You can find the code written for many of the figures in the final paper: https://doi.org/10.1039/D2LC00462C
</details>

<details>
<summary>Testing (click to expand)</summary><br>
Nearly all figures and tables from the paper are outlined in yolov3.ipynb and yolov5.ipynb colab notebooks. For example Table 2 displays the annotation summary for cell and droplet models before augmentations. This can be shown in section 2.1 of the colab notebook:<br><br>
<img src="https://user-images.githubusercontent.com/91646805/186248580-b9d22a03-ee4f-451f-bd5e-19af2ee4fb01.PNG"/></a>
<br><br>
You may run this for example by first uncommenting section 1.1 labeled "Data with No Augmentation (No_Augmentation)":<br><br>
<img src="https://user-images.githubusercontent.com/91646805/186249216-0d38a78b-a25b-436e-919b-e94f19776039.PNG"/></a>
<br><br>
then uncommenting section 2. labeled: "For droplet model". Then the following output will be printed:<br><br>
<img src="https://user-images.githubusercontent.com/91646805/186249418-e3420c46-20d0-4803-bd65-38fe8fb1bea1.PNG"/></a>
<br><br>
The same procedure can be used for the cell model to produce the following result:<br><br>
<img src="https://user-images.githubusercontent.com/91646805/186249529-27c786d9-fb73-47a9-9590-dcc8f5fd277f.PNG"/></a>
<br><br>
This matches Table 2 in the publication:<br><br>
<img src="https://user-images.githubusercontent.com/91646805/186249617-8fddc568-d50a-443b-a4b3-fb9186071308.PNG"/></a>

</details>

<details open>
<summary>Contributions</summary><br>

 **Publication Authors:**<br>Karl Gardner, Md Mezbah Uddin, Linh Tran, Thanh Pham, Siva Vanapalli, and Wei Li<br><br>
 
 **Publication Acknowledgements:**<br>WL acknowledge support from National Science Foundation (CBET, Grant No. 1935792) and National Institute of Health (IMAT, Grant No. 1R21CA240185-01).
</details>

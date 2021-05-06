%******************************************************************************************************************%
% The AIC21 vehicle counting benchmark is captured by 20 cameras in real-world traffic surveillance environment.   %
% A total of 9 hours of videos are split into two sets A (5 hours) and B (4 hours).                                %
% Data set A is made avaible to the participating teams in this package.                                           %
% Data set B is reserved for program testing to determine the final winners.                                       %
% For more details please refer to the AIC Challenge website: https://www.aicitychallenge.org/                     %
%******************************************************************************************************************%


Vehicle Counting Task:
Each of the 20 camera scenes has a predefined region of interest (ROI) and a set of movements of interests (MOIs).
All vehicles appeared in the ROI are eligible to be counted.
If an eligible vehicle belongs to one of the MOI, it should be counted by the time it fully exits the ROI.  
Four-wheel vehicles and freight trucks are counted separately. Two-wheelers (motercycles, bicycles) are not counted.


Content in the directory:
1. "Dataset_A/*.mp4". It contains 5 hours of videos for vehicle counting. 
2. "Dataset_A/list_video_id.txt". It contains the index of all 31 videos alphanumeric order starting with 1. 
3. "Dataset_A/list_class_id.txt". It contains the index of vehicle classes. 
4. "Dataset_A/datasetA_vid_stats.txt". It contains fps and number of frames of each video acquired by ffprobe.
4. "screen_shot_with_roi_and_movement/*.jpg". 20 annotated images one for each camera describing the movement of interests (MOIs) and region of interests (ROIs) for counting. 
5. "ROIs/*.txt". 20 text files one for each camera describing the polygon vertices of the ROIs in pixel coordinates. 
6. "movement_description/*.txt". 20 text files one for each camera describing the MOIs in words. 
7. "counting_gt_sample/counting_example_cam_5_1min.csv". It is the mannually created counting ground truth for the first minute of `Dataset_A/cam_5.mp4` for demonstration purpose. 
8. "counting_gt_sample/counting_example_cam_5_1min.mp4". It is the annotated video showing the counting results created from `counting_example_cam_5_1min.csv` and `cam_5.mp4` for demonstration purpose. Whenever an eligible vehicle exiting the ROI the <movement_id><C or T>-<total counts> is displayed. For example, `11C-2` means it is the second Car of movement 11, `6T-1` means it is the first Truck of movment 6. 


If you have any question, please contact aicitychallenges@gmail.com.

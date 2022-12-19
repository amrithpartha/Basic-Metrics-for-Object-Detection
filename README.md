# Predicting and Evaluating the Object Detection

## About:

This program is to Evaluate the Object Detection with was build with Yolo_V5 on COCO Dataset. The Metrics are used are True Positive, False Positive, True Negative and False Negative which was evaluated based Intersection over Union.

The Intersection over Union is calculated with help of ground Truth and Predticted Output from Yolo_V5

## Implementation:

To run this prject, please run the ./run.sh file in your terminal

Please Change the path according your system directory

To check the True Positive with different threshold, just change the value in the arguement parser(--iou_threshold)

## Output Samples:

<img src = "https://github.com/SP-Amrith-Shrivas/Basic-Metrics-for-Object-Detection/blob/main/img_0.jpg?raw=true" alt="Alt text" title="False Positive">
<img src="https://github.com/SP-Amrith-Shrivas/Basic-Metrics-for-Object-Detection/blob/main/img_2.jpg?raw=true" alt="Alt text" title="False Negative">

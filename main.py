import cv2
import pandas as pd
import os
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import glob


#--------------------------------Problem-------------------------------------
# Create a Python Function that gets two csv files and a RGB image as an input.
# 1st csv - 10 bounding boxes of inference results
# 2nd csv 10 bounding boxes of Ground Truth
#------------------------------------------------------------------------------
# Function should run a 2D-IOU test on all boxes.
# IOU with more that 70% can be considered as TP
#------------------------------------------------------------------------------
# Two Output from Function:
# 1: Pie chart with count of all TP/TN/FP/FN
# 2: Need to get False Positive and False Negative by plotting bbox in 
#    different colour in single image for multiple objects.
#------------------------------------------------------------------------------


#Plotting the Pie chart
def plot_graph(values):
    color = ['Green','Blue','Black','Red']
    labels = ["True Positive","False positive","True Negative","False Negative"]
    
    plt.pie(values,explode=(0,0,0,0),labels=labels,colors=color,autopct='%2.1f%%',shadow=True, startangle=60)
    plt.axis('equal')
    plt.show()
    plt.close()

# Finding the Metrics
def truepositive(iou_list,threshold =0.7):
    tp = 0
    fp = 0
    fn= 0
    print(type(threshold))
    for iou in iou_list:
        if abs(iou) >= abs(threshold):
            tp+=1
        elif abs(iou) > 0 and abs(iou) < abs(threshold):
            fp+=1
        else:
            fn+=1

    #Calculating the Metrics in Percentage
    tp = tp/len(iou_list)*100
    fp = fp/len(iou_list)*100
    fn = fn/len(iou_list)*100
    tn = (100 - (tp+fp+fn))/len(iou_list)*100

    values= [tp,fp,tn,fn]

    #Calling Function to plot the Pie chart
    plot_graph(values)
    print("Perceptage of True Positives is:{}".format(values))
    return tp

# Estimating the Intersection over Union
def predict_iou(gt,pred):

    inf_df_iou=[pred[0],pred[1],pred[0]+pred[2],pred[1]+pred[3]]
    gt_df_iou=[gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]

    # iou_of_image = jaccard_score(gt_df_iou,inf_df_iou,average="micro")

    x1 = max(gt_df_iou[0], inf_df_iou[0])
    y1 = max(gt_df_iou[1], inf_df_iou[1])
    x2 = min(gt_df_iou[2], inf_df_iou[2])
    y2 = min(gt_df_iou[3], inf_df_iou[3])
    
    # Height and width of Intersection
    height = max(0,y2-y1+1)
    width = max(0,x2-x1+1)

    #Overlaped Area
    area = height*width

    # Height and width of ground_truth
    gt_height = gt_df_iou[3] - gt_df_iou[1] + 1
    gt_width = gt_df_iou[2] - gt_df_iou[0] + 1

    # Height and width of prediction
    pred_height = inf_df_iou[3] - inf_df_iou[1] + 1
    pred_width = inf_df_iou[2] - inf_df_iou[0] + 1
    
    # Area Union
    area_tot =  (gt_height * gt_width) + (pred_height *pred_width) -area

    iou_of_image = area/area_tot
    return iou_of_image

#Plotting the bounding box for ground truth and predicted Truth
def bounding_box(img,gt_df,inf_df,predicted):
        c2 = (int(gt_df[0]+gt_df[2]), int(gt_df[1]+gt_df[3]))
        c1 = (int(gt_df[0]), int(gt_df[1]))

        c3 = (int(inf_df[0]+inf_df[2]), int(inf_df[1]+inf_df[3]))
        c4 = (int(inf_df[0]), int(inf_df[1]))
        cv2.rectangle(img, c1, c2, (0, 255, 0), 2)
        cv2.putText(img, 'Ground Truth', (int(gt_df[0]), int(gt_df[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(img, c3,c4, (0, 0, 255), 2)
        cv2.putText(img, 'Predicted', (int(inf_df[0]), int(inf_df[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Plotting the Metrics to display
        cv2.putText(img, "Metrics: {}".format(predicted), (int(inf_df[1]), int(inf_df[2])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)  

#main function
def main(args):
    path = args.pred #import the inference file as argument parser
    label = args.gt #import the ground truth file as argument parser
    
    image_path=  glob.glob("/home/mcw/Documents/AutoBrains_Assignment/Code/images/*.jpg")
    image_list = [cv2.imread(file) for file in image_path]
    #image = cv2.imread(args.images)

    #extracting the values from ground_truth csv file
    with open(label) as file_obj:
        # Create reader object by passing the file 
        # object to reader method
        reader_obj = csv.reader(file_obj)
        # Iterate over each row in the csv 
        # file using reader object
        index1=0
        gt_df1 = []
        for row in reader_obj:
            index1+=1
            if index1 ==1:
                continue
            gt_df1.append(row[4:8]) #4:8
        
        print(gt_df1)

    #extracting the values from prediction csv file
    with open(path) as file_obj:
        # Create reader object by passing the file 
        # object to reader method
        reader_obj = csv.reader(file_obj)
        # Iterate over each row in the csv 
        # file using reader objec
        index2 = 0
        inf_df1= []
        image_name=[]
        for row in reader_obj: 
            index2+=1
            if index2 ==1:
                continue
            inf_df1.append(row[4:8]) #4:8

            # Reading image name from csv file for ordering
            image_name.append(row[0]) #0

    # To append the predicted IoU for all images
    iou_list = []
    values =[]
    
    print(len(gt_df1))
    #To read bounding boxes for all images in inference csv file
    for index in range(len(inf_df1)):
        for index1 in range(len(image_list)):
            if image_name[index] in image_path[index1]:
                image=image_list[index1]
        
        inf_df = inf_df1[index]
        for i in range(len(inf_df)):
            inf_df[i]=float(inf_df[i])
        
        gt_df = gt_df1[index]
        for j in range(len(gt_df1)):
            gt_df[j]=float(gt_df[j])
        
        #predicting the IoU
        iou = predict_iou(gt_df,inf_df)

        #Appending all the IoU results
        iou_list.append(iou)

        if iou >= args.iou_threshold:  
              predicted = "True Positive"
        elif iou < (args.iou_threshold) and iou > 0:
            # compute the intersection over union and display it
            predicted = "False Positive"
        elif iou == 0:
            predicted = "False Negative"
        else:
            predicted ="True Negative"

        #Predicting the bounding boxes
        bounding_box(image,gt_df,inf_df,predicted)
                # compute the intersection over union and display it
        
        cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("The IoU is: {:.4f}".format(iou))
        
        # show the output image
        cv2.imshow("Image", image)
        cv2.imwrite("img_{}.jpg".format(index),image)
        cv2.waitKey(0)

    #Detecting Metrics for IoU
    TP = truepositive(iou_list,args.iou_threshold)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--iou_threshold',type=float,help=' path to load the images from directory',required=False)
    arg.add_argument('--pred',help='path to load the predicted output')
    arg.add_argument('--gt',help='path to load the ground truth')
    args = arg.parse_args()
    main(args)

#!/bin/bash

# Run this command for prediciton and change the threshold according to you need
python main.py --pred /path/Prediction.csv --gt path/Groundtruth.csv --iou_threshold 0.9

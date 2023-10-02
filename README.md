# Overview

Idea of the project was to build a model capable of recognizing type of sport activity
in the video.

## Tech stack
- tensorflow, keras
- cv2
- numpy
- sklearn
- matplotlib

## Process

Backbone of the MobileNet network was used in this project.
At first, I was searching for optimal learning rate and number of layers and neurons at each layer,
then after training some of the first versions of a model I looked at the confusion matrix, to correct the model's errors I decided to increase
number of frames that is passed to a model for each sample. Given low memory capability further increasing of number of frames caused errors.
Instead, during training I chose to yield more samples of those classes, that were misclassified. 

# Results

0 - Archery  
1 - CleanAndJerk  
2 - Diving  
3 - PushUps  
4 - Skiing   

![confusion matrix](https://i.ibb.co/kXpMrT6/confusion-matrices.png)

## Dataset link

[Dataset](https://www.crcv.ucf.edu/data/UCF101.php)
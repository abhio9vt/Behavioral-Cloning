# Behavioral-Cloning

## Project Overview:
In this project deep neural network is used to clone human behavior and drive the car around tracks in a simulator. For training, the car was driven in a simulator and left, right and center camera images were recorded, with the steering angles. This data was then split into training and validation set and used to train the neural network. The model was tested on two tracks and considered successful if the car doesn't leave the driving area.

## Implementation:

### Training data: 

I used the training data provided by Udacity. Note that the data can also be obtained by driving the car in simulator. If we need to generate the data using simulator, the car should be driven around the center of the track for 3-4 laps, and then it should also be driven for recovery, i.e. stop recording and take the car towards the corner and then start recording and bring the car towards the center. Training the car for recovery ensures that if the car leaves the center of the track it knows how to come back towards the center.

The training data contains left, center and right camera images and also the steering angle of the car. Since the left camera detects that the car is towards the left of the lane and the right camera detects vice versa, we need to augment the steering angles for the images captured via left camera and right camera. This is done by adding 0.25 to the steering angles corresponding to the left camera images and subtracting 0.25 from the steering angles corresponding to the right camera images.

The training dataset now contains a total of 24108 images (8036 per camera).

### Preprocessing

(i) Cropping the image: It was observed that the top one-fifth of the image did not have any lanes and just had landscape. Thus its a good idea to crop that portion out. Similarly around 25 pixels at the bottom of the image had the car in it. So we have cropped out those pixels too.

(ii) Image resizing: The images are resized to fit the input dimensions of (64 x 64 x 3) for the NVIDIA CNN model.

(iii) Brightness augmentation: We generated images with different brightness by first converting images to HSV, scaling up or down the V channel and converting back to the RGB channel.

### Network Architecture

The CNN model is based on NVIDIA's End to end learning for self driving cars paper. There is a minor change to the NVIDIA's model, that we have added Max Pooling layers after each convolutional layers. This helped reduce training time and also reduce overfitting. Note that this can also be done by adding dropouts after the fully connected layers. A rather interesting approach to elimiate overfitting in convolutional layers is to implement stochastic pooling. This would be done later.

The model summary is as follows:

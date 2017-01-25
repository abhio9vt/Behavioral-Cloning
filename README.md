# Behavioral-Cloning

## Project Overview:
In this project deep neural network is used to clone human behavior and drive the car around tracks in a simulator. For training, the car was driven in a simulator and left, right and center camera images were recorded, with the steering angles. This data was then split into training and validation set and used to train the neural network. The model was tested on two tracks and considered successful if the car doesn't leave the driving area.

## Implementation:

### Training data collection: 

I used the training data provided by Udacity. Note that the data can also be obtained by driving the car in simulator. If we need to generate the data using simulator, the car should be driven around the center of the track for 3-4 laps, and then it should also be driven for recovery, i.e. stop recording and take the car towards the corner and then start recording and bring the car towards the center. Training the car for recovery ensures that if the car leaves the center of the track it knows how to come back towards the center.

The training data contains left, center and right camera images and also the steering angle of the car. Since the left camera detects that the car is towards the left of the lane and the right camera detects vice versa, we need to augment the steering angles for the images captured via left camera and right camera. This is done by adding 0.25 to the steering angles corresponding to the left camera images and subtracting 0.25 from the steering angles corresponding to the right camera images. Note that the left and right camera images are essential to train the car for recovery, i.e. if the car strays from its path, it can learn how to come back towards the center of the lane.

The training dataset now contains a total of 24108 images (8036 per camera).

### Preprocessing

(i) Cropping the image: It was observed that the top one-fifth of the image did not have any lanes and just had landscape. Thus its a good idea to crop that portion out. Similarly around 25 pixels at the bottom of the image had the car in it. So we have cropped out those pixels too.

(ii) Image resizing: The images are resized to fit the input dimensions of (64 x 64 x 3) for the NVIDIA CNN model.

(iii) Brightness augmentation: We generated images with different brightness by first converting images to HSV, scaling up or down the V channel and converting back to the RGB channel.

(iv) On plotting the steering angles in a histogram, it was noted that the distribution is skewed. This happens as the car is driven in counterclockwise direction in the simulator. To address this issue, we flip the image using openCV function cv2.flip and then take the negative of the steering angle of the original image.

The images at various stages of preprocessing are as follows;
a. Original image (from data provided by Udacity)
![alt-tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/original_img.png)

b. Cropped image - after cropping top 35% and bottom 10% of the image
![alt-tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/cropped_image.png)

c. Flipped image - after performing a vertical flipping using cv2.flip()
![alt-tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/flipped_image.png)

d. Resized image - after resizing the image to the new dimension 64 x 64
![alt-tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/resized_img.png)

e. Histogram of the original steering angles in the data provided nyn Udacity (This lead me to think towards augmenting data and generate more training examples, specially by flipping 50% of the images at random)
![alt-tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/steering_histogram.png)

### Network Architecture

The CNN model is based on NVIDIA's End to end learning for self driving cars paper. There is a minor change to the NVIDIA's model, that we have added Max Pooling layers after each convolutional layers. The max pooling layers help in model regularization. Note thar dropouts overfitting is also reduced by adding dropouts after the fully connected layers. A rather interesting approach to elimiate overfitting in convolutional layers is to implement stochastic pooling. This would be done later.

The model has a normalization layer followed by 5 convolutional layers. Note that each convolutional layer is followed by Max Pooling and an activation funtion- relu. The kernel size for first three convolutional layers is set at 5x5 with a stride of 2x2, while the kernel size for next 2 convolutional layers is 3 x 3 with a stride of 1 x 1. The model is flattened at the end of 5th convolutional layer. The last 5 layers are all fully connected layers, each activated by activation function - relu. The model is compiled using an Adam optimizer with a learning rate of 0.0001. Note that different learning rates from 0.0001 to 0.01 were tested, and a learning rate of 0.0001 was found to give best results.

Details are as follows: Input shape = 64x64x3

1. normalization layer, output shape = 64x64x3

2. convolutional layer, kernel size = 5x5, stride = 2x2, filter_size = 24, padding = valid, activation = relu. This is followed by a max pooling layer with pool_size = 2x2 and stride = 1x1

3. convolutional layer, kernel size = 5x5, stride = 2x2, filter_size = 36, padding = valid, activation = relu. This is followed by a max pooling layer with pool_size = 2x2 and stride = 1x1

4. convolutional layer, kernel size = 5x5, stride = 2x2, filter_size = 48, padding = valid, activation = relu. This is followed by a max pooling layer with pool_size = 2x2 and stride = 1x1

5. convolutional layer, kernel size = 3x3, stride = 1x1, filter_size = 64, output shape = 32x32x24, padding = valid, activation = relu. This is followed by a max pooling layer with pool_size = 2x2 and stride = 1x1

6. convolutional layer, kernel size = 3x3, stride = 1x1, filter_size = 64, output shape = 32x32x24, padding = valid, activation = relu. This is followed by a max pooling layer with pool_size = 2x2 and stride = 1x1

7. After all the convolutional layers 5 fully connected layers are added with 1164, 100, 50, 10 and 1 output neurons respectively. Note that dropouts are added after first and second fully connected layers to reduce overfitting.

Please see the pic for detailed model summary obtained using model.summary().
![alt tag](https://github.com/abhio9vt/Behavioral-Cloning/blob/master/model_summary.png)

### Training Process
After preprocessing the images and augmenting the steering angles, I used fit_generator API of Keras library to train the model. The generator function helps as it generates images on the fly and trains them in batches. We designed 2 generators, one for training data and the other for validation data. Adam optimizer with a learning rate of 0.0001 was found to be working best during our simulation runs. The batch size for both the generators was set at 64, and the number of epochs was 
set at 8 after trying various other #epochs ranging from 5 to 10. It was noticed that after 8 epochs there was very less decrease in validation error.

Note that the generator function is a really good way to train and fit the model, as otherwise we need to store the images and steering angles in an array, which would take a lot of memory.


### Running the model
The data preprocessing, network architecture and training code is in the file model.py. Running model.py outputs model.json and model.h5. These files contain the weights and other CNN model parameters. I have used the same drive.py file as provided by Udacity.
To run the simulator, go to the terminal and cd to the folder where your project is saved. In the terminal enter

python drive.py model.json

## Results
The car is able to navigate on both the test sets correctly!

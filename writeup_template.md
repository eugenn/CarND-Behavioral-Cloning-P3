#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

---

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* util.py The script to provide useful functionalities (i.e. image preprocessing and augumentation)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

##Model Architecture and Training Strategy

###1. An appropriate model architecture has been employed

The design of the network is based on [the NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), which has been used by NVIDIA for the end-to-end self driving test. 
Actually I simplified a bit the original design and have added the following adjustments to the model. 

- I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
- I've added an additional 3 dropout layers  to avoid overfitting the model.
- I've also included RELU for activation function for every layer except for the output layer to introduce non-linearity.

In the end, the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Drop out (0.5)
- Fully connected: neurons:  10, activation: RELU
- Drop out (0.5)
- Fully connected: neurons:   1 (output)

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |Connected to     |
|--------------------------------|------------------|-------:|-----------------|
|lambda_18 (Lambda)              |(None, 66, 200, 3)|0       |lambda_input_18  |
|convolution2d_86 (Convolution2D)|(None, 31, 98, 24)|1824    |lambda_18        |
|convolution2d_87 (Convolution2D)|(None, 14, 47, 36)|21636   |convolution2d_86 |
|convolution2d_88 (Convolution2D)|(None, 5, 22, 48) |43248   |convolution2d_87 |
|convolution2d_89 (Convolution2D)|(None, 3, 20, 64) |27712   |convolution2d_88 |
|convolution2d_90 (Convolution2D)|(None, 1, 18, 64) |36928   |convolution2d_89 |
|flatten_18 (Flatten)            |(None, 1152)      |0       |convolution2d_90 |
|dense_70 (Dense)                |(None, 50)        |57650   |flatten_18       |
|dropout_52 (Dropout             |(None, 50 )       |0       |dense_70         |
|dense_71 (Dense)                |(None, 10)        |510     |dropout_52       |
|dropout_53 (Dropout             |(None, 10)        |0       |dense_71         |
|dense_72 (Dense)                |(None, 1)         |11      |dropout_53       |
|                                |**Total params**  |189,519 |                 |


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. I've convert images from RGB to YUV and randomly flip some images. After that I've modified drive.py file and apply same prepossessing for input images (cropping, resizing and rgb2yuv) 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


####3. Model parameter tuning

The model used an adam optimizer so the learning rate was not tuned manually. After several tests I've choose a small value of batch_size because with that value the car less moved between left and right lines. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


##Model Architecture and Training Strategy

###1. Solution Design Approach

The overall strategy for deriving a model architecture was to build very simple model and see how the model behaves during test simulations. The first attempts showed that the model on the first turn rolls off the road.

My first step was to use a convolution neural network model similar to the NVIDIA.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
I used ModelCheckpoint from Keras to save the model only if the validation loss is improved which is checked for every epoch.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by mean squared error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
The size of final model is about 2.3Mb it is quite compact file.

## Conclusion
I successfully pass the first track but the second track is much more complicated. There are a lot of tricky curves, slopes and many shadows. So maybe need to update the current network design make a bit more complicated, generate more data for tricky cases and increase quantity of epoch.   
## References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
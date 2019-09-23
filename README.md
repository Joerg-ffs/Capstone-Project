# Capstone-Project
Distracted Driving Detection using Python.

## First Prototype
Used an XBOX controller with the game "Track Mania" into a Extremely Randomized Trees (ERT) model to detect distracted driving.

To use this prototype check out the prototype folder.

Demo Video found [here](https://drive.google.com/file/d/1J2EZdBJU70OkKv2cZhLlnJ9tYzt1Xt7s/view)

## Final Prototype
The final prototype consists of two models: a Convolutional Neural Network (CNN) as well as the Extreme Random Forest model similarly to the initial prototype.

## Tier 1
This project uses the Logitech G920 Racing Wheel and acompanying pedals to capture inputs. However, any racing wheel or other form of input can be used as long as the event codes are properly mapped using device_config.py.

To start collecting data, use Inputs_G920.py in the Tier 1 folder. This file will output a folder of images as well as a csv file containg image paths and corrosponding angles.

#### Please note: The prototypes will only function using Ubuntu due to the usage of evdev python library for input polling.

Sample data can be found here: [images](https://drive.google.com/file/d/1ZrqJtPqAVVuZAycNJN9hl2rEKFiiJ80-/view) and [csv](https://drive.google.com/file/d/1996maLE5TBk1y27EInuKVQmH3TUtaa3F/view?usp=sharing)

These inputs are then loaded into Tier_1_Training.ipynb in the Tier 1 folder. Once the notebook is completed, the last output is a model file containing the weights and structure of the CNN. An example is Tier_1_Model.h5.

## Tier 2

The .h5 file is then used in Inputs_G920_Distraction.py in the Tier 2 folder. This will generate a csv file containing actual and predicted angles as well as gas and brake values. An example csv is [here](https://drive.google.com/open?id=1r5CpDfeUh8HH-mqdJl6BhRxLJxrpq-4C)

This csv file is usedin Tier_2_Training.ipynb to create another model file in a .pkl format. An example is [Tier_2_Model.pkl](https://drive.google.com/file/d/13EaXiiwL8QkGVRmzWvlR6II9gzFNARvz/view?usp=sharing)

Both model files are used in Real_Time_Prediction.py to predict distraction levels real time.

Demo video of final prototype found [here](https://drive.google.com/open?id=1Q2nekokL9cWF7Bn4HbtTa8DLEzJBVpIU)

## Data Upgrade

The next level of the models development transitions from using Euro Truck Simulator to real world test data. The data is 300 hours of commuting along California highways and can be found [here](https://github.com/commaai/comma2k19) 

The data was originally in video formatt with 15 frames per second. The data was manipulated from video to jpeg fromatt and adjusted to insure the driving data (steering, brakes, acceleration, etc) was present accross all frames. Outliers in the steering angle data were removed as the focus of the model is highway driving distraction where large changes in steering angle are not relevant. 

In order to combat [Catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference) the totality of the dataset was randomly broken into 10 segements to improve ease of use. Then the datapoints were scrambled to improve overall model performance. A total of 3.2 million frames and data points were used to train the model. 

After adjusting the model weights a final correlation between raw and predicted angles reached 95% providing a very accurate prediction which can then be fed into the distraction model.

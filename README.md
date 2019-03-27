# Capstone-Project
Distracted Driving Detection using Python.

## First Prototype
Used an XBOX controller with the game "Track Mania" into a Extremely Randomized Trees (ERT) model to detect distracted driving.

To use this prototype check out the prototype folder.

Demo Video found [here](https://drive.google.com/file/d/1J2EZdBJU70OkKv2cZhLlnJ9tYzt1Xt7s/view)

## Final Prototype
The final prototype consists of two models: a Convolutional Neural Network (CNN) as well as the ERT model from the initial prototype.

### Tier 1
This project uses the Logitech G920 Racing Wheel to capture inputs. However, any racing wheel can be used as long as the event codes are properly mapped.

To start collecting data, use Inputs_G920.py in the Tier 1 folder. This file will output a folder of images as well as a csv file containg image paths and corrosponding angles.

Sample data can be found here: [images](https://drive.google.com/file/d/1ZrqJtPqAVVuZAycNJN9hl2rEKFiiJ80-/view) and [csv](https://drive.google.com/file/d/1996maLE5TBk1y27EInuKVQmH3TUtaa3F/view?usp=sharing)

These inputs are then loaded into Tier_1_Training.ipynb in the Tier 1 folder. Once the notebook is completed, the last output is a model file containing the weights and structure of the CNN. An example is Capstone.h5.

### Tier 2

The .h5 file is then used in Inputs_G920_Distraction.py in the Tier 2 folder. This will generate a csv file containing actual and predicted angles as well as gas and brake values. An example csv is [here](https://drive.google.com/open?id=1r5CpDfeUh8HH-mqdJl6BhRxLJxrpq-4C)

This csv file is usedin Tier_2_Training.ipynb to create another model file in a .pkl format. An example is Tier_2_Model.pkl.

Both model files are used in Real_Time_Prediction.py to predict distraction levels real time.




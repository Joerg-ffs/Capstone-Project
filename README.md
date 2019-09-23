# Capstone-Project
Distracted Driving Detection using Python.

# First Prototype
Used an XBOX controller with the game "Track Mania" into a Extremely Randomized Trees (ERT) model to detect distracted driving.

To use this prototype check out the prototype folder.

Demo Video found [here](https://drive.google.com/file/d/1J2EZdBJU70OkKv2cZhLlnJ9tYzt1Xt7s/view)

# Final Prototype
The final prototype consists of two models: a Convolutional Neural Network (CNN) as well as the Extreme Random Forest model similarly to the initial prototype.

# Tier 1
This project uses the Logitech G920 Racing Wheel and acompanying pedals to capture inputs. However, any racing wheel or other form of input can be used as long as the event codes are properly mapped using device_config.py.

To start collecting data, use Inputs_G920.py in the Tier 1 folder. This file will output a folder of images as well as a csv file containg image paths and corrosponding angles.

#### Please note: The prototypes will only function using Ubuntu due to the usage of evdev python library for input polling.

Sample data can be found here: [images](https://drive.google.com/file/d/1ZrqJtPqAVVuZAycNJN9hl2rEKFiiJ80-/view) and [csv](https://drive.google.com/file/d/1996maLE5TBk1y27EInuKVQmH3TUtaa3F/view?usp=sharing)

These inputs are then loaded into Tier_1_Training.ipynb in the Tier 1 folder. Once the notebook is completed, the last output is a model file containing the weights and structure of the CNN. An example is Tier_1_Model.h5.

# Tier 2

The .h5 file is then used in Inputs_G920_Distraction.py in the Tier 2 folder. This will generate a csv file containing actual and predicted angles as well as gas and brake values. An example csv is [here](https://drive.google.com/open?id=1r5CpDfeUh8HH-mqdJl6BhRxLJxrpq-4C)

This csv file is usedin Tier_2_Training.ipynb to create another model file in a .pkl format. An example is [Tier_2_Model.pkl](https://drive.google.com/file/d/13EaXiiwL8QkGVRmzWvlR6II9gzFNARvz/view?usp=sharing)

Both model files are used in Real_Time_Prediction.py to predict distraction levels real time.

Demo video of final prototype found [here](https://drive.google.com/open?id=1Q2nekokL9cWF7Bn4HbtTa8DLEzJBVpIU)

# Data Upgrade

The next level of the models development transitions from using Euro Truck Simulator to real world test data. The data is 300 hours of commuting along California highways and can be found [here](https://github.com/commaai/comma2k19) 

The data was originally in video formatt with 15 frames per second. The data was manipulated from video to jpeg fromatt and adjusted to insure the driving data (steering, brakes, acceleration, etc) was present accross all frames. Outliers in the steering angle data were removed as the focus of the model is highway driving distraction where large changes in steering angle are not relevant. 
Sample data can be found [here](https://github.com/Joerg-ffs/Capstone-Project/tree/master/Data%20Upgrade/sample%20data)

In order to combat [Catastrophic interference](https://en.wikipedia.org/wiki/Catastrophic_interference) the totality of the dataset was randomly broken into 10 segements to improve ease of use. Then the datapoints were scrambled to improve overall model performance. A total of 3.2 million frames and data points were used to train the model. 

After adjusting the model weights a final correlation between raw and predicted angles reached 95% providing a very accurate prediction which can then be fed into the distraction model.

# Training Methods

## Prototype:

The first prototype of the model simply used the python library sk-learn, the following models were tested:
![Model Performance](https://github.com/Joerg-ffs/Capstone-Project/blob/master/Prototype/model%20performance.PNG)

Due to the simple nature and limited amount of data the was trained on a MSI GE72 2QD Apache Pro using the i7-4700HQ CPU for full specs click [here](https://www.msi.com/Laptop/GE72-2QD-Apache-Pro)

## Tier 2:

With the addition of image data and the implimentation of [CNN](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) training the model on a laptop would not be feasible. Due to limited resources the free tool  [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true), which initalizes a jupyter notebook running on a NVIDIA TESLA K80 GPU was used. This tool was very helpful in the testing of the model as it also allows colaboration between team members however it also had frequent crashes and other issues.
![Tier 2 Performance](https://github.com/Joerg-ffs/Capstone-Project/blob/master/Final%20Prototype/Tier%202/model%20performance%202.PNG)

## Updated Model:

Moving from a smaller dataset using cloud tools to a 3.2 million image dataset needed a drastic change to our training workflow. A new system was purchased to locally train the model, initally the model was trained using basic tensorflow using the keras API on an [Intel Core i7-9700K CPU @ 3.60GHz](https://ark.intel.com/content/www/us/en/ark/products/186604/intel-core-i7-9700k-processor-12m-cache-up-to-4-90-ghz.html). However training the model using a CPU was very time extensive, after further research into tensorflow-gpu and Nvidia CUDA drivers this method was implimented. 

Please note setting up a non-linux based tensorflow GPU enviroment in this case Windows 10 is a very convoluted process proceed with caution.

Once the tensorflow GPU process was configured properly we were able to run batches of 75-100k images at a time dratically improving our training workflow. The GPU used was the [Nvidia Geforce RTX 2070](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2070/). 

Without any further data manipulation the correlation for ~100,000 samples is 89.7% when comparing prediction and actual steering angles. When looking at a 15 frame rolling average the correlation goes to to 95%.

The results can be seen below:
![Model 3](https://github.com/Joerg-ffs/Capstone-Project/blob/master/Data%20Upgrade/Model%203%20graph.png)

# Next Steps

There are three critical steps in the development of this project, firstly updating the distraction model, secondly implimenting  physical data extraction from a car, and finally creating a dashboard for visualization of the data.

## Distraction model update:
Going forward an updated version of the tier 2 model needs to be developed, this poses a difficult challenge because acquiring real world distracted driving in an enthical manner is difficult. The best course of action seems to be simulating the distraction data using our past knowledge from the previous models and then iterating on the model until real world outputs are accurate. 

## Implimenting physical data extraction:
At this point the plan of action is to utilze the [panda OBD2 interface](https://comma.ai/shop/products/panda-obd-ii-dongle) which is a state of the art OBD2 extraction tool that can live stream data via USB or wifi at rates much higher then industry standards. From the OBD2 port we will use a [Raspberry Pi 4](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) which is an inexpensive micro computer that can run both the tier 1 and 2 models. Finally a forward facing [Pi Camera](https://www.raspberrypi.org/products/camera-module-v2/) will live stream the image data to the Pi 4. 

## Dashboard

In order to analize the data a dashboard will be created. At this point an minimal viable product using [Tableau](https://www.tableau.com/) will be made, in future a webapp using the python API [dash](https://dash.plot.ly/) will be implimented.


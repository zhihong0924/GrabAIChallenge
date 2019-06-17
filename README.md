# GrabAIChallenge
# Overview
This project is designed for Grab AI Challenge to detect the make, model and year of a car.
Dataset for training can be obtained from here: https://ai.stanford.edu/~jkrause/cars/car_dataset.html

This program is successfully executed by Python 3.6.7 and Python 3.7.3.

# How to run this program
1.Change directory to the working directory.
2. pip install -r requirements.txt
3. Run python MainProgram.py from console. (Don't run from idle)
4. Open browser and go to http://localhost:5000/
5. You may use the web UI to test the model.

# How to use the web UI
1. User may select multiple images for detection.
2. Id and class name will be displayed on the UI.
3. A text file contains the detected id can be found in result folder of main directory. (It would be useful in case of model evaluating)
4. User may refer to labelmap.pbtxt in inference_graph folder to visualize the class id and name respectively.



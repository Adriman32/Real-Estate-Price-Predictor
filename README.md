# Real Estate Price Predictor
Predicting the Price Per Square Area of houses using Multi-Layer Perceptron Artificial Neural Networks

### Description
- Custom methods for easily creating and training artificial neural networks using SKLearn MLP Regressor
- Custom methods for creating train/test datasets given parameters using Pandas DataFrames
- Given an array of nodes and maximum number of hidden layers, every possible combination of networks will be created, trained and tested
- Model with least Mean Square Error loss will be selected and saved
- Plots showing losses over time displayed

# How To Use
In your console, navigate to desired directory and enter the following
- git clone https://github.com/Adriman32/Real-Estate-Price-Predictor.git
- cd Real-Estate-Predictor
- pip install -r requirements.txt

# Requirements
- Python
- requirements.txt (included)



Running 'main.py' will begin the training of networks with the given data. The default settings for the training are hidden_nodes = [50,128,256,512], and max_hidden_layer_size = 3.
The *createLayers* function will create every combination of layers given the inputs. The number of layers that will be created is equal to `x^1 + x^2 +...+ x^n` where `x = len(hidden_nodes)` and `n = max_hidden_layer_size`
For the default example, this would result in 84 networks. These are then trained and tested, and the model with best performance will be saved.

### Pre-Trained Models
The `models` folder included in the repository contains several different models of varying hidden layer sizes and MSE loss. Their filenames correspond to their layer size and MSE values. Use the included *runModel.py* program to test the different models. Below are several plots achieved from the included networks.

![Figure 0](/plots/Figure_0.png)

![Figure 1](/plots/Figure_1.png)





Using Real Estate dataset from https://www.kaggle.com/alexanderliapatis/nn-regression-real-estate


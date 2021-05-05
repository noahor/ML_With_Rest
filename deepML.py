from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt  
import math
import keras
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model 



NUM_NAME ="1 with 5000 nodes"

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='C:\\Users\\oron.noah\\Documents\\MY_ML\\logs\\{}'.format(NUM_NAME),
    write_graph=True,
    histogram_freq=5
)


model = Sequential()
model.add(Dense(500,input_dim=9,activation ='relu'))

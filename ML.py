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
from keras.models import load_model
from keras.layers import *
from Model import *
import json


ai_Model = Ai_Model()


# Train the Model 
def Train():
  
    global ai_Model
    training_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_training.csv") 
    testing_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_test.csv") 


    X = training_data_df.drop('total_earnings',axis=1).values
    Y = training_data_df[['total_earnings']].values

     
    X_test = testing_data_df.drop('total_earnings',axis=1).values
    Y_test = testing_data_df[['total_earnings']].values

    
    ai_Model.build()
    ai_Model.fit(X,Y)
    test_error_rate =ai_Model.evaluate(X_test,Y_test)
    
    path = "C:\\Users\\oron.noah\\Documents\\MY_ML\\MY_ML.h5"
    #model.save(path)
    #current_Model = model
    result = dict();  
    result['path'] = path
    result['test_error_rate']   = test_error_rate
    return result

def save(path):
    global ai_Model
    ai_Model.save_to_path(path)

def predict ():
    global ai_Model
    x_new_data_to_prediect = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\04\\proposed_new_product.csv")
    prediection = ai_Model.predict(x_new_data_to_prediect)
    return prediection[0][0]


def load_model_from_path (model_path):
    global ai_Model
    new_model = load_model(model_path)
    
    with open(model_path+'scaler_Y_Info.json') as json_file:
        scalar_data = json.load(json_file)
    ai_Model = Ai_Model(new_model,MinMaxScaler(feature_range=(0,1)),MinMaxScaler(feature_range=(0,1)),scalar_data)

    

from sklearn.preprocessing import MinMaxScaler

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

# Load training data set from CSV file
training_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_training.csv") 

# Load testing data set from CSV file
test_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_test.csv") 

# Data needs to be scaled to a small range like 0 to 1 for the neural
# network to work well.
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale both the training inputs and outputs
scaled_training = scaler.fit_transform(training_data_df)
scaled_testing = scaler.transform(test_data_df)

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='C:\\Users\\oron.noah\\Documents\\MY_ML\\logs\\{}'.format(NUM_NAME),
    write_graph=True,
    histogram_freq=5
)


print("Note: total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}".format(scaler.scale_[9], scaler.min_[9]))
scale_factor = scaler.scale_[8]
scale_min = scaler.min_[8]
# Create new pandas DataFrame objects from the scaled data
scaled_training_df = pd.DataFrame(scaled_training, columns=training_data_df.columns.values)
scaled_testing_df = pd.DataFrame(scaled_testing, columns=test_data_df.columns.values)

# Save scaled data dataframes to new CSV files
scaled_training_df.to_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_training_scaled.csv", index=False)
scaled_testing_df.to_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_testing_scaled.csv", index=False)

training_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_training_scaled.csv")
test_data_df = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\03\\sales_data_testing_scaled.csv")
X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

# Define the model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X,Y,epochs=50,shuffle=True,verbose=2,callbacks=[logger])


X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

test_error_rate = model.evaluate(X_test, Y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# Load the data we make to use to make a prediction
X = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\04\\proposed_new_product - Copy.csv").values
X1 = pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\04\\proposed_new_product _total_earnings.csv").values
X1_df =pd.read_csv("C:\\Users\\oron.noah\\Documents\\ML2\\Ex_Files_Building_Deep_Learning_Apps\\Exercise Files\\04\\proposed_new_product _total_earnings.csv")
scaled_testing = scaler.transform(X1)

scaled_X1_pd = pd.DataFrame(scaled_testing, columns=X1_df.columns.values)
scaled_X1 = scaled_X1_pd.drop('total_earnings', axis=1)
# Make a prediction with the neural network
prediction = model.predict(X)
prediction_new = model.predict(scaled_X1)
# Grab just the first element of the first prediction (since that's the only have one)
prediction = prediction[0][0]
prediction_new = prediction_new[0][0]
# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally scaled down to the 0-to-1 range

print(scale_min)

prediction = prediction - scale_min
prediction = prediction / scale_factor
print("Earnings Prediction for Proposed Product - ${}".format(prediction))

prediction_new = prediction_new - scale_min
prediction_new =prediction_new / scale_factor
print("Earnings Prediction for Proposed Product - ${}".format(prediction_new))

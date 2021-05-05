from keras.models import Sequential
import json
from keras.layers import *

from sklearn.preprocessing import MinMaxScaler

class Ai_Model:
    
    def __init__(self, current_Model = Sequential(),scaler_X = MinMaxScaler(feature_range=(0,1)),scaler_Y = MinMaxScaler(feature_range=(0,1)),scaler_Y_Info={}):
        self.current_Model = current_Model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.scaler_Y_Info = scaler_Y_Info


    def get_scaler_X(self):
        return self.scaler_X 
    def get_scaler_Y(self):
        return self.scaler_Y

    def build(self):

        self.current_Model.add(Dense(50,input_dim=9,activation ='relu'))
        self.current_Model.add(Dense(100,activation ='relu'))
        self.current_Model.add(Dense(50,activation ='relu'))
        self.current_Model.add(Dense(1))
        self.current_Model.compile(loss='mean_squared_error',optimizer='adam')


    def fit(self,X,Y):

        X_scale  = self.scaler_X.fit_transform(X)
        Y_scale  = self.scaler_Y.fit_transform(Y)
        self.scaler_Y_Info = {
            'scale_value': self.scaler_Y.scale_[0],
            'scale_min': self.scaler_Y.min_[0]
        }
      

       
        self.current_Model.fit(X_scale,Y_scale,epochs=50,shuffle=True,verbose=2,batch_size=10)

    def evaluate(self,X_test,Y_test):

        X_test_scale  = self.scaler_X.fit_transform(X_test)
        Y_test_scale  = self.scaler_Y.fit_transform(Y_test)
        evaluation = self.current_Model.evaluate(X_test_scale, Y_test_scale, verbose=0)
        return evaluation

    def save_to_path(self,path):
        self.current_Model.save(path)
       

        with open(path+'scaler_Y_Info.json', 'w') as json_file:
            json.dump(self.scaler_Y_Info, json_file)
        print("save scalr to json")
               
    

 

    def predict(self,x_new_data_to_prediect):

        scale_prediect = self.scaler_X.fit_transform(x_new_data_to_prediect)
        prediection = self.current_Model.predict(scale_prediect)

        print("Scalar min is- ${}".format(self.scaler_Y_Info['scale_min']))
        print("Scalar scale_value is- ${}".format(self.scaler_Y_Info['scale_value']))
        prediction_new = prediection - self.scaler_Y_Info['scale_min']
        prediction_new = prediction_new / self.scaler_Y_Info['scale_value']
        print("Earnings new Prediction for Proposed Product - ${}".format(prediction_new))

        return  prediction_new






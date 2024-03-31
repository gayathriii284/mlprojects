import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


from logger import logging
from exception import CustomException
from utils import save_object,evaluate_model

#Import all the required models
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor



@dataclass
class ModelTrainingConfig:
    training_model_file_path = os.path.join("artifacts","model.pkl")
    
class ModelTraining:
    def __init__(self):
        #Store the file path in the model_training_config attribute
        self.model_training_config = ModelTrainingConfig()
        
    def initiate_model_training(self,train_arr,test_arr):
        """
        Initiate Model Training
        """
        try:
            logging.info("Initiate Model Training")
            
            #get x_train and x_test from data transformation
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            #Create a dictionary of all the models that cam be tried and trainied with
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XGBoost Regressor": XGBRFRegressor(),
                "Linear Regression" : LinearRegression(),
                "KNeigbours Regressor": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boost Regressor": GradientBoostingRegressor()
                }
            
            #Call evaluate model from utils
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            logging.info(f"Model Report {model_report}")
            
            #Get the best model score
            best_model_score = max(model_report.values())
            
            logging.info(f"Best Score:{best_model_score}")
            
            #Convert to lists since dictionaries are unordered and cannot be indexed through positions
            listModelReport = list(model_report)
            listModelReportThroughValues = list(model_report.values())
            
            #To get the best model name 
            best_model_name = listModelReport[listModelReportThroughValues.index(best_model_score)]
            best_model = models[best_model_name]
            
            #Deleting the list variables
            del listModelReport,listModelReportThroughValues
                        
            logging.info(f"The best model is {best_model_name} with a score of {best_model_score}")
            
            #Raise an exception if in case the r2 score is not beyond 0.6
            if best_model_score<0.6:
                raise CustomException("The accuracy is too low and thus no best model found!",sys)
            
            logging.info("Created the best model. Saving it")
            
            save_object(
                file_path = self.model_training_config.training_model_file_path,
                obj = best_model
            )
            
            logging.info("Saved best model succesfully")
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #Creating path for the pickle file i.e the model
    preprocess_obj_file_path=os.path.join('artifacts','preprocess_obj.pkl')

class DataTransformation:
    """
    Takes the raw dataset and transforms into a format that is ingestible for the model
    """
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        logging.info("Entered the Data Transformation function")

        try:
            numerical_features = ["reading_score","writing_score"]
            categorical_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            logging.info("Categorical features are:",categorical_features)
            logging.info("Numerical features are:",numerical_features)

            #Creating pipelines for numerical and categorical features
            numerical_pipeline=Pipeline(
                steps= [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )

            logging.info("Numerical pipeline created")

            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler())
                ]
            )

            logging.info("Categorical pipeline created")

            #Combining both the pipelines
            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline",numerical_pipeline,numerical_features),
                    ("categorical_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            logging.info("Combined both the pipelines")

            #Return the columntransformer
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        """
        Applying transformations to the train and test datasets
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading from train and test path completed")

            preprocessor_obj = self.get_data_transformer_object()

            logging.info("Getting the data transformer object")

            #Creating the target column
            target_column = ["math_score"]

            #TRAINING DATAFRAME
            #Removing the target label from the training dataset
            X_train_df = train_df.drop(columns=[target_column],axis=1)
            #Target
            y_train_df= train_df[target_column]

            #TEST DATAFRAME
            #Applying the same for test dataframe
            X_test_df = test_df.drop(columns=[target_column],axis=1)
            #Target
            y_test_df = test_df[target_column]

            logging.info("Applying preprocessing  on the training and testing sets")

            X_train_df_arr  = preprocessor_obj.fit_transform(X_train_df)
            X_test_df_arr = preprocessor_obj.transform(X_test_df)
            
            #Creating the train and the test array
            train_arr = np.c_[
                X_train_df_arr,np.array(y_train_df)
            ]

            test_arr = np.c_[
                X_test_df_arr,np.array(y_test_df)
            ]

            logging.info("Created the respective train test arrays")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path
                obj=preprocessor_obj
            )

            logging.info("Successfully saved the object")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path
            )


import sys
import os

sys.path.append('..')
from logger import logging
from  exception import CustomException
import pandas as pd
from data_transformation import DataTransformation
from model_training import  ModelTraining

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    #Input Data
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        #All three file paths will be stored in this variable
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiate Data Ingestion")
        try:
            #Read the dataset (at present consider directly reading from the dataset)
            df=pd.read_csv("C:\\Users\\user\\Documents\\MLProjects\\notebook\\data\\stud.csv")

            #Create a directory for the training data path
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #Take the raw data and convert it into csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Initiating train-test-split")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            #Storing train and test data in their respective artifacts
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Created the required train,test,raw files successfully")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_df,test_df=obj.initiate_data_ingestion() 

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_df,test_df)
    
    #Model Training
    model_training = ModelTraining()
    model_training.initiate_model_training(train_arr,test_arr)
import os
import sys
import dill

import pandas as pd
import numpy as np
from src.exception import CustomException

def save_object(file_path,obj):
    """
    Function to save the processor object as a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)

        #Create directory
        os.makedirs(dir_path,exist_ok=True)

        #open and read the file
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException

def save_object(file_path, obj):
    """
    This function saves the object to a file using pickle.
    """
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            # Using dill to save the object
            # dill is used instead of pickle for better compatibility with complex objects
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e
import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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
    



def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    """
    This function evaluates the model and returns the model report.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Train the model
            # model.fit(X_train, y_train) 
            
            
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys) from e
    



def load_object(file_path):
    """
    This function loads the object from a file using dill.
    """
    
    try:
        with open(file_path, 'rb') as file_obj:
            # Using dill to load the object
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import classification_report, log_loss, f1_score

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_report = classification_report(y_train, y_train_pred)
            test_model_report = classification_report(y_test, y_test_pred)
            train_model_loss = log_loss(y_train, y_train_pred)
            test_model_loss = log_loss(y_test, y_test_pred)
            train_model_score = f1_score(y_train, y_train_pred)
            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)

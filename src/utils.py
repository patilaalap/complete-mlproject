import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import classification_report, log_loss, f1_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            # gs = GridSearch(model, para, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
            gs = GridSearchCV(model, para, cv=5)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            # train_model_report = classification_report(y_train, y_train_pred)
            # test_model_report = classification_report(y_test, y_test_pred)
            # train_model_loss = log_loss(y_train, y_train_pred)
            # test_model_loss = log_loss(y_test, y_test_pred)
            # train_model_score = f1_score(y_train, y_train_pred)
            test_model_score = f1_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import sklearn.metrics as met
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


class Projekt:

    def dataset_generator(self):
        self.adult_train = pd.read_csv("Adult_train.tab", '\t')
        self.adult_test = pd.read_csv("Adult_test_without_class.tab", '\t')


    def dataset_preprocesin(self):
        self.adult_train["y"] = self.adult_train.y.astype("category")
        self.adult_train["workclass_cat"] = pd.Categorical(self.adult_train.workclass).codes
        self.adult_train["education_cat"] = pd.Categorical(self.adult_train.education).codes
        self.adult_train["marital_status_cat"] = pd.Categorical(self.adult_train.marital_status).codes
        self.adult_train["occupation_cat"] = pd.Categorical(self.adult_train.occupation).codes
        self.adult_train["relationship_cat"] = pd.Categorical(self.adult_train.relationship).codes
        self.adult_train["race_cat"] = pd.Categorical(self.adult_train.race).codes
        self.adult_train["sex_cat"] = pd.Categorical(self.adult_train.sex).codes
        self.adult_train["native_country_cat"] = pd.Categorical(self.adult_train.native_country).codes
        self.adult_train["y_cat"] = pd.Categorical(self.adult_train.y).codes
        columns = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week', 'workclass_cat',
                   'education_cat', 'marital_status_cat', 'occupation_cat', 'relationship_cat', 'race_cat', 'sex_cat',
                   'native_country_cat']
        x_data = self.adult_train.loc[:, columns]
        y_data = self.adult_train.loc[:, ["y_cat"]]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data, y_data, test_size=0.2)

    def training_process(self):
        methods = [RandomForestClassifier(n_estimators=100, min_samples_split=5), DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3)]

        res_train = []
        res_train_f = []
        res_test_f = []
        res_test = []

        j = 1
        kk = 1

        for method in methods:

            rf_clas = method.fit(self.x_train, self.y_train)
            y_pred_train = rf_clas.predict(self.x_train)
            y_pred_test = rf_clas.predict(self.adult_test)


            # met.accuracy_score(self.y_train, y_pred_train)
            # print(met.accuracy_score(self.y_train, y_pred_train))

            # metrics_train["accuracy"] = met.accuracy_score(self.y_train, y_pred_train)
            # metrics_train["precision"] = met.precision_score(self.y_train, y_pred_train)
            # metrics_train["recall"] = met.recall_score(self.y_train, y_pred_train)
            # metrics_train["f1"] = met.f1_score(self.y_train, y_pred_train)
            # metrics_train["roc_auc"] = met.roc_auc_score(self.y_train, y_pred_train)
            #
            # metrics_test["accuracy"] = met.accuracy_score(self.y_test, y_pred_test)
            # metrics_test["precision"] = met.precision_score(self.y_test, y_pred_test)
            # metrics_test["recall"] = met.recall_score(self.y_test, y_pred_test)
            # metrics_test["f1"] = met.f1_score(self.y_test, y_pred_test)
            # metrics_test["roc_auc"] = met.roc_auc_score(self.y_test, y_pred_test)

            if j == 1:
                res_train_f = y_pred_train
            else:
                for index, v in enumerate(res_train):
                    res_train_f[index] = res_train_f[index] + y_pred_train[index]
            j = j+1


            if kk == 1:
                res_test_f = y_pred_test
            else:
                for index, v in enumerate(res_test):
                    res_test_f[index] = res_test_f[index] + y_pred_test[index]
            kk = kk+1



projekt = Projekt()
projekt.dataset_generator()
projekt.dataset_preprocesin()
projekt.training_process()
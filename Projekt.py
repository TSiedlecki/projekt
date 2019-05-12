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
        self.adult_test = pd.read_csv("Adults_test_without_class.tab", '\t')


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

        self.adult_test["workclass_cat"] = pd.Categorical(self.adult_test.workclass).codes
        self.adult_test["education_cat"] = pd.Categorical(self.adult_test.education).codes
        self.adult_test["marital_status_cat"] = pd.Categorical(self.adult_test.marital_status).codes
        self.adult_test["occupation_cat"] = pd.Categorical(self.adult_test.occupation).codes
        self.adult_test["relationship_cat"] = pd.Categorical(self.adult_test.relationship).codes
        self.adult_test["race_cat"] = pd.Categorical(self.adult_test.race).codes
        self.adult_test["sex_cat"] = pd.Categorical(self.adult_test.sex).codes
        self.adult_test["native_country_cat"] = pd.Categorical(self.adult_test.native_country).codes
        self.x_data_test = self.adult_test.loc[:, columns]



    def training_process(self):
        methods = [RandomForestClassifier(n_estimators=100, min_samples_split=5), DecisionTreeClassifier(), KNeighborsClassifier(n_neighbors=3)]

        res_train = []
        res_train_f = []
        self.res_test_f = []
        res_test = []

        j = 1
        kk = 1

        for method in methods:

            rf_clas = method.fit(self.x_train, self.y_train)
            y_pred_train = rf_clas.predict(self.x_train)
            y_pred_test = rf_clas.predict(self.x_data_test)


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
                self.res_test_f = y_pred_test
            else:
                for index, v in enumerate(res_test):
                    self.res_test_f[index] = res_test_f[index] + y_pred_test[index]
            kk = kk+1

    def output_data(self):
        print(len(self.res_test_f))
        xzy="["
        for item in self.res_test_f:
            xzy+=str(item)+","
        xzy+="]"
        with open('result.txt', 'w') as f:

                f.write(xzy)




projekt = Projekt()
projekt.dataset_generator()
projekt.dataset_preprocesin()
projekt.training_process()
projekt.output_data()
import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
import matplotlib as mpl
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import common
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = "DATOS-ACTIVIDAD.xlsx"


def random_forest(df):
    target = df["REALIZADA"]
    df = df.drop(["REALIZADA"], axis=1)

    x, y = df.values, target.values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=1)
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    TP, FN, FP, TN = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')


def decision_tree(df):
    target = df["REALIZADA"]
    df = df.drop(["REALIZADA"], axis=1)

    x, y = df.values, target.values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=1)
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    TP, FN, FP, TN = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')


def preprocessing():
    actividad = pd.read_excel(path_to_actividad, sheet_name='ACTIVIDAD')

    df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "PREST", "ESTADO",
                                                            "DEMORA", "TVISITA", "PROCEDENCIA", "REALIZADA"])

    columns_to_expand = ["ESTADO", "TVISITA", "DEMORA", "PREST", "SERVICIO"]
    count = 0
    for idx_actividad, cod_actividad in enumerate(actividad["CODIGO"]):
        df.loc[idx_actividad]["CODIGO"] = actividad.iloc[idx_actividad]["CODIGO"]

        df.loc[idx_actividad]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
        df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]
        df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        df.loc[idx_actividad]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
        df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
        if actividad.iloc[idx_actividad]["ESTADO"] == "PRG":
            df.loc[idx_actividad]["ESTADO"] = ""
        if actividad.iloc[idx_actividad]["REALIZADA"] == "U":
            df.loc[idx_actividad]["REALIZADA"] = ""
        elif actividad.iloc[idx_actividad]["REALIZADA"] == "S":
            df.loc[idx_actividad]["REALIZADA"] = 1
        elif actividad.iloc[idx_actividad]["REALIZADA"] == "N":
            df.loc[idx_actividad]["REALIZADA"] = 0
        count = count + 1
        # if count == 5000:
        #    break

    df = df.dropna(subset=["CODIGO"])
    df = df.dropna(subset=["SERVICIO"])
    df = df.dropna(subset=["PREST"])
    df = df.dropna(subset=["ESTADO"])
    df = df.dropna(subset=["DEMORA"])
    df = df.dropna(subset=["TVISITA"])
    df = df.dropna(subset=["PROCEDENCIA"])
    df = df.dropna(subset=["REALIZADA"])
    df = pd.get_dummies(df, columns=columns_to_expand)
    df = pd.DataFrame.drop(df, columns=["CODIGO"])

    return df


if __name__ == '__main__':
    dataframe = preprocessing()
    random_forest(dataframe)
    decision_tree(dataframe)
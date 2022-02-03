import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from datetime import datetime
from sklearn.impute import SimpleImputer
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = "DATOS-ACTIVIDAD.xlsx"


def random_forest(df):
    target = df["REALIZADA"]
    df = df.drop(["REALIZADA"], axis=1)
    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer = median_imputer.fit(df)
    imputed_df = median_imputer.transform(df.values)
    df = pd.DataFrame(data=imputed_df, columns=df.columns)"""
    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, stratify=y, test_size=0.2, random_state=1)
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
    feature_importances_df = pd.DataFrame(
        {"feature": list(df.columns), "importance": classifier.feature_importances_}).sort_values("importance",
                                                                                                 ascending=False)
    sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="large")
    plt.show()


def decision_tree(df):
    target = df["REALIZADA"]
    df = df.drop(["REALIZADA"], axis=1)
    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer = median_imputer.fit(df)
    imputed_df = median_imputer.transform(df.values)
    df = pd.DataFrame(data=imputed_df, columns=df.columns)"""
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

    df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "DELTA_DIAS", "ANTERIOR",
                                                            "TVISITA", "TURNO", "REALIZADA"])

    """df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "PREST", "ESTADO",
                                                            "DEMORA", "TVISITA", "PROCEDENCIA", "REALIZADA"])"""

    columns_to_expand = ["TVISITA", "SERVICIO", "ANTERIOR"]
    count = 0
    actividad = actividad.drop_duplicates('CODIGO')

    for idx_actividad, cod_actividad in enumerate(actividad["CODIGO"]):
        df.loc[idx_actividad]["CODIGO"] = actividad.iloc[idx_actividad]["CODIGO"]
        df.loc[idx_actividad]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
        df.loc[idx_actividad]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
        #df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        #df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        #df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
        df.loc[idx_actividad]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] - actividad.iloc[idx_actividad]
                                              ["FECHAGRABACION"]).total_seconds() / (60 * 60 * 24)
        time = datetime.strptime("15:00:00", '%H:%M:%S')
        if len(actividad.iloc[idx_actividad]["HORAINI"].split(":")) == 2:
            appointment_time_formatted = actividad.iloc[idx_actividad]["HORAINI"] + ":00"
        else:
            appointment_time_formatted = actividad.iloc[idx_actividad]["HORAINI"]
        appointment_time = datetime.strptime(appointment_time_formatted, '%H:%M:%S')
        if appointment_time >= time:
            df.loc[idx_actividad]["TURNO"] = 1
        else:
            df.loc[idx_actividad]["TURNO"] = 0

        """if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
            df.loc[idx_actividad]["ESTADO"] = None
        else:
            df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]"""

        if actividad.iloc[idx_actividad]["REALIZADA"] == "U":
            df.loc[idx_actividad]["REALIZADA"] = None
        else:
            df.loc[idx_actividad]["REALIZADA"] = actividad.iloc[idx_actividad]["REALIZADA"]
        """elif actividad.iloc[idx_actividad]["REALIZADA"] == "S" and actividad.iloc[idx_actividad]["ESTADO"] == "CAP" \
                and count != 40000:
            df.loc[idx_actividad]["REALIZADA"] = "N"
            count = count + 1"""

        if actividad.iloc[idx_actividad]["ANTERIOR"] is None or actividad.iloc[idx_actividad]["ANTERIOR"] != 0 \
                or actividad.iloc[idx_actividad]["ANTERIOR"] == "":
            df.loc[idx_actividad]["ANTERIOR"] = "S"
        else:
            df.loc[idx_actividad]["ANTERIOR"] = "N"

        #if count == 5000:
        #    break

    df = df.dropna(subset=["CODIGO"])
    df = df.dropna(subset=["SERVICIO"])
    df = df.dropna(subset=["TVISITA"])
    #df = df.dropna(subset=["PROCEDENCIA"])
    df = df.dropna(subset=["TURNO"])
    df = df.dropna(subset=["DELTA_DIAS"])
    df = df.dropna(subset=["ANTERIOR"])
    df = df.dropna(subset=["REALIZADA"])

    df = pd.get_dummies(df, columns=columns_to_expand)
    df = pd.DataFrame.drop(df, columns=["CODIGO"])

    return df


if __name__ == '__main__':
    dataframe = preprocessing()
    random_forest(dataframe)
    decision_tree(dataframe)
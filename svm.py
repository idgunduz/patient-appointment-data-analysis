import matplotlib as mpl
import numpy as np

from datetime import datetime
from imblearn.under_sampling import NearMiss
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from common import *
from sklearn import metrics
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = "datos_falsos/DATOS-ACTIVIDAD.xlsx"


def svm(df):
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)

    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, stratify=y_under, test_size=0.3, random_state=1)
    svc_model = SVC()
    svc_model.fit(x_train, y_train)
    y_pred = svc_model.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    TP, FN, FP, TN = confusion_matrix(y_test, y_pred, labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')


def preprocessing():
    actividad = pd.read_excel(path_to_actividad, sheet_name='ACTIVIDAD')

    df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "ESTADO", "DELTA_DIAS",
                                                            "TVISITA", "PROCEDENCIA", "TURNO", "REALIZADA"])

    """df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "PREST", "ESTADO",
                                                            "DEMORA", "TVISITA", "PROCEDENCIA", "REALIZADA"])"""

    columns_to_expand = ["TVISITA", "SERVICIO"]
    count = 0
    actividad = actividad.drop_duplicates('CODIGO')

    for idx_actividad, cod_actividad in enumerate(actividad["CODIGO"]):
        df.loc[idx_actividad]["CODIGO"] = actividad.iloc[idx_actividad]["CODIGO"]
        df.loc[idx_actividad]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
        #df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]
        #df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        df.loc[idx_actividad]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
        df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
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
        if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
            df.loc[idx_actividad]["ESTADO"] = ""
        if actividad.iloc[idx_actividad]["REALIZADA"] == "U":
            df.loc[idx_actividad]["REALIZADA"] = ""
        else:
            df.loc[idx_actividad]["REALIZADA"] = actividad.iloc[idx_actividad]["REALIZADA"]

        count = count + 1
        #if count == 5000:
        #    break
    df = pd.DataFrame.drop(df, columns=["NHC"])
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


if __name__ == '__main__':
    #dataframe = preprocessing()
    path_to_excel = r'/Users/zeyna/Documents/TFG/dataframes/df_decision_trees_7.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    svm(dataframe)

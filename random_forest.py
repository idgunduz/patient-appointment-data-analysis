import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np

from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = 'datos/ACTIVIDAD.xlsx'
path_to_salidas = "datos/SALIDAS.xlsx"
path_to_scae = "datos/SCAE.xlsx"


def random_forest(df):
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)
    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, stratify=y_under,
                                                        test_size=0.2, random_state=7)
    classifier = RandomForestClassifier()
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test.astype(int))
    print(metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TN, FN, FP, TP = confusion_matrix(y_test.astype(int), y_pred.astype(int),
                                      labels=[1, 0]).reshape(-1)
    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')
    feature_importances_df = pd.DataFrame({"feature": list(df.columns),
                                           "importance": classifier.feature_importances_})\
                                           .sort_values("importance", ascending=False)

    sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
    plt.xlabel("Features")
    plt.ylabel("Feature Importance Score")
    plt.title("Visualizing Important Features")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize=10)
    plt.show()


def decision_tree(df):
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)
    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, stratify=y_under, test_size=0.2, random_state=7)
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test.astype(int))
    print(metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')


def preprocessing():

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_salidas)
        f2 = executor.submit(pd.read_excel, path_to_actividad)
        #f3 = executor.submit(pd.read_excel, path_to_scae)

        salidas = f1.result()
        actividad = f2.result()
        #scae = f3.result()

    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO"])

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')
    actividad = actividad.drop_duplicates('NHC_ENCRIPTADO')
    #scae = scae.drop_duplicates('NHC_ENCRIPTADO')

    for idx_actividad, nhc_actividad in enumerate(actividad["NHC_ENCRIPTADO"]):
        for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
            if nhc_actividad == nhc_salidas:
                df.loc[idx_salidas]["NHC"] = nhc_salidas
                df.loc[idx_salidas]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
                df.loc[idx_salidas]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
                df.loc[idx_salidas]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] -
                                                     actividad.iloc[idx_actividad]["FECHAGRABACION"])\
                                                     .total_seconds() / (60 * 60 * 24)

                if actividad.iloc[idx_actividad]["ANTERIOR"] is None \
                        or actividad.iloc[idx_actividad]["ANTERIOR"] != 0 \
                        or actividad.iloc[idx_actividad]["ANTERIOR"] == "":
                    df.loc[idx_salidas]["ANTERIOR"] = "S"
                else:
                    df.loc[idx_salidas]["ANTERIOR"] = "N"

                if salidas.iloc[idx_salidas]["TURNO"] != "N":
                    df.loc[idx_salidas]["TURNO"] = salidas.iloc[idx_salidas]["TURNO"]
                else:
                    df.loc[idx_salidas]["TURNO"] = None

                if salidas.iloc[idx_salidas]["GR_ETARIO"] == "IA":
                    df.loc[idx_salidas]["GR_ETARIO"] = "I"
                else:
                    df.loc[idx_salidas]["GR_ETARIO"] = salidas.iloc[idx_salidas]["GR_ETARIO"]

                if salidas.iloc[idx_salidas]["LIBRELEC"] == 0:
                    df.loc[idx_salidas]["LIBRELEC"] = 0
                else:
                    df.loc[idx_salidas]["LIBRELEC"] = 1

                if salidas.iloc[idx_salidas]["TIPENTR"] == 1:
                    df.loc[idx_salidas]["TIPENTR"] = 1
                else:
                    df.loc[idx_salidas]["TIPENTR"] = 0

                if salidas.iloc[idx_salidas]["ULTESP"] == 1:
                    df.loc[idx_salidas]["ULTESP"] = 1
                else:
                    df.loc[idx_salidas]["ULTESP"] = 0

                if salidas.iloc[idx_salidas]["CIRPRES"] == 2 or salidas.iloc[idx_salidas]["CIRPRES"] == 4:
                    df.loc[idx_salidas]["CIRPRES"] = 0
                elif salidas.iloc[idx_salidas]["CIRPRES"] == 1:
                    df.loc[idx_salidas]["CIRPRES"] = 1

                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0
                break

    """for idx_scae, nhc_scae in enumerate(scae["NHC_ENCRIPTADO"]):
        for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
            if nhc_scae == nhc_salidas: 
                df.loc[idx_salidas]["NHC"] = nhc_salidas
                #df.loc[idx_salidas]["PREST"] = scae.iloc[idx_scae]["Prestación"]
                #df.loc[idx_salidas]["PREFERENTE"] = scae.iloc[idx_scae]["Preferente"]
                df.loc[idx_salidas]["EDAD"] = (scae.iloc[idx_scae]["Fecha Cita"] - scae.iloc[idx_scae][
                    "Fecha de Nacimiento"]).total_seconds() / (60 * 60 * 24 * 365)
                if salidas.iloc[idx_salidas]["TURNO"] != "N":
                    df.loc[idx_salidas]["TURNO"] = salidas.iloc[idx_salidas]["TURNO"]
                else:
                    df.loc[idx_salidas]["TURNO"] = None

                #df.loc[idx_salidas]["GR_ETARIO"] = salidas.iloc[idx_salidas]["GR_ETARIO"]

                if salidas.iloc[idx_salidas]["LIBRELEC"] == 0:
                    df.loc[idx_salidas]["LIBRELEC"] = 0
                else:
                    df.loc[idx_salidas]["LIBRELEC"] = 1
                # df.loc[idx_salidas]["SOSPECHA"] = salidas.iloc[idx_salidas]["SOSPECHA"]
                if salidas.iloc[idx_salidas]["TIPENTR"] == 1:
                    df.loc[idx_salidas]["TIPENTR"] = 1
                else:
                    df.loc[idx_salidas]["TIPENTR"] = 0

                if salidas.iloc[idx_salidas]["ULTESP"] == 1:
                    df.loc[idx_salidas]["ULTESP"] = 1
                else:
                    df.loc[idx_salidas]["ULTESP"] = 0

                if salidas.iloc[idx_salidas]["CIRPRES"] == 2 or salidas.iloc[idx_salidas]["CIRPRES"] == 4:
                    df.loc[idx_salidas]["CIRPRES"] = 0
                elif salidas.iloc[idx_salidas]["CIRPRES"] == 1:
                    df.loc[idx_salidas]["CIRPRES"] = 1

                # df.loc[idx_salidas]["TIPPRES"] = salidas.iloc[idx_salidas]["TIPPRES"]

                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0
                break"""

    df = pd.DataFrame.drop(df, columns=["NHC"])
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['TIPSAL'])
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


def preprocessing2():
    salidas = pd.read_excel(r'datos/actividad_salidas.xlsx')

    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO"])

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
        df.loc[idx_salidas]["NHC"] = nhc_salidas
        df.loc[idx_salidas]["SERVICIO"] = salidas.iloc[idx_salidas]["SERVICIO"]
        df.loc[idx_salidas]["TVISITA"] = salidas.iloc[idx_salidas]["TVISITA"]
        df.loc[idx_salidas]["DELTA_DIAS"] = (salidas.iloc[idx_salidas]["FECHA"] -
                                             salidas.iloc[idx_salidas]["FECHAGRABACION"])\
                                             .total_seconds() / (60 * 60 * 24)

        if salidas.iloc[idx_salidas]["ANTERIOR"] is None \
                or salidas.iloc[idx_salidas]["ANTERIOR"] != 0 \
                or salidas.iloc[idx_salidas]["ANTERIOR"] == "":
            df.loc[idx_salidas]["ANTERIOR"] = "S"
        else:
            df.loc[idx_salidas]["ANTERIOR"] = "N"

        if salidas.iloc[idx_salidas]["TURNO"] != "N":
            df.loc[idx_salidas]["TURNO"] = salidas.iloc[idx_salidas]["TURNO"]
        else:
            df.loc[idx_salidas]["TURNO"] = None

        if salidas.iloc[idx_salidas]["GR_ETARIO"] == "IA":
            df.loc[idx_salidas]["GR_ETARIO"] = "I"
        else:
            df.loc[idx_salidas]["GR_ETARIO"] = salidas.iloc[idx_salidas]["GR_ETARIO"]

        if salidas.iloc[idx_salidas]["LIBRELEC"] == 0:
            df.loc[idx_salidas]["LIBRELEC"] = 0
        else:
            df.loc[idx_salidas]["LIBRELEC"] = 1

        if salidas.iloc[idx_salidas]["TIPENTR"] == 1:
            df.loc[idx_salidas]["TIPENTR"] = 1
        else:
            df.loc[idx_salidas]["TIPENTR"] = 0

        if salidas.iloc[idx_salidas]["ULTESP"] == 1:
            df.loc[idx_salidas]["ULTESP"] = 1
        else:
            df.loc[idx_salidas]["ULTESP"] = 0

        if salidas.iloc[idx_salidas]["CIRPRES"] == 2 or salidas.iloc[idx_salidas]["CIRPRES"] == 4:
            df.loc[idx_salidas]["CIRPRES"] = 0
        elif salidas.iloc[idx_salidas]["CIRPRES"] == 1:
            df.loc[idx_salidas]["CIRPRES"] = 1

        if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
            df.loc[idx_salidas]["TIPSAL"] = 1
        elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
            df.loc[idx_salidas]["TIPSAL"] = 0

    df = pd.DataFrame.drop(df, columns=["NHC"])
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['TIPSAL'])
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


if __name__ == '__main__':
    #dataframe = preprocessing()
    #dataframe = preprocessing2()
    #dataframe.to_excel(r'/Users/zeyna/Documents/TFG/dataframes/df12.xlsx', index=False)
    path_to_excel = r'dfs/df2.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    random_forest(dataframe)
    decision_tree(dataframe)
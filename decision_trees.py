import imblearn
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
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


path_to_actividad = '/Users/zeyna/Documents/TFG/actividad_con_nhc.xlsx'
path_to_salidas = "datos_falsos/SALIDAS.xlsx"
path_to_scae = "datos_falsos/SCAE.xlsx"
path_to_codigos = "datos_falsos/CODIGOS-ACTIVIDAD.xlsx"


def random_forest(df):
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)
    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer = median_imputer.fit(df)
    imputed_df = median_imputer.transform(df.values)
    df = pd.DataFrame(data=imputed_df, columns=df.columns)"""
    """x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)"""
    """label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)"""

    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, stratify=y_under, test_size=0.2, random_state=7)
    classifier = RandomForestClassifier()
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test.astype(int))
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
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
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)
    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer = median_imputer.fit(df)
    imputed_df = median_imputer.transform(df.values)
    df = pd.DataFrame(data=imputed_df, columns=df.columns)"""

    """label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)"""
    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, stratify=y_under, test_size=0.2, random_state=7)
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test.astype(int))
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
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
        #f4 = executor.submit(pd.read_excel, path_to_codigos)
        salidas = f1.result()
        actividad = f2.result()
        #scae = f3.result()
        #codigos = f4.result()

    #actividad = pd.concat([actividad, codigos], axis=1)
    #actividad = pd.read_excel(path_to_actividad, sheet_name='ACTIVIDAD')
    #actividad.to_excel(r'/Users/zeyna/Documents/TFG/actividad_con_nhc.xlsx', index=False)
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "SERVICIO", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                                                          "GR_ETARIO", "TIPSAL"])"""
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO"])
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", 
                                                             "LIBRELEC", "SERVICIO", "TURNO", "CIRPRES", 
                                                             "ULTESP", "TIPENTR", "GR_ETARIO", "TIPSAL",
                                                             "PREST", "PREFERENTE"])"""

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "PREST", "PREFERENTE"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')
    actividad = actividad.drop_duplicates('NHC_ENCRIPTADO')
    #scae = scae.drop_duplicates('NHC_ENCRIPTADO')

    for idx_actividad, nhc_actividad in enumerate(actividad["NHC_ENCRIPTADO"]):
        for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
            if nhc_actividad == nhc_salidas:
                df.loc[idx_salidas]["NHC"] = nhc_salidas
                df.loc[idx_salidas]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
                df.loc[idx_salidas]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
                #df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
                #df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
                #df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
                df.loc[idx_salidas]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] -
                                                     actividad.iloc[idx_actividad]["FECHAGRABACION"])\
                                                     .total_seconds() / (60 * 60 * 24)

                """if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
                    df.loc[idx_actividad]["ESTADO"] = None
                else:
                    df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]"""

                """elif actividad.iloc[idx_actividad]["REALIZADA"] == "S" and actividad.iloc[idx_actividad]["ESTADO"] == "CAP" \
                        and count != 40000:
                    df.loc[idx_actividad]["REALIZADA"] = "N"
                    count = count + 1"""

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
                #df.loc[idx_salidas]["SOSPECHA"] = salidas.iloc[idx_salidas]["SOSPECHA"]
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

                #df.loc[idx_salidas]["TIPPRES"] = salidas.iloc[idx_salidas]["TIPPRES"]

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

    #df = df.dropna()
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


def preprocessing2():
    salidas = pd.read_excel(r'/Users/zeyna/Documents/TFG/datos_falsos/actividad_salidas.xlsx')
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "SERVICIO", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                                                          "GR_ETARIO", "TIPSAL"])"""
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO"])
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", 
                                                             "LIBRELEC", "SERVICIO", "TURNO", "CIRPRES", 
                                                             "ULTESP", "TIPENTR", "GR_ETARIO", "TIPSAL",
                                                             "PREST", "PREFERENTE"])"""

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "PREST", "PREFERENTE"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
        df.loc[idx_salidas]["NHC"] = nhc_salidas
        df.loc[idx_salidas]["SERVICIO"] = salidas.iloc[idx_salidas]["SERVICIO"]
        df.loc[idx_salidas]["TVISITA"] = salidas.iloc[idx_salidas]["TVISITA"]
        #df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        #df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        #df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
        df.loc[idx_salidas]["DELTA_DIAS"] = (salidas.iloc[idx_salidas]["FECHA"] -
                                             salidas.iloc[idx_salidas]["FECHAGRABACION"])\
                                             .total_seconds() / (60 * 60 * 24)

        """if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
            df.loc[idx_actividad]["ESTADO"] = None
        else:
            df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]"""

        """elif actividad.iloc[idx_actividad]["REALIZADA"] == "S" and actividad.iloc[idx_actividad]["ESTADO"] == "CAP" \
                and count != 40000:
            df.loc[idx_actividad]["REALIZADA"] = "N"
            count = count + 1"""

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
        #df.loc[idx_salidas]["SOSPECHA"] = salidas.iloc[idx_salidas]["SOSPECHA"]
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

        #df.loc[idx_salidas]["TIPPRES"] = salidas.iloc[idx_salidas]["TIPPRES"]

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

    #df = df.dropna()
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


if __name__ == '__main__':
    #dataframe = preprocessing()
    #dataframe = preprocessing2()
    #dataframe.to_excel(r'/Users/zeyna/Documents/TFG/dataframes/df12.xlsx', index=False)
    path_to_excel = r'/Users/zeyna/Documents/TFG/dfs/df1.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    """columns_count = ["TIPPRES", "CIRPRES", "ULTESP", "TIPENTR", "LIBRELEC", "TVISITA_P", "TVISITA_S", "TVISITA_P",
                     "SERVICIO_GASB", "SERVICIO_GASC", "SERVICIO_GASE", "SERVICIO_GASH", "ANTERIOR_N", "ANTERIOR_S",
                     "SOSPECHA_N", "SOSPECHA_S", "TURNO_N", "TURNO_M", "TURNO_T", "GR_ETARIO_IA", "GR_ETARIO_A"]"""
    """columns_count = ["CIRPRES", "ULTESP", "TIPENTR", "LIBRELEC", "TVISITA_P", "TVISITA_S", "TVISITA_P",
                     "SERVICIO_GASB", "SERVICIO_GASC", "SERVICIO_GASE", "SERVICIO_GASH", "ANTERIOR_N", "ANTERIOR_S",
                     "TURNO_N", "TURNO_M", "TURNO_T", "GR_ETARIO_IA", "GR_ETARIO_A", "GR_ETARIO_I"]
    main_exploratory(columns_count, dataframe)"""
    random_forest(dataframe)
    decision_tree(dataframe)
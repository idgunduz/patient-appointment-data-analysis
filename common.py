import pandas as pd
import re

from sklearn.impute import SimpleImputer


path_to_file = "datos/DATOS_DIGESTIVO.xlsx"


def sort_two_lists_by_second(list1, list2):
    """
    Sorts two lists by the second one provided
    :param list1: list to order
    :param list2: list to order BY.
    :return: two lists ordered by the second one.
    """
    def get_key(item):
        return item[0]

    list2_sorted, list1_sorted = list(zip(*sorted(zip(list2, list1), key=get_key, reverse=True)))
    return list1_sorted, list2_sorted


def preprocess():
    import concurrent.futures
    import numpy as np

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="ACTIVIDAD")
        salidas = f1.result()
        actividad = f2.result()

    salidas = salidas.drop_duplicates('NHC')
    actividad = actividad.drop_duplicates('NHC')
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ULTESP", "TIPENTR", "SEXO", "EDAD",
                                                          "TIPSAL"])

    for idx_salidas, cipa_salidas in enumerate(salidas["CIPA"]):
        for idx_actividad, cipa_actividad in enumerate(actividad["CIPA"]):
            if cipa_salidas == cipa_actividad:
                df.loc[idx_salidas]["NHC"] = salidas.iloc[idx_salidas]["NHC"]
                if actividad.iloc[idx_actividad]["SEXO"] == "V":
                    df.loc[idx_salidas]["SEXO"] = 0
                if actividad.iloc[idx_actividad]["SEXO"] == "M":
                    df.loc[idx_salidas]["SEXO"] = 1
                df.loc[idx_salidas]["EDAD"] = (salidas.iloc[idx_salidas]["FC"] -
                                               actividad.iloc[idx_actividad]["FECHANAC"]) \
                                                  .total_seconds() / (60 * 60 * 24 * 365)
                df.loc[idx_salidas]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] -
                                                     salidas.iloc[idx_salidas]["FG"]) \
                                                        .total_seconds() / (60 * 60 * 24)
                if salidas.iloc[idx_salidas]["TIPENTR"] == 1:
                    df.loc[idx_salidas]["TIPENTR"] = 1
                else:
                    df.loc[idx_salidas]["TIPENTR"] = 0

                if salidas.iloc[idx_salidas]["ULTESP"] == 1:
                    df.loc[idx_salidas]["ULTESP"] = 1
                else:
                    df.loc[idx_salidas]["ULTESP"] = 0
                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0
                break

    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["TIPSAL"])
    """median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    median_imputer = median_imputer.fit(df)
    imputed_df = median_imputer.transform(df.values)
    df = pd.DataFrame(data=imputed_df, columns=df.columns)"""
    #df = pd.get_dummies(df, columns=columns_to_convert_to_numeric_value)
    # TODO: Delete columns that have low ocurrences

    return df


def load_and_preprocess(con_sexo=True, con_edad=True, con_distancia=True, sin_desconocidos=False,
                        agrupar_ausencias=True, dummify=True, sin_NHC=True):
    """
    This function pre-processes the actividad dataframe from DATOS_DIGESTIVO appending data from scae dataframe
    :return: salidas: dataframe with the preprocessed data.
    """

    import concurrent.futures
    import numpy as np
    import pickle

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="ACTIVIDAD")
        salidas = f1.result()
        actividad = f2.result()

    labels_to_drop = ["NUMICU", "CIPA", "SERVIC_AUT", "SERVIC", "SERV_LOCAL", "TIPPRES", "CIRPRES", "FI", "FV",
                      "FH", "FH_AGENDA", "FS", "AGENDA", "FC", "FG", "FNAC", "COD_PRESTA", "MES_CITA", "PRESTA_LOCAL",
                      "DIA_SEMANA", "SEMANA_MES"]

    if sin_NHC:
        labels_to_drop.append("NHC")

    actividad = actividad.drop_duplicates('NHC')
    salidas = salidas.loc[salidas.COD_PRESTA == "CPRIMERA"]
    sexo_index_salidas = None
    sexo_index_actividad = None
    fnac_index_salidas = None
    fnac_index_actividad = None

    if con_sexo:
        salidas["SEXO"] = ""
        sexo_index_salidas = salidas.columns.get_loc("SEXO")
        sexo_index_actividad = actividad.columns.get_loc("SEXO")
    if con_edad:
        salidas["FNAC"] = salidas["FC"]
        fnac_index_salidas = salidas.columns.get_loc("FNAC")
        fnac_index_actividad = actividad.columns.get_loc("FECHANAC")
    """if con_distancia:
        salidas["DIST"] = 0
        cod_post_index_salidas = salidas.columns.get_loc("DIST")
        cod_post_index_actividad = actividad.columns.get_loc("CODPOST")"""

    if con_edad | con_sexo | con_distancia:
        for idx_salidas, cipa_salidas in enumerate(salidas["CIPA"]):
            for idx_actividad, cipa_actividad in enumerate(actividad["CIPA"]):
                if cipa_salidas == cipa_actividad:
                    if con_sexo:
                        salidas.iloc[idx_salidas, sexo_index_salidas] = actividad.iloc[idx_actividad,
                                                                                       sexo_index_actividad]
                    if con_edad:
                        salidas.iloc[idx_salidas, fnac_index_salidas] = actividad.iloc[idx_actividad,
                                                                                       fnac_index_actividad]
                    """if con_distancia:
                        cod_paciente = actividad.iloc[idx_actividad, cod_post_index_actividad]
                        if cod_paciente:
                            if not pd.isna(cod_paciente):
                                cod_paciente = "{:05d}".format(int(cod_paciente))
                                dist = 0
                                with open("codes.dictionary", "rb") as f:
                                    cod_dict = pickle.load(f)
                                if cod_paciente in cod_dict:
                                    dist = np.float32(cod_dict[cod_paciente])
                                if (dist is np.nan) | (dist is np.inf) | (dist is -np.inf):
                                    dist = 0
                                salidas.iloc[idx_salidas, cod_post_index_salidas] = int(dist)"""

                    break

    salidas["MES_CITA"] = salidas["FC"].dt.month
    salidas["SEMANA_MES"] = get_week_of_month(salidas["FC"])
    salidas = salidas.loc[salidas["COD_PRESTA"] == "CPRIMERA"]
    salidas["DIA_SEMANA"] = salidas["FC"].dt.dayofweek
    salidas["DELTA_DIAS"] = (salidas["FC"] - salidas["FG"]).dt.total_seconds() / (60 * 60 * 24)
    salidas["EDAD"] = (salidas["FC"] - salidas["FNAC"]).dt.total_seconds() / (60 * 60 * 24 * 365)

    salidas = pd.DataFrame.drop(salidas, columns=labels_to_drop)
    salidas.replace([np.inf, -np.inf], np.nan)
    salidas = pd.DataFrame.dropna(salidas, 0)
    salidas.loc[salidas["TIPENTR"] == 1, "TIPENTR"] = 0
    salidas.loc[salidas["TIPENTR"] != 0, "TIPENTR"] = 1
    salidas.loc[salidas["ULTESP"] == 1, "ULTESP"] = 0
    salidas.loc[salidas["ULTESP"] != 0, "ULTESP"] = 1
    salidas1 = salidas.loc[salidas["TIPSAL"] == 1]
    salidas5 = salidas.loc[salidas["TIPSAL"] == 5]
    salidas6 = salidas.loc[salidas["TIPSAL"] == 6]
    salidas7 = salidas.loc[salidas["TIPSAL"] == 7]
    salidas10 = salidas.loc[salidas["TIPSAL"] == 10]
    salidas_faltas = [salidas5, salidas6, salidas7, salidas10]
    salidas_faltas = pd.concat(salidas_faltas)
    if agrupar_ausencias:
        salidas_faltas["TIPSAL"] = 5
    salidas = pd.concat([salidas1, salidas_faltas])

    return salidas


def get_week(s):
    prev_week = (s - pd.to_timedelta(7, unit='d')).dt.week
    return (
        s.dt.week
            .where((s.dt.month != 1) | (s.dt.week < 50), 0)
            .where((s.dt.month != 12) | (s.dt.week > 1), prev_week + 1)
    )


def get_week_of_month(s):
    first_day_of_month = s - pd.to_timedelta(s.dt.day - 1, unit='d')
    first_week_of_month = get_week(first_day_of_month)
    current_week = get_week(s)
    return current_week - first_week_of_month


def get_words(content):
    """
    Gets all unique words of a string without stopwords or symbols.
    :param content: str, text to process
    :return: list of str, list of unique words in content
    """

    to_substitute = [(r"\(-\)", "negativo"), (r"\.", " "), (",", " "), (r"/", " "), (r"\|", " "),
                     ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"), ("ñ", "n"), (r"\(", " "),
                     (r"\)", " "), (r"\*", " "), (r"-", " "), (r"\bca\b", "cancer")]

    to_delete = ("ruego", r"\bdel\b", r"\bha\b", "pido", "valoracion", "solicito", r"\b[a-zA-Z]\b", "saludo",
                 "gracias", "remito", "deriva ", "derivo", "rogamos", "recomiendo", r"\bun\b",
                 r"\bde\b", r"\bpara\b", r"\bpor\b", r"\bla\b", r"\bni\b", r"\be.\b", r"\bno\b", r"\bhacia\b",
                 r"[\d\-\+\*\?\"\:\<\>]", r"\bbien\b", r"\bcon\b", r"\banos\b", r"\bse\b", r"\bhace\b", r"\bque\b",
                 r"\bsin\b", r"\bya\b", r"\bpaciente\b", r"\bdesde\b", r"\brefiere\b", r"\blos\b", r"\bmeses\b",
                 r"\bpeso\b", r"\bl.\b", r"\bmes\b", r"\bmas\b", r"\bb\b", r"\bef\b", r"\bap\b", r"\bsi\b", r"\btras\b",
                 r"\baf\b", r"\bvalora\b", r"\best.\b", r"\bal\b", r"\bpresenta\b", r"\bpero\b", r"\bdx\b", r"\bhan\b",
                 r"\b\vuestra\b", r"\bveo\b", r"\buna\b", r"\bvalorar\b", r"\bver\b")

    content = content.lower()
    for subs in to_substitute:
        content = re.sub(subs[0], subs[1], content)

    for dels in to_delete:
        content = re.sub(dels, "", content)

    word_list = [x for x in content.split(" ") if x]
    content = list(dict.fromkeys(word_list))

    return content


def score(y_true, y_pred, *y_score):
    """
    Calculate various score metrics and print them to console
    :param y_true: true label of the classification
    :param y_pred: predicted label
    :param y_score: probabilities of predictions. Used to calculate the AUC for the ROC
    :return: None
    """
    from sklearn.metrics import precision_score, multilabel_confusion_matrix, accuracy_score, roc_auc_score

    precision = precision_score(y_true, y_pred, average=None)
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    if y_score:
        y_score = y_score[0][:, 1]
        roc_auc = roc_auc_score(y_true, y_score, average=None, multi_class='ovo')

    for i in range(0, len(matrices)):
        matrix = matrices[i]
        (TN, FP), (FN, TP) = matrix
        specificity = TN / (TN + FP)
        sensitivity = TP / (TP + FN)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        print("")
        print(matrix)
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")
        print(f"Precision: {precision[i]}")
        print(f"PPV: {PPV}")
        print(f"NPV: {NPV}")
    print(" ")
    if y_score.any():
        print(f"AUC: {roc_auc}")
    print(f"Accuracy: {accuracy}")



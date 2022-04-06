import matplotlib as mpl
import numpy as np
import fasttext as ft
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.under_sampling import NearMiss
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = 'datos/ACTIVIDAD.xlsx'
path_to_salidas = "datos/SALIDAS.xlsx"
path_to_scae = "datos/SCAE.xlsx"
path_to_file = "datos/DATOS_DIGESTIVO.xlsx"


def word_frequency_model():

    import concurrent.futures
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import Levenshtein as Lvn

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="SCAE")
        salidas = f1.result()
        scae = f2.result()

    observations = scae["Observaciones PIC"]

    words, frequencies, word_freq = create_word_count(observations)
    words, _ = sort_two_lists_by_second(words, frequencies)
    words_to_use = words[0:50]

    from wordcloud import WordCloud

    # Generate word cloud
    wc = WordCloud(width=400, height=330, max_words=100, background_color='white').generate_from_frequencies(word_freq)
    """
    #Create word cloud
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    """

    scae = scae.drop_duplicates('NHC')
    salidas = salidas.loc[salidas.COD_PRESTA == "CPRIMERA"]
    labels_to_drop = []
    for element in salidas.columns:
        if (element != 'NHC') & (element != 'TIPSAL'):
            labels_to_drop.append(element)

    salidas = pd.DataFrame.drop(salidas, columns=labels_to_drop)
    salidas = pd.DataFrame.dropna(salidas, 0)

    for i in range(0, len(words_to_use)):
        salidas[words_to_use[i]] = 0

    words_dataframe_index = []
    column_names = salidas.columns
    for word in words_to_use:
        for idx, name in enumerate(column_names):
            if word == name:
                words_dataframe_index.append(idx)
                break

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_salidas == (nhc_scae.upper()):
                if not pd.isnull(observations[idx_scae]):
                    patient_words = get_words(observations[idx_scae])
                    for word in patient_words:
                        for iw, word_t_u in enumerate(words_to_use):
                            if Lvn.distance(word, word_t_u) <= 1:
                                salidas.iloc[idx_salidas, words_dataframe_index[iw]] = 1
                                break
                    break

    salidas = pd.DataFrame.drop(salidas, columns=["NHC"])

    salidas1 = salidas.loc[salidas["TIPSAL"] == 1]
    salidas5 = salidas.loc[salidas["TIPSAL"] == 5]
    salidas6 = salidas.loc[salidas["TIPSAL"] == 6]
    salidas7 = salidas.loc[salidas["TIPSAL"] == 7]
    salidas10 = salidas.loc[salidas["TIPSAL"] == 10]
    salidas_faltas = [salidas5, salidas6, salidas7, salidas10]
    salidas_faltas = pd.concat(salidas_faltas)
    salidas_faltas["TIPSAL"] = 5
    salidas = pd.concat([salidas1, salidas_faltas])

    y = salidas["TIPSAL"]
    x = pd.DataFrame.drop(salidas, columns=["TIPSAL"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=27)

    from sklearn.naive_bayes import GaussianNB

    # Initialize GaussianNB classifier
    model = GaussianNB()
    # Fit the model on the train dataset
    model = model.fit(x_train, y_train)
    # Make predictions on the test dataset
    y_score = model.predict_proba(x_test)
    y_pred = model.predict(x_test)
    print("Word frequency model with Gaussian")
    score(y_test, y_pred, y_score)

    clf = DecisionTreeClassifier(random_state=0, class_weight='balanced', ccp_alpha=0.002)
    clf.fit(x_train, y_train)
    y_score = clf.predict_proba(x_test)
    y_pred = clf.predict(x_test)
    print("Word frequency model")
    score(y_test, y_pred, y_score)

"""
def get_subwords(sent):
    tokenizer = WordPunctTokenizer()
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(sent)]
"""


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


def create_word_count(array, dist=1):
    """
    Creates a word count from an array of strings, accounting for possible misspelling by Levenshtein distance.
    :param array: iterable containing strings, corpus from which to build the word count
    :param dist: int, threshold of Levenshtein distance to consider two words are the same
    :return: returns two ordered lists with the unique words without stopwords and their frequencies
    """
    import Levenshtein as Lvn

    words = {}
    for content in array:
        if not pd.isna(content):
            content = get_words(content)
            for word in content:
                if words:
                    if word in words:
                        words[word] += 1
                    else:
                        for key in words.keys():
                            distance = Lvn.distance(word, key)
                            if distance == dist:
                                words[key] += 1
                                break
                        else:
                            words[word] = 1
                else:
                    words[word] = 1
    return list(words.keys()), list(words.values()), words


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


def word_embeddings():

    model = ft.load_model(r"/Users/zeyna/Documents/TFG/modelo_lenguaje/Scielo_wiki_FastText300.bin")
    salidas_medical = load_and_preprocess(sin_desconocidos=True, sin_NHC=False)

    scae = pd.read_excel(path_to_file, sheet_name="SCAE")
    observations = scae["Observaciones PIC"]

    embed_index = []
    for i in range(0, 50):
        salidas_medical[f"c{i}"] = 0
        embed_index.append(list(salidas_medical).index(f"c{i}"))
    all_sentence_embeddings = np.empty((0, 300), dtype=np.float32)

    for idx_salidas, nhc_salidas in enumerate(salidas_medical["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_salidas == (nhc_scae.upper()):
                if not pd.isnull(observations[idx_scae]):
                    patient_words = get_words(observations[idx_scae])
                    vector = np.zeros((1, 300))
                    for word in patient_words:
                        embedded_word = model.get_word_vector(word)
                        is_empty = embedded_word == np.zeros((1, 300))
                        if not is_empty.all():
                            vector += embedded_word
                    all_sentence_embeddings = np.append(all_sentence_embeddings, vector, axis=0)
                    break

    """
    import io
    # Write out the embedding vectors and metadata
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    for word_num in range(0, 50):
        word = words[word_num]
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in all_sentence_embeddings]) + "\n")
    out_v.close()
    out_m.close()
    """

    pca = PCA(n_components=50)
    pca.fit(all_sentence_embeddings)
    empty_description_idx = []

    for idx_salidas, nhc_salidas in enumerate(salidas_medical["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_salidas == (nhc_scae.upper()):
                if not pd.isnull(observations[idx_scae]):
                    patient_words = get_words(observations[idx_scae])
                    vector = np.zeros((1, 300))
                    for word in patient_words:
                        embedded_word = model.get_word_vector(word)
                        if embedded_word.any():
                            vector += embedded_word
                    is_empty = vector == np.zeros((1, 300))
                    if not vector.all():
                        empty_description_idx.append(idx_salidas)
                    else:
                        vector = pca.transform(vector)
                    for ind, element in enumerate(vector[0]):
                        salidas_medical.iloc[idx_salidas, embed_index[ind]] = element
                    break

    salidas_medical.drop(empty_description_idx)
    salidas_medical = pd.DataFrame.drop(salidas_medical, columns=["NHC"])
    salidas_medical = pd.get_dummies(salidas_medical, columns=["SEXO"])
    salidas_medical = pd.DataFrame.drop(salidas_medical, columns=["SEXO_"])
    salidas1 = salidas_medical.loc[salidas_medical["TIPSAL"] == 1]
    salidas5 = salidas_medical.loc[salidas_medical["TIPSAL"] == 5]
    salidas6 = salidas_medical.loc[salidas_medical["TIPSAL"] == 6]
    salidas7 = salidas_medical.loc[salidas_medical["TIPSAL"] == 7]
    salidas10 = salidas_medical.loc[salidas_medical["TIPSAL"] == 10]
    salidas_faltas = [salidas5, salidas6, salidas7, salidas10]
    salidas_faltas = pd.concat(salidas_faltas)
    salidas_faltas["TIPSAL"] = 0
    salidas_medical = pd.concat([salidas1, salidas_faltas])
    y = salidas_medical["TIPSAL"]
    x = pd.DataFrame.drop(salidas_medical, columns=["TIPSAL"])
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print(metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))
    """TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')"""
    feature_importances_df = pd.DataFrame({"feature": list(x.columns),
                                           "importance": classifier.feature_importances_})\
                                            .sort_values("importance", ascending=False)
    sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="large")
    plt.show()

    """clf2 = DecisionTreeClassifier(random_state=0, class_weight='balanced', ccp_alpha=0.002)
    clf2.fit(x_train, y_train)
    y_score = clf2.predict_proba(x_test)
    y_pred = clf2.predict(x_test)
    print("Embedding model")
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')"""


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


if __name__ == '__main__':
    """dataframe = preprocessing2()
    dataframe.to_excel(r'/Users/zeyna/Documents/TFG/dataframes/df12.xlsx', index=False)"""
    path_to_excel = r'dfs/df2.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    decision_tree(dataframe)
    word_frequency_model()

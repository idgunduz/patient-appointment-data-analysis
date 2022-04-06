import fasttext as ft
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics

path_to_file = "datos/DATOS_DIGESTIVO.xlsx"


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
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ULTESP", "TIPENTR",
                                                          "SEXO", "EDAD", "TIPSAL"])

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
                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or \
                        salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or \
                        salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or \
                        salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or \
                        salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or \
                        salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0
                break

    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["TIPSAL"])

    return df


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


def main():

    path_model = r"/Users/zeyna/Documents/TFG/modelo_lenguaje/Scielo_wiki_FastText300.bin"
    model = ft.load_model(path_model)

    df = preprocess()

    scae = pd.read_excel(path_to_file, sheet_name="SCAE")
    observations = scae["Observaciones PIC"]

    embed_index = []
    for i in range(0, 50):
        df[f"c{i}"] = 0
        embed_index.append(list(df).index(f"c{i}"))
    all_sentence_embeddings = np.empty((0, 300), dtype=np.float32)

    for idx_df, nhc_df in enumerate(df["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_df == (nhc_scae.upper()):
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

    for idx_df, nhc_df in enumerate(df["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_df == (nhc_scae.upper()):
                if not pd.isnull(observations[idx_scae]):
                    patient_words = get_words(observations[idx_scae])
                    vector = np.zeros((1, 300))
                    for word in patient_words:
                        embedded_word = model.get_word_vector(word)
                        if embedded_word.any():
                            vector += embedded_word
                    is_empty = vector == np.zeros((1, 300))
                    if not vector.all():
                        empty_description_idx.append(idx_df)
                    else:
                        vector = pca.transform(vector)
                    for ind, element in enumerate(vector[0]):
                        df.iloc[idx_df, embed_index[ind]] = element
                    break

    df.drop(empty_description_idx)
    df = df.reset_index(drop=True)
    df = pd.DataFrame.drop(df, columns=["NHC"])
    #df.to_excel(r'dfs/df_observaciones_1.xlsx', index=False)

    y = df["TIPSAL"]
    x = pd.DataFrame.drop(df, columns=["TIPSAL"])
    counter = Counter(y)
    print(counter)

    oversample = SMOTE()
    x, y = oversample.fit_resample(x.astype(int), y.astype(int))
    counter = Counter(y)
    print(counter)

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2,
                                                        random_state=0)
    print(f"x_test: {len(x_test)}")
    print(f"x_train: {len(x_train)}")
    print(f"y_test: {len(y_test)}")
    print(f"y_train: {len(y_train)}")
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train.astype(int), y_train.astype(int))

    y_pred = classifier.predict(x_test)
    print(metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    print(confusion_matrix(y_test.astype(int), y_pred.astype(int)))

    feature_importances_df = pd.DataFrame(
        {"feature": list(x.columns), "importance": classifier.feature_importances_}).sort_values("importance",
                                                                                                 ascending=False)
    sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="large")
    plt.show()


if __name__ == "__main__":
    main()

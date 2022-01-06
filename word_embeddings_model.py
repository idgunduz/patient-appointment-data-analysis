import fasttext as ft
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA

path_to_file = "/Users/zeyna/Documents/TFG/ListasDeEsperaV1/DATOS_DIGESTIVO.xlsx"


def medical_model():

    from common import load_and_preprocess, get_words, score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

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

    salidas1 = salidas_medical.loc[salidas_medical["TIPSAL"] == 1]
    salidas5 = salidas_medical.loc[salidas_medical["TIPSAL"] == 5]
    salidas6 = salidas_medical.loc[salidas_medical["TIPSAL"] == 6]
    salidas7 = salidas_medical.loc[salidas_medical["TIPSAL"] == 7]
    salidas10 = salidas_medical.loc[salidas_medical["TIPSAL"] == 10]
    salidas_faltas = [salidas5, salidas6, salidas7, salidas10]
    salidas_faltas = pd.concat(salidas_faltas)
    salidas_faltas["TIPSAL"] = 5
    salidas_medical = pd.concat([salidas1, salidas_faltas])

    y = salidas_medical["TIPSAL"]
    x = pd.DataFrame.drop(salidas_medical, columns=["TIPSAL"])
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

    clf2 = DecisionTreeClassifier(random_state=0, class_weight='balanced', ccp_alpha=0.002)
    clf2.fit(x_train, y_train)
    y_score = clf2.predict_proba(x_test)
    y_pred = clf2.predict(x_test)
    print("Embedding model")
    accuracy = accuracy_score(y_test, y_pred)
    score(y_test, y_pred, y_score)
    print(accuracy)


if __name__ == "__main__":
    medical_model()

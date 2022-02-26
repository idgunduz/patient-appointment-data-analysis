import fasttext as ft
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics


path_to_file = "datos/DATOS_DIGESTIVO.xlsx"


def medical_model():

    from common import load_and_preprocess, get_words

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
    salidas_medical.to_excel(r'/Users/zeyna/Documents/TFG/df2.xlsx', index=True)
    y = salidas_medical["TIPSAL"]
    x = pd.DataFrame.drop(salidas_medical, columns=["TIPSAL"])
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')
    feature_importances_df = pd.DataFrame(
        {"feature": list(x.columns), "importance": classifier.feature_importances_}).sort_values("importance",
                                                                                                 ascending=False)
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


def medical_model2():

    from common import preprocess, get_words

    model = ft.load_model(r"/Users/zeyna/Documents/TFG/modelo_lenguaje/Scielo_wiki_FastText300.bin")
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

    salidas1 = df.loc[df["TIPSAL"] == 1]
    salidas5 = df.loc[df["TIPSAL"] == 5]
    salidas6 = df.loc[df["TIPSAL"] == 6]
    salidas7 = df.loc[df["TIPSAL"] == 7]
    salidas10 = df.loc[df["TIPSAL"] == 10]
    salidas_faltas = [salidas5, salidas6, salidas7, salidas10]
    salidas_faltas = pd.concat(salidas_faltas)
    salidas_faltas["TIPSAL"] = 0
    df = pd.concat([salidas1, salidas_faltas])
    df.to_excel(r'/Users/zeyna/Documents/TFG/df.xlsx', index=True)
    y = df["TIPSAL"]
    x = pd.DataFrame.drop(df, columns=["TIPSAL"])
    print(x.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    TP, FN, FP, TN = confusion_matrix(y_test.astype(int), y_pred.astype(int), labels=[1, 0]).reshape(-1)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    print(f'Specificidad: {specificity}')
    print(f'Sensibilidad: {sensitivity}')
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
    medical_model()

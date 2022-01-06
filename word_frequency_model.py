import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

path_to_file = "/Users/zeyna/Documents/TFG/DATOS_DIGESTIVO.xlsx"


def medical_model():

    import concurrent.futures
    from common import sort_two_lists_by_second, score, get_words
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

def create_word_count(array, dist=1):
    """
    Creates a word count from an array of strings, accounting for possible misspelling by Levenshtein distance.
    :param array: iterable containing strings, corpus from which to build the word count
    :param dist: int, threshold of Levenshtein distance to consider two words are the same
    :return: returns two ordered lists with the unique words without stopwords and their frequencies
    """
    import Levenshtein as Lvn
    from common import get_words

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

if __name__ == "__main__":
    medical_model()

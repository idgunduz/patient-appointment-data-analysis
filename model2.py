import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import concurrent.futures
from common import score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

path_to_file = "/Users/zeyna/Documents/TFG/ListasDeEsperaV1/DATOS_DIGESTIVO.xlsx"

def model2():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SCAE")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="ACTIVIDAD")
        f3 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        scae = f1.result()
        actividad = f2.result()
        salidas = f3.result()

    actividad = actividad.drop_duplicates('NHC')
    scae = scae.drop_duplicates('NHC')

    """ Se ha borrado el nombre del campo PROCEDENCIA """
    df = pd.DataFrame(index = range(len(salidas)), columns = ["NHC", "SEXO", "SOSPECHA", "PREFERENTE", "CIRPRES", "TIPSAL"])

    for idx_salidas, cipa_salidas in enumerate(salidas["CIPA"]):
        for idx_actividad, cipa_actividad in enumerate(actividad["CIPA"]):
            if cipa_salidas == cipa_actividad:
                df.loc[idx_salidas]["NHC"] = salidas.iloc[idx_salidas]["NHC"]
                """ df.loc[idx_salidas]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"] """
                if actividad.iloc[idx_actividad]["SEXO"] == 'M':
                    df.loc[idx_salidas]["SEXO"] = 0
                else:
                    df.loc[idx_salidas]["SEXO"] = 1

                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                if salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0

                df.loc[idx_salidas]["CIRPRES"] = salidas.iloc[idx_salidas]["CIRPRES"]

    """ df = df.dropna(subset=['PROCEDENCIA']) """
    df = df.dropna(subset=['SEXO'])
    df = df.dropna(subset=['TIPSAL'])
    df = df.reset_index(drop=True)

    for idx_df, nhc_df in enumerate(df["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_df == (nhc_scae.upper()):
                if scae.iloc[idx_scae]["Sospecha Malignidad"] == "NO":
                    df.loc[idx_df]["SOSPECHA"] = 0
                else:
                    df.loc[idx_df]["SOSPECHA"] = 1

                if scae.iloc[idx_scae]["Preferente"] == "NO":
                    df.loc[idx_df]["PREFERENTE"] = 0
                else:
                    df.loc[idx_df]["PREFERENTE"] = 1

    df = df.dropna(subset=['SOSPECHA'])
    df = df.dropna(subset=['PREFERENTE'])
    df = df.reset_index(drop=True)

    df = pd.DataFrame.drop(df, columns=["NHC"])

    y = df["TIPSAL"]
    x = pd.DataFrame.drop(df, columns=["TIPSAL"])

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(x_train.astype(int), y_train.astype(int))
    y_score = clf2.predict_proba(x_test)
    y_pred = clf2.predict(x_test)
    print("Model 2")
    accuracy = accuracy_score(y_test.astype(int), y_pred.astype(int))
    print(accuracy)
    score(y_test.astype(int), y_pred.astype(int), y_score.astype(int))



if __name__ == "__main__":
    model2()
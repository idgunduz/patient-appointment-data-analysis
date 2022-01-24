import pandas as pd
import seaborn as sns
import concurrent.futures
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


path_to_file = "DATOS_DIGESTIVO.xlsx"


def main():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SCAE")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="ACTIVIDAD")
        f3 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        scae = f1.result()
        actividad = f2.result()
        salidas = f3.result()

    actividad = actividad.drop_duplicates('NHC')
    scae = scae.drop_duplicates('NHC')

    """ Se ha borrado el nombre del campo PROCEDENCIA, ULTESP y "SOSPECHA """
    df = pd.DataFrame(index = range(len(salidas)), columns = ["NHC", "SEXO", "PREFERENTE", "CIRPRES",
                                                              "EDAD", "DELTA_DIAS", "TIPENTR", "ULTESP", "TIPSAL"])

    columns_to_expand = ["SEXO", "PREFERENTE", "CIRPRES"]
    for idx_salidas, cipa_salidas in enumerate(salidas["CIPA"]):
        for idx_actividad, cipa_actividad in enumerate(actividad["CIPA"]):
            if cipa_salidas == cipa_actividad:
                df.loc[idx_salidas]["NHC"] = salidas.iloc[idx_salidas]["NHC"]
                """ df.loc[idx_salidas]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"] """
                df.loc[idx_salidas]["SEXO"] = actividad.iloc[idx_actividad]["SEXO"]
                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                if salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0

                df.loc[idx_salidas]["CIRPRES"] = salidas.iloc[idx_salidas]["CIRPRES"]
                df.loc[idx_salidas]["DELTA_DIAS"] = (salidas.iloc[idx_salidas]["FC"] - salidas.iloc[idx_salidas]["FG"]).total_seconds() / (60 * 60 * 24)
                #df.loc[salidas["ULTESP"] == 1, "ULTESP"] = 0
                #df.loc[salidas["ULTESP"] != 0, "ULTESP"] = 1
                df.loc[idx_salidas]["TIPENTR"] = salidas.iloc[idx_salidas]["TIPENTR"]
                df.loc[idx_salidas]["ULTESP"] = salidas.iloc[idx_salidas]["ULTESP"]


    """ df = df.dropna(subset=['PROCEDENCIA']) """
    df = df.dropna(subset=['SEXO'])
    df = df.dropna(subset=['TIPSAL'])
    df = df.dropna(subset=['DELTA_DIAS'])
    df = df.dropna(subset=['ULTESP'])
    df = df.dropna(subset=['TIPENTR'])
    df = df.reset_index(drop=True)

    for idx_df, nhc_df in enumerate(df["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_df == (nhc_scae.upper()):
                #if scae.iloc[idx_scae]["Sospecha Malignidad"] == "NO":
                #    df.loc[idx_df]["SOSPECHA"] = 0
                #else:
                #    df.loc[idx_df]["SOSPECHA"] = 1

                df.loc[idx_df]["PREFERENTE"] = scae.iloc[idx_scae]["Preferente"]
                df.loc[idx_df]["EDAD"] = (scae.iloc[idx_scae]["Fecha Cita"] - scae.iloc[idx_scae]["Fecha de Nacimiento"]).total_seconds() / (60 * 60 * 24 * 365)

    #df = df.dropna(subset=['SOSPECHA'])
    df = df.dropna(subset=['PREFERENTE'])
    df = df.dropna(subset=['EDAD'])
    df = df.reset_index(drop=True)
    df = pd.get_dummies(df, columns=columns_to_expand)
    df = pd.DataFrame.drop(df, columns=["NHC"])

    y = df["TIPSAL"]
    x = pd.DataFrame.drop(df, columns=["TIPSAL"])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, stratify=y, test_size=0.10, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100)
    classifier.fit(x_train.astype(int), y_train.astype(int))
    y_pred = classifier.predict(x_test)
    print("Accuracy:", metrics.classification_report(y_test.astype(int), y_pred.astype(int)))
    print("Accuracy:", metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))
    feature_importances_df = pd.DataFrame(
        {"feature": list(x.columns), "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    sns.barplot(x=feature_importances_df.feature, y=feature_importances_df.importance)
    plt.xlabel("Feature Importance Score")
    plt.ylabel("Features")
    plt.title("Visualizing Important Features")
    plt.xticks(rotation=45, horizontalalignment="right", fontweight="light", fontsize="large")
    plt.show()


if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from common import *

path_to_file = "datos/DATOS_DIGESTIVO.xlsx"


def main_exploratory():

    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_file, sheet_name="SCAE")
        f2 = executor.submit(pd.read_excel, path_to_file, sheet_name="ACTIVIDAD")
        f3 = executor.submit(pd.read_excel, path_to_file, sheet_name="SALIDAS")
        scae = f1.result()
        actividad = f2.result()
        salidas = f3.result()

    actividad = actividad.drop_duplicates('NHC')
    scae = scae.drop_duplicates('NHC')

    df_sexo = pd.DataFrame(index = range(len(salidas)), columns = ["SEXO", "TIPSAL"])
    df_proc = pd.DataFrame(index = range(len(salidas)), columns = ["PROCEDENCIA", "TIPSAL"])
    for idx_salidas, cipa_salidas in enumerate(salidas["CIPA"]):
        for idx_actividad, cipa_actividad in enumerate(actividad["CIPA"]):
            if cipa_salidas == cipa_actividad:
                df_proc.loc[idx_salidas]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
                df_sexo.loc[idx_salidas]["SEXO"] = actividad.iloc[idx_actividad]["SEXO"]
                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                    or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                    or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17 :
                        df_sexo.loc[idx_salidas]["TIPSAL"] = "Presentados"
                        df_proc.loc[idx_salidas]["TIPSAL"] = "Presentados"
                if salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                    or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15 :
                        df_sexo.loc[idx_salidas]["TIPSAL"] = "No Presentados"
                        df_proc.loc[idx_salidas]["TIPSAL"] = "No Presentados"

    df_proc = df_proc.dropna(subset=['PROCEDENCIA'])
    df_sexo = df_sexo.dropna(subset=['SEXO'])
    df_proc = df_proc.dropna(subset=['TIPSAL'])
    df_sexo = df_sexo.dropna(subset=['TIPSAL'])
    df_sexo = df_sexo.reset_index(drop=True)
    df_proc = df_proc.reset_index(drop=True)

    df_sosp = pd.DataFrame(index=range(len(salidas)), columns=["SOSPECHA", "TIPSAL"])
    df_pref = pd.DataFrame(index=range(len(salidas)), columns=["PREFERENTE", "TIPSAL"])

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC"]):
        for idx_scae, nhc_scae in enumerate(scae["NHC"]):
            if nhc_salidas == (nhc_scae.upper()):
                df_sosp.loc[idx_salidas]["SOSPECHA"] = scae.iloc[idx_scae]["Sospecha Malignidad"]
                df_pref.loc[idx_salidas]["PREFERENTE"] = scae.iloc[idx_scae]["Preferente"]
                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df_sosp.loc[idx_salidas]["TIPSAL"] = "Presentados"
                    df_pref.loc[idx_salidas]["TIPSAL"] = "Presentados"
                if salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df_sosp.loc[idx_salidas]["TIPSAL"] = "No Presentados"
                    df_pref.loc[idx_salidas]["TIPSAL"] = "No Presentados"

    df_sosp = df_sosp.dropna(subset=['SOSPECHA'])
    df_pref = df_pref.dropna(subset=['PREFERENTE'])
    df_pref = df_pref.dropna(subset=['TIPSAL'])
    df_sosp = df_sosp.dropna(subset=['TIPSAL'])
    df_sosp = df_sosp.reset_index(drop=True)
    df_pref = df_pref.reset_index(drop=True)

    df_salidas = pd.DataFrame(index=range(len(salidas)), columns=["CIRPRES", "TIPSAL"])
    for idx_salidas, nhc_salidas in enumerate(salidas["NHC"]):
        df_salidas.loc[idx_salidas]["CIRPRES"] = salidas.iloc[idx_salidas]["CIRPRES"]
        if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
            df_salidas.loc[idx_salidas]["TIPSAL"] = "Presentados"
        if salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
            df_salidas.loc[idx_salidas]["TIPSAL"] = "No Presentados"

    df_salidas = df_salidas.dropna(subset=['TIPSAL'])
    df_salidas = df_salidas.dropna(subset=['CIRPRES'])
    df_salidas = df_salidas.reset_index(drop=True)

    sns.countplot(x="SEXO", hue="TIPSAL", data=df_sexo).set(title='SEXO', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PROCEDENCIA", hue="TIPSAL", data=df_proc).set(title='PROCEDENCIA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SOSPECHA", hue="TIPSAL", data=df_sosp).set(title='SOSPECHA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREFERENTE", hue="TIPSAL", data=df_pref).set(title='PREFERENTE', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="CIRPRES", hue="TIPSAL", data=df_salidas).set(title='CIRPRES', ylabel='Total Pacientes')
    plt.show()

    print("ppppp")


if __name__ == "__main__":
    main_exploratory()
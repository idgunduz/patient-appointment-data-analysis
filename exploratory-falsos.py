import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from common import *
from datetime import datetime
path_to_actividad = "datos_falsos/DATOS-ACTIVIDAD.xlsx"


def main_exploratory():
    actividad = pd.read_excel(path_to_actividad, sheet_name='ACTIVIDAD')

    df = pd.DataFrame(index=range(len(actividad)), columns=["CODIGO", "SERVICIO", "ESTADO", "DELTA_DIAS", "DEMORA",
                                                            "PREST", "TVISITA", "PROCEDENCIA", "TURNO", "ANTERIOR",
                                                            "REALIZADA"])
    actividad = actividad.drop_duplicates('CODIGO')

    """df_servicio = pd.DataFrame(index=range(len(actividad)), columns=["SERVICIO", "REALIZADA"])
    df_proc = pd.DataFrame(index=range(len(actividad)), columns=["PROCEDENCIA", "REALIZADA"])
    df_estado = pd.DataFrame(index=range(len(actividad)), columns=["ESTADO", "REALIZADA"])
    df_delta_dias = pd.DataFrame(index=range(len(actividad)), columns=["DELTA_DIAS", "REALIZADA"])
    df_demora = pd.DataFrame(index=range(len(actividad)), columns=["DEMORA", "REALIZADA"])
    df_prest = pd.DataFrame(index=range(len(actividad)), columns=["PREST", "REALIZADA"])
    df_tvisita= pd.DataFrame(index=range(len(actividad)), columns=["TVISITA", "REALIZADA"])
    df_turno = pd.DataFrame(index=range(len(actividad)), columns=["TURNO", "REALIZADA"])
    df_anterior = pd.DataFrame(index=range(len(actividad)), columns=["ANTERIOR", "REALIZADA"])"""

    for idx_actividad, codigo_actividad in enumerate(actividad["CODIGO"]):
        df.loc[idx_actividad]["CODIGO"] = actividad.iloc[idx_actividad]["CODIGO"]
        df.loc[idx_actividad]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
        df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]
        df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        df.loc[idx_actividad]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
        df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
        df.loc[idx_actividad]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] - actividad.iloc[idx_actividad]
                                              ["FECHAGRABACION"]).total_seconds() / (60 * 60 * 24)
        time = datetime.strptime("15:00:00", '%H:%M:%S')
        if len(actividad.iloc[idx_actividad]["HORAINI"].split(":")) == 2:
            appointment_time_formatted = actividad.iloc[idx_actividad]["HORAINI"] + ":00"
        else:
            appointment_time_formatted = actividad.iloc[idx_actividad]["HORAINI"]
        appointment_time = datetime.strptime(appointment_time_formatted, '%H:%M:%S')
        if appointment_time >= time:
            df.loc[idx_actividad]["TURNO"] = 1
        else:
            df.loc[idx_actividad]["TURNO"] = 0
        if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
            df.loc[idx_actividad]["ESTADO"] = ""
        if actividad.iloc[idx_actividad]["REALIZADA"] == "U":
            df.loc[idx_actividad]["REALIZADA"] = ""
        elif actividad.iloc[idx_actividad]["REALIZADA"] == "S":
            df.loc[idx_actividad]["REALIZADA"] = "Presentados"
        else:
            df.loc[idx_actividad]["REALIZADA"] = "No Presentados"

    df = df.dropna(subset=['PROCEDENCIA'])
    df = df.dropna(subset=['SERVICIO'])
    df = df.dropna(subset=['ESTADO'])
    df = df.dropna(subset=['DELTA_DIAS'])
    df = df.dropna(subset=['DEMORA'])
    df = df.dropna(subset=['PREST'])
    df = df.dropna(subset=['TVISITA'])
    df = df.dropna(subset=['TURNO'])
    df = df.dropna(subset=['ANTERIOR'])
    df = df.dropna(subset=['REALIZADA'])

    df = df.reset_index(drop=True)

    """df_proc.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
    df_servicio.loc[idx_actividad]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
    df_estado.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]
    df_delta_dias.loc[idx_actividad]["DELTA_DIAS"] = actividad.iloc[idx_actividad]["DELTA_DIAS"]
    df_demora.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
    df_prest.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
    df_tvisita.loc[idx_actividad]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
    df_turno.loc[idx_actividad]["TURNO"] = actividad.iloc[idx_actividad]["TURNO"]
    df_anterior.loc[idx_actividad]["ANTERIOR"] = actividad.iloc[idx_actividad]["ANTERIOR"]"""

    """df_proc = df_proc.dropna(subset=['PROCEDENCIA'])
    df_servicio = df_servicio.dropna(subset=['SERVICIO'])
    df_estado= df_estado.dropna(subset=['ESTADO'])
    df_delta_dias = df_delta_dias.dropna(subset=['DELTA_DIAS'])
    df_demora = df_demora.dropna(subset=['DEMORA'])
    df_prest = df_prest.dropna(subset=['PREST'])
    df_tvisita= df_tvisita.dropna(subset=['TVISITA'])
    df_turno = df_turno.dropna(subset=['TURNO'])
    df_anterior = df_anterior.dropna(subset=['ANTERIOR'])

    df_proc = df_proc.reset_index(drop=True)
    df_prest = df_prest.reset_index(drop=True)
    df_servicio = df_servicio.reset_index(drop=True)
    df_estado = df_estado.reset_index(drop=True)
    df_demora = df_demora.reset_index(drop=True)
    df_delta_dias = df_delta_dias.reset_index(drop=True)
    df_tvisita = df_tvisita.reset_index(drop=True)
    df_turno = df_turno.reset_index(drop=True)
    df_anterior= df_anterior.reset_index(drop=True)"""

    sns.countplot(x="SERVICIO", hue="REALIZADA", data=df).set(title='SERVICIO', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PROCEDENCIA", hue="REALIZADA", data=df).set(title='PROCEDENCIA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST", hue="REALIZADA", data=df).set(title='PREST', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="DEMORA", hue="REALIZADA", data=df).set(title='DEMORA', ylabel='Total Pacientes')
    plt.show()

    """sns.countplot(x="DELTA_DIAS", hue="REALIZADA", data=df_pref).set(title='DELTA_DIAS', ylabel='Total Pacientes')
    plt.show()"""

    sns.countplot(x="TVISITA", hue="REALIZADA", data=df).set(title='TVISITA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TURNO", hue="REALIZADA", data=df).set(title='TURNO', ylabel='Total Pacientes')
    plt.show()

    print("ppppp")


if __name__ == "__main__":
    main_exploratory()
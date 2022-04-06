import matplotlib as mpl
import concurrent.futures
import numpy as np
import pandas as pd

from exploratory import main_exploratory

mpl.rcParams.update(mpl.rcParamsDefault)

path_to_actividad = 'datos/ACTIVIDAD.xlsx'
path_to_salidas = "datos/SALIDAS.xlsx"
path_to_scae = "datos/SCAE.xlsx"
path_to_codigos = "datos/CODIGOS-ACTIVIDAD.xlsx"
path_to_datos_actividad = "datos/DATOS-ACTIVIDAD.xlsx"


def concat_actividad_datos_y_codigos_files():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_codigos)
        f2 = executor.submit(pd.read_excel, path_to_datos_actividad)
        codigos_actividad = f1.result()
        datos_actividad = f2.result()

    actividad_con_nhc = pd.concat([datos_actividad, codigos_actividad], axis=1)
    actividad_con_nhc.to_excel(r'datos/ACTIVIDAD.xlsx', index=False)


def concat_data_files():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_salidas)
        f2 = executor.submit(pd.read_excel, path_to_actividad)
        f3 = executor.submit(pd.read_excel, path_to_scae)

        salidas = f1.result()
        actividad = f2.result()
        scae = f3.result()

    dataframes = pd.concat([actividad, salidas], axis=1)
    dataframes.to_excel(r'datos/actividad_salidas.xlsx', index=False)
    actividad_salidas = pd.read_excel(r'datos/actividad_salidas.xlsx')
    return actividad_salidas


def iterate_concat_data_files():
    salidas = pd.read_excel(r'datos/actividad_salidas.xlsx')

    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO", "SEXO", "RANGOEDAD"])

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "SEXO", "RANGOEDAD"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
        df.loc[idx_salidas]["NHC"] = nhc_salidas
        df.loc[idx_salidas]["SEXO"] = salidas.iloc[idx_salidas]["SEXO"]
        df.loc[idx_salidas]["RANGOEDAD"] = salidas.iloc[idx_salidas]["RANGOEDAD"]
        df.loc[idx_salidas]["SERVICIO"] = salidas.iloc[idx_salidas]["SERVICIO"]
        df.loc[idx_salidas]["TVISITA"] = salidas.iloc[idx_salidas]["TVISITA"]
        df.loc[idx_salidas]["DELTA_DIAS"] = (salidas.iloc[idx_salidas]["FECHA"] -
                                             salidas.iloc[idx_salidas]["FECHAGRABACION"]) \
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

        if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
            df.loc[idx_salidas]["TIPSAL"] = 1
        elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
            df.loc[idx_salidas]["TIPSAL"] = 0

    df = pd.DataFrame.drop(df, columns=["NHC"])
    df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['TIPSAL'])

    # df = df.dropna()
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


def iterate_data_files():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_salidas)
        f2 = executor.submit(pd.read_excel, path_to_actividad)

        salidas = f1.result()
        actividad = f2.result()

    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO", "SEXO", "RANGOEDAD"])

    columns_to_expand = ["TVISITA", "SERVICIO", "TURNO", "GR_ETARIO", "RANGOEDAD", "SEXO"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')
    actividad = actividad.drop_duplicates('NHC_ENCRIPTADO')
    #scae = scae.drop_duplicates('NHC_ENCRIPTADO')

    for idx_actividad, nhc_actividad in enumerate(actividad["NHC_ENCRIPTADO"]):
        for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
            if nhc_actividad == nhc_salidas:
                df.loc[idx_salidas]["NHC"] = nhc_salidas
                df.loc[idx_salidas]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
                df.loc[idx_salidas]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
                df.loc[idx_salidas]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] -
                                                     actividad.iloc[idx_actividad]["FECHAGRABACION"])\
                                                     .total_seconds() / (60 * 60 * 24)

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

                df.loc[idx_salidas]["SEXO"] = salidas.iloc[idx_salidas]["SEXO"]
                df.loc[idx_salidas]["RANGOEDAD"] = salidas.iloc[idx_salidas]["RANGOEDAD"]

                if salidas.iloc[idx_salidas]["TIPSAL"] == 1 or salidas.iloc[idx_salidas]["TIPSAL"] == 2 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 3 or salidas.iloc[idx_salidas]["TIPSAL"] == 12 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 16 or salidas.iloc[idx_salidas]["TIPSAL"] == 17:
                    df.loc[idx_salidas]["TIPSAL"] = 1
                elif salidas.iloc[idx_salidas]["TIPSAL"] == 4 or salidas.iloc[idx_salidas]["TIPSAL"] == 5 \
                        or salidas.iloc[idx_salidas]["TIPSAL"] == 6 or salidas.iloc[idx_salidas]["TIPSAL"] == 15:
                    df.loc[idx_salidas]["TIPSAL"] = 0
                break

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


if __name__ == '__main__':
    #concat_actividad_datos_y_codigos_files()
    #concat_data_files()
    #iterate_concat_data_files()
    """dataframe = iterate_data_files()
    dataframe.to_excel(r'dfs/df2.xlsx', index=False)"""

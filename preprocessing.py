import matplotlib as mpl
import concurrent.futures
import numpy as np

from common import *
from exploratory2 import main_exploratory

mpl.rcParams.update(mpl.rcParamsDefault)


path_to_actividad = 'datos_falsos/ACTIVIDAD.xlsx'
path_to_salidas = "DatosFalsos/SALIDAS.xlsx"
path_to_scae = "DatosFalsos/scae.xlsx"
path_to_codigos = "datos_falsos/CODIGOS-ACTIVIDAD.xlsx"


def concat_data_files():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        f1 = executor.submit(pd.read_excel, path_to_salidas)
        f2 = executor.submit(pd.read_excel, path_to_actividad)
        #f3 = executor.submit(pd.read_excel, path_to_scae)
        f4 = executor.submit(pd.read_excel, path_to_codigos)
        salidas = f1.result()
        actividad = f2.result()
        #scae = f3.result()
        codigos = f4.result()

    actividad = pd.concat([actividad, codigos], axis=1)
    actividad_con_nhc = actividad.to_excel(r'/Users/zeyna/Documents/TFG/Datos Falsos/actividad_con_nhc.xlsx', index=False)
    dataframes = pd.concat([actividad_con_nhc, salidas], axis=1)
    actividad_salidas = dataframes.to_excel(r'/Users/zeyna/Documents/TFG/Datos Falsos/actividad_salidas.xlsx', index=False)
    return actividad_salidas


def iterate_concat_data_files():
    salidas = pd.read_excel(r'/Users/zeyna/Documents/TFG/Datos Falsos/actividad_salidas.xlsx')
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "SERVICIO", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                                                          "GR_ETARIO", "TIPSAL"])"""
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO", "SEXO", "RANGOEDAD"])
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", 
                                                             "LIBRELEC", "SERVICIO", "TURNO", "CIRPRES", 
                                                             "ULTESP", "TIPENTR", "GR_ETARIO", "TIPSAL",
                                                             "PREST", "PREFERENTE"])"""

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "SEXO", "RANGOEDAD"]
    # columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO"]
    # columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "PREST", "PREFERENTE"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')

    for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
        df.loc[idx_salidas]["NHC"] = nhc_salidas
        df.loc[idx_salidas]["SEXO"] = salidas.iloc[idx_salidas]["SEXO"]
        df.loc[idx_salidas]["RANGOEDAD"] = salidas.iloc[idx_salidas]["RANGOEDAD"]
        df.loc[idx_salidas]["SERVICIO"] = salidas.iloc[idx_salidas]["SERVICIO"]
        df.loc[idx_salidas]["TVISITA"] = salidas.iloc[idx_salidas]["TVISITA"]
        # df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
        # df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
        # df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
        df.loc[idx_salidas]["DELTA_DIAS"] = (salidas.iloc[idx_salidas]["FECHA"] -
                                             salidas.iloc[idx_salidas]["FECHAGRABACION"]) \
                                                .total_seconds() / (60 * 60 * 24)

        """if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
            df.loc[idx_actividad]["ESTADO"] = None
        else:
            df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]"""

        """elif actividad.iloc[idx_actividad]["REALIZADA"] == "S" and actividad.iloc[idx_actividad]["ESTADO"] == "CAP" \
                and count != 40000:
            df.loc[idx_actividad]["REALIZADA"] = "N"
            count = count + 1"""

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

        # df.loc[idx_salidas]["TIPPRES"] = salidas.iloc[idx_salidas]["TIPPRES"]

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
        #f2 = executor.submit(pd.read_excel, path_to_actividad)
        #f3 = executor.submit(pd.read_excel, path_to_scae)
        #f4 = executor.submit(pd.read_excel, path_to_codigos)
        salidas = f1.result()
        #actividad = f2.result()
        #scae = f3.result()
        #codigos = f4.result()

    #actividad = pd.concat([actividad, codigos], axis=1)
    actividad = pd.read_excel("DatosFalsos/actividad_con_nhc.xlsx")
    #actividad.to_excel(r'/Users/zeyna/Documents/TFG/repo/tfg/patient-appointment-data-analysis/DatosFalsos/actividad_con_nhc.xlsx',
    #                   index=False)
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "SERVICIO", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                                                          "GR_ETARIO", "TIPSAL"])"""
    df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", "LIBRELEC",
                                                          "TURNO", "CIRPRES", "ULTESP", "TIPENTR", "SERVICIO",
                                                          "TIPSAL", "GR_ETARIO", "SEXO", "RANGOEDAD"])
    """df = pd.DataFrame(index=range(len(salidas)), columns=["NHC", "DELTA_DIAS", "ANTERIOR", "TVISITA", 
                                                             "LIBRELEC", "SERVICIO", "TURNO", "CIRPRES", 
                                                             "ULTESP", "TIPENTR", "GR_ETARIO", "TIPSAL",
                                                             "PREST", "PREFERENTE"])"""

    columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "RANGOEDAD", "SEXO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO"]
    #columns_to_expand = ["TVISITA", "ANTERIOR", "SERVICIO", "TURNO", "GR_ETARIO", "PREST", "PREFERENTE"]
    salidas = salidas.drop_duplicates('NHC_ENCRIPTADO')
    actividad = actividad.drop_duplicates('NHC_ENCRIPTADO')
    #scae = scae.drop_duplicates('NHC_ENCRIPTADO')

    for idx_actividad, nhc_actividad in enumerate(actividad["NHC_ENCRIPTADO"]):
        for idx_salidas, nhc_salidas in enumerate(salidas["NHC_ENCRIPTADO"]):
            if nhc_actividad == nhc_salidas:
                df.loc[idx_salidas]["NHC"] = nhc_salidas
                df.loc[idx_salidas]["SERVICIO"] = actividad.iloc[idx_actividad]["SERVICIO"]
                df.loc[idx_salidas]["TVISITA"] = actividad.iloc[idx_actividad]["TVISITA"]
                #df.loc[idx_actividad]["PREST"] = actividad.iloc[idx_actividad]["PREST"]
                #df.loc[idx_actividad]["DEMORA"] = actividad.iloc[idx_actividad]["DEMORA"]
                #df.loc[idx_actividad]["PROCEDENCIA"] = actividad.iloc[idx_actividad]["PROCEDENCIA"]
                df.loc[idx_salidas]["DELTA_DIAS"] = (actividad.iloc[idx_actividad]["FECHA"] -
                                                     actividad.iloc[idx_actividad]["FECHAGRABACION"])\
                                                     .total_seconds() / (60 * 60 * 24)

                """if actividad.iloc[idx_actividad]["ESTADO"] != "CAP":
                    df.loc[idx_actividad]["ESTADO"] = None
                else:
                    df.loc[idx_actividad]["ESTADO"] = actividad.iloc[idx_actividad]["ESTADO"]"""

                """elif actividad.iloc[idx_actividad]["REALIZADA"] == "S" and actividad.iloc[idx_actividad]["ESTADO"] == "CAP" \
                        and count != 40000:
                    df.loc[idx_actividad]["REALIZADA"] = "N"
                    count = count + 1"""

                if actividad.iloc[idx_actividad]["ANTERIOR"] is None \
                        or actividad.iloc[idx_actividad]["ANTERIOR"] != 0 \
                        or actividad.iloc[idx_actividad]["ANTERIOR"] == "":
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
                #df.loc[idx_salidas]["SOSPECHA"] = salidas.iloc[idx_salidas]["SOSPECHA"]
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

                #df.loc[idx_salidas]["TIPPRES"] = salidas.iloc[idx_salidas]["TIPPRES"]
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
    df.to_excel(r'/Users/zeyna/Documents/TFG/dfs/df_to_explore.xlsx', index=False)
    #df = df.dropna()
    df = pd.get_dummies(df, columns=columns_to_expand)

    return df


if __name__ == '__main__':
    #concat_data_files()
    #iterate_concat_data_files()
    #dataframe = iterate_data_files()
    #dataframe.to_excel(r'/Users/zeyna/Documents/TFG/dfs/df1.xlsx', index=False)
    #iterate_data_files()
    path_to_excel = r'/Users/zeyna/Documents/TFG/dfs/df_to_explore.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    columns_count = ["ANTERIOR", "TVISITA", "LIBRELEC", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                     "SERVICIO", "GR_ETARIO", "SEXO", "RANGOEDAD"]
    main_exploratory(columns_count, dataframe)